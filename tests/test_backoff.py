"""Паузы вместо банов: кто виноват, сколько не трогать, кого пробовать последним."""

from __future__ import annotations

import unittest

import httpx

from llm_rotator import (
    DEFAULT_BACKOFF_BASE,
    DEFAULT_BACKOFF_MAX,
    ConfigError,
    GatewayError,
    SessionState,
    key_target,
    load_backoff,
    proxy_target,
    route_target,
)
from tests.test_registry import KEY_ERROR, StubGateway, fetch, make_registry, resp

NETWORK_DENIED = {
    "error": {"message": "Access denied. Please check your network settings."}
}


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def state_with_clock(**kwargs) -> tuple[SessionState, Clock]:
    clock = Clock()
    return SessionState(clock=clock, **kwargs), clock


class PenaltyTests(unittest.TestCase):
    def test_pause_doubles_up_to_the_ceiling(self):
        state, _ = state_with_clock(backoff_base=120, backoff_max=600)
        target = proxy_target("p1")
        pauses = [state.penalize(target) for _ in range(5)]
        self.assertEqual(pauses, [120, 240, 480, 600, 600])

    def test_success_starts_over_from_the_minimum(self):
        state, clock = state_with_clock(backoff_base=120)
        target = key_target("K1")
        state.penalize(target)
        state.penalize(target)
        state.forgive(target)
        self.assertFalse(state.blocked(target))
        self.assertEqual(state.penalize(target), 120)

    def test_blocked_then_doubtful_after_the_pause(self):
        state, clock = state_with_clock(backoff_base=120)
        target = route_target("p1", "groq")
        state.penalize(target)
        self.assertTrue(state.blocked(target))
        self.assertFalse(state.doubtful(target))
        clock.now += 120
        self.assertFalse(state.blocked(target))
        self.assertTrue(state.doubtful(target))

    def test_defaults(self):
        state = SessionState()
        self.assertEqual(state.backoff_base, DEFAULT_BACKOFF_BASE)
        self.assertEqual(state.backoff_max, DEFAULT_BACKOFF_MAX)


class GroqStoryTests(unittest.TestCase):
    """Живой случай: Groq не пускает адрес, а ротатор считал виноватым ключ."""

    def registry(self):
        return make_registry(
            proxies=("p1", "p2", "p3"),
            key_proxies=(("K2", ("p1", "p2")), ("K3", ("p2", "p3"))),
        )

    def test_key_survives_and_the_address_is_closed_for_this_provider(self):
        reg = self.registry()
        state, _ = state_with_clock()
        gw = StubGateway(
            reg,
            [resp(403, NETWORK_DENIED), resp(403, NETWORK_DENIED), resp(200, {})],
            state=state,
        )
        fetch(gw, reg)
        # K2 через p2 после p1 — не «следующий ключ»: ключ жив.
        self.assertEqual(gw.calls, ["K2@p1", "K2@p2", "K3@p3"])
        self.assertEqual(state.bad_keys, set())
        self.assertEqual(state.bad_proxies, set())
        self.assertEqual(state.blocked_routes, {("p1", "p"), ("p2", "p")})

    def test_the_same_closed_address_is_not_tried_with_another_key(self):
        reg = self.registry()
        state, _ = state_with_clock()
        state.penalize(route_target("p2", "p"))
        gw = StubGateway(reg, [resp(403, NETWORK_DENIED), resp(200, {})], state=state)
        fetch(gw, reg)
        self.assertEqual(gw.calls, ["K2@p1", "K3@p3"])


class ScopeTests(unittest.TestCase):
    def two_providers(self):
        return make_registry(
            proxies=("p1",),
            key_proxies=(("K1", ("p1",)), ("Q1", ("p1",))),
            extra_providers={"q": "https://api.q/v1"},
            model_providers=("p", "q"),
            key_owner={"Q1": "q"},
        )

    def test_a_closed_address_still_works_for_another_provider(self):
        reg = self.two_providers()
        gw = StubGateway(reg, [resp(403, NETWORK_DENIED), resp(200, {})])
        fetch(gw, reg)
        self.assertEqual(gw.calls, ["K1@p1", "Q1@p1"])

    def test_a_dead_proxy_is_dead_for_everyone(self):
        reg = self.two_providers()
        gw = StubGateway(reg, [httpx.ConnectError("refused")])
        with self.assertRaises(GatewayError):
            fetch(gw, reg)
        self.assertEqual(gw.calls, ["K1@p1"])
        self.assertEqual(gw.state.bad_proxies, {"p1"})

    def test_key_refusal_is_on_the_key(self):
        reg = make_registry(key_proxies=(("K1", ("p1", "p2")), ("K2", ("p2",))))
        gw = StubGateway(reg, [resp(401, KEY_ERROR), resp(200, {})])
        fetch(gw, reg)
        self.assertEqual(gw.calls, ["K1@p1", "K2@p2"])
        self.assertEqual(gw.state.bad_keys, {"K1"})


class RecoveryTests(unittest.TestCase):
    def test_paused_comes_back_after_the_pause_and_is_tried_last(self):
        reg = make_registry(key_proxies=(("K1", ("p1", "p2")),))
        state, clock = state_with_clock(backoff_base=120)
        gw = StubGateway(
            reg,
            [httpx.ConnectError("refused"), resp(200, {}), resp(200, {})],
            state=state,
        )
        fetch(gw, reg)
        self.assertEqual(gw.calls, ["K1@p1", "K1@p2"])
        # Пауза кончилась: p1 снова в деле, но после надёжного p2.
        clock.now += 121
        fetch(gw, reg)
        self.assertEqual(gw.calls[-1], "K1@p2")

    def test_doubtful_is_tried_when_nothing_else_is_left(self):
        reg = make_registry(key_proxies=(("K1", ("p1",)),))
        state, clock = state_with_clock(backoff_base=120)
        gw = StubGateway(
            reg,
            [
                httpx.ConnectError("refused"),
                httpx.ConnectError("refused"),
                resp(200, {}),
            ],
            state=state,
        )
        with self.assertRaises(GatewayError):
            fetch(gw, reg)
        clock.now += 121
        with self.assertRaises(GatewayError):
            fetch(gw, reg)
        # Второй отказ подряд — пауза вдвое длиннее.
        self.assertAlmostEqual(state.left(proxy_target("p1")), 240)
        clock.now += 241
        fetch(gw, reg)
        # Сработало — пауза сброшена до минимальной.
        self.assertEqual(state.penalties, {})
        self.assertEqual(state.penalize(proxy_target("p1")), 120)

    def test_success_forgives_key_address_and_route(self):
        reg = make_registry(key_proxies=(("K1", ("p1",)),))
        state, clock = state_with_clock(backoff_base=10)
        for target in (key_target("K1"), proxy_target("p1"), route_target("p1", "p")):
            state.penalize(target)
        clock.now += 11
        gw = StubGateway(reg, [resp(200, {})], state=state)
        fetch(gw, reg)
        self.assertEqual(state.penalties, {})

    def test_hint_names_what_waits_and_for_how_long(self):
        state, _ = state_with_clock(backoff_base=120)
        state.penalize(key_target("K1"))
        state.penalize(proxy_target("p1"))
        state.penalize(route_target("p2", "groq"))
        hint = state.retry_hint()
        self.assertIn("K1 ещё 120 с", hint)
        self.assertIn("p1 ещё 120 с", hint)
        self.assertIn("p2 для groq ещё 120 с", hint)


class SettingsTests(unittest.TestCase):
    def test_env(self):
        self.assertEqual(load_backoff({}), (None, None))
        self.assertEqual(
            load_backoff({"LLM_BACKOFF_BASE": "60", "LLM_BACKOFF_MAX": "7200"}),
            (60.0, 7200.0),
        )
        for env in (
            {"LLM_BACKOFF_BASE": "много"},
            {"LLM_BACKOFF_BASE": "0"},
            {"LLM_BACKOFF_BASE": "600", "LLM_BACKOFF_MAX": "60"},
        ):
            with self.subTest(env=env), self.assertRaises(ConfigError):
                load_backoff(env)

    def test_gateway_takes_the_settings_from_the_registry(self):
        reg = make_registry()
        reg.backoff_base, reg.backoff_max = 30.0, 90.0
        gw = StubGateway(reg, [])
        self.assertEqual((gw.state.backoff_base, gw.state.backoff_max), (30.0, 90.0))


if __name__ == "__main__":
    unittest.main()
