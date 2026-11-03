# Фикстуры DeepSeek Web

Обезличенные образцы ответов, собранные **вручную** по формату из
`docs/PLAN-web-transports.md` (раздел 1), без живого захвата. Токенов, cookie
и идентификаторов реальных чатов здесь нет. Нужны тестам разбора SSE и
классификации ошибок (этапы 2–3).

| Файл | Что показывает |
|---|---|
| `completion_basic.sse` | обычный ответ: стартовый патч + дозапись `content` + короткие `{"v":...}` |
| `completion_think.sse` | фрагмент THINK, затем RESPONSE через `APPEND` в `response/fragments` |
| `completion_backend_busy.sse` | HTTP 200, `event: hint`, `finish_reason: backend_busy` |
| `completion_too_many_messages.sse` | HTTP 200, лимит аккаунта (текст про «too many messages») |
| `completion_incomplete.sse` | `response/status = INCOMPLETE` — обрезано по длине |
| `error_invalid_token.json` | тело `code 40003` при HTTP 200 — токен протух |
| `error_invalid_chat_session.json` | `biz_code 1 "invalid chat session id"` |

Формат `rate_limit` и точный текст «too many messages» в живую не снимались —
при первом живом прогоне (этап 6/8, с разрешения) сверить и поправить.
