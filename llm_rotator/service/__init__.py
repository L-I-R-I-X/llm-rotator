"""Режим сервиса: OpenAI-совместимый HTTP-эндпоинт поверх `ModelRunner`.

Нужны extras `[server]` (starlette, uvicorn). Само ядро ротации от них не
зависит; импорт `llm_rotator.service.app` без них даст понятную ошибку.
"""
