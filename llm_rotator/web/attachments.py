"""Вложения хода: текст, не влезший в сообщение, и картинки.

Файл принадлежит аккаунту, на котором загружен, поэтому грузится через связку
(не при планировании). Планировщик только решает, что приложить.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .fingerprint import is_system

MAX_IMAGES = 4
MAX_IMAGE_BYTES = 7 * 1024 * 1024
IMAGE_MIMES = re.compile(r"^image/(?:png|jpe?g|webp|gif)$", re.IGNORECASE)
_DATA_URL = re.compile(r"^data:([\w.+-]+/[\w.+-]+)?((?:;[^;,]*)*),([\s\S]*)$")

HIDDEN_IMAGE = "[Image attached; not visible through this proxy]"


@dataclass(frozen=True, slots=True)
class Attachment:
    name: str
    data: bytes
    mime: str = "text/plain"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


def text_attachment(name: str, text: str) -> Attachment:
    return Attachment(name, text.encode("utf-8"), "text/plain")


def image_of_part(part: object) -> Attachment | None:
    """Картинка из части сообщения; `None` для ссылок, SVG и слишком больших."""
    if not isinstance(part, Mapping):
        return None
    mime = ""
    payload = ""
    if part.get("type") == "image":
        source = part.get("source")
        if isinstance(source, Mapping) and source.get("type") == "base64":
            mime, payload = (
                str(source.get("media_type") or ""),
                str(source.get("data") or ""),
            )
    elif part.get("type") in ("image_url", "input_image"):
        image = part.get("image_url")
        url = (
            image.get("url") if isinstance(image, Mapping) else image or part.get("url")
        )
        match = _DATA_URL.match(url) if isinstance(url, str) else None
        if match and re.search(r";base64", match.group(2), re.IGNORECASE):
            mime, payload = match.group(1) or "image/png", match.group(3)
    if not payload or not IMAGE_MIMES.match(mime):
        return None
    try:
        data = base64.b64decode(re.sub(r"\s+", "", payload), validate=False)
    except (binascii.Error, ValueError):
        return None
    if not data or len(data) > MAX_IMAGE_BYTES:
        return None
    digest = hashlib.sha256(data).hexdigest()[:12]
    extension = re.sub(r"\W", "", mime.split("/")[1])[:8] or "png"
    return Attachment(f"image-{digest}.{extension}", data, mime)


def collect_images(
    messages: Sequence[Mapping], limit: int = MAX_IMAGES
) -> tuple[Attachment, ...]:
    """Картинки сообщений по порядку, без повторов; остаются самые свежие."""
    images: list[Attachment] = []
    for message in messages:
        content = message.get("content")
        if is_system(message) or not isinstance(content, list):
            continue
        for part in content:
            image = image_of_part(part)
            if image is None:
                continue
            images = [i for i in images if i.name != image.name]  # повтор — в конец
            images.append(image)
    return tuple(images[-limit:])
