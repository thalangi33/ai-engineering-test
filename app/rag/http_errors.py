"""Shared HTTP error mapping for embedding and chat providers."""

from typing import NoReturn

import httpx


def raise_http_error(exc: httpx.HTTPError, what: str) -> NoReturn:
    if isinstance(exc, httpx.HTTPStatusError):
        detail = exc.response.text.strip() or exc.response.reason_phrase
        raise RuntimeError(
            f"{what} failed ({exc.response.status_code}): {detail}"
        ) from exc
    raise RuntimeError(f"{what} failed: {exc}") from exc
