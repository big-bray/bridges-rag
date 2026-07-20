"""HTTP client for scraping the Bridges archive."""

from __future__ import annotations

import time
from types import TracebackType

import httpx

USER_AGENT = "bridges-rag research scraper"


class Scraper:
    def __init__(self, delay: float = 1.0, timeout: float = 30.0) -> None:
        self._delay = delay
        self._last_request: float | None = None
        self._client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        )

    def get(self, url: str) -> httpx.Response:
        self._throttle()
        response = self._client.get(url)
        response.raise_for_status()
        self._last_request = time.monotonic()
        return response

    def _throttle(self) -> None:
        if self._last_request is not None:
            elapsed = time.monotonic() - self._last_request
            remaining = self._delay - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Scraper:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
