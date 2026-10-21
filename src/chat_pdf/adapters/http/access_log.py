"""Access-log filter: the UI polls a few GETs every second or two; keep their successes out."""

import logging
import re

_POLLING = re.compile(
    r"^/api/v1/(documents/[0-9a-f-]{36}|chats/[0-9a-f-]{36}/artifacts)$", re.IGNORECASE
)


class PollingAccessFilter(logging.Filter):
    """Drops 2xx/3xx access lines of `GET /documents/{id}` and `GET /chats/{id}/artifacts`.
    Errors and every other request still log."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple) or len(args) < 5:
            return True  # not uvicorn's (client, method, path, version, status) shape
        _, method, path, _, status = args[:5]
        if method != "GET" or not isinstance(status, int) or status >= 400:
            return True
        return _POLLING.match(str(path).split("?", 1)[0]) is None
