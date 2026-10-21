import logging

from chat_pdf.adapters.http.access_log import PollingAccessFilter

ID = "0b6a3c0e-5f7c-4a39-9d58-2c0f4a1e8e11"


def _record(method, path, status):
    return logging.LogRecord(
        "uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
        ("1.2.3.4:5", method, path, "1.1", status), None,
    )  # fmt: skip


def test_drops_successful_polling_gets_only():
    f = PollingAccessFilter()
    noisy = [
        ("GET", f"/api/v1/documents/{ID}", 200),
        ("GET", f"/api/v1/chats/{ID}/artifacts", 200),
        ("GET", f"/api/v1/documents/{ID}?x=1", 304),
    ]
    kept = [
        ("GET", f"/api/v1/documents/{ID}", 500),
        ("GET", f"/api/v1/documents/{ID}", 404),
        ("GET", f"/api/v1/chats/{ID}/artifacts", 500),
        ("POST", f"/api/v1/chats/{ID}/artifacts", 202),
        ("GET", f"/api/v1/documents/{ID}/file", 200),
        ("GET", f"/api/v1/documents/{ID}/chunks", 200),
        ("GET", "/api/v1/chats", 200),
    ]
    assert not any(f.filter(_record(*r)) for r in noisy)
    assert all(f.filter(_record(*r)) for r in kept)


def test_unexpected_record_shapes_pass_through():
    f = PollingAccessFilter()
    assert f.filter(logging.LogRecord("x", logging.INFO, "", 0, "plain", None, None))
