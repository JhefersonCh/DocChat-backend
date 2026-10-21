from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from chat_pdf.domain.errors import (
    Conflict,
    DomainError,
    InvalidInput,
    NotFound,
    ProviderError,
    ServiceUnavailable,
)

_STATUS_BY_TYPE: tuple[tuple[type[DomainError], int], ...] = (
    (NotFound, 404),
    (Conflict, 409),
    (InvalidInput, 422),
    (ServiceUnavailable, 503),
    (ProviderError, 502),
)
_STATUS_BY_CODE = {"unsupported_file_type": 415}  # overrides the status of its class


def status_of(exc: DomainError) -> int:
    if exc.code in _STATUS_BY_CODE:
        return _STATUS_BY_CODE[exc.code]
    return next((status for kind, status in _STATUS_BY_TYPE if isinstance(exc, kind)), 500)


def error_body(exc: Exception) -> dict[str, str]:
    """The single error shape of the API (also the payload of the SSE `error` event)."""
    if isinstance(exc, DomainError):
        return {"code": exc.code, "message": exc.message}
    return {"code": "internal_error", "message": str(exc) or type(exc).__name__}


def _response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"code": code, "message": message})


async def _domain_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, DomainError)
    return JSONResponse(status_code=status_of(exc), content=error_body(exc))


async def _validation_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    message = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
    return _response(422, "invalid_input", message)


async def _http_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)  # unknown route, wrong method, ...
    return _response(exc.status_code, "http_error", str(exc.detail))


async def _unexpected_error(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content=error_body(exc))


def register_error_handlers(app: FastAPI) -> None:
    handlers: dict[Any, Any] = {
        DomainError: _domain_error,
        RequestValidationError: _validation_error,
        StarletteHTTPException: _http_error,
        Exception: _unexpected_error,
    }
    for exc_type, handler in handlers.items():
        app.add_exception_handler(exc_type, handler)
