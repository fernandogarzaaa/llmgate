"""Errors returned to clients in the OpenAI error shape."""

from __future__ import annotations

from typing import Any, Dict, Optional


class GatewayError(Exception):
    """An error with an HTTP status and an OpenAI-style error body."""

    def __init__(
        self,
        status: int,
        message: str,
        type_: str = "invalid_request_error",
        code: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.type = type_
        self.code = code
        self.headers = headers or {}

    def body(self) -> Dict[str, Any]:
        return {"error": {"message": self.message, "type": self.type, "code": self.code}}
