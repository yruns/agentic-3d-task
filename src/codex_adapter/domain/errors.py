"""Safe, stable errors crossing adapter boundaries."""

from __future__ import annotations


class AdapterError(Exception):
    """A public error whose message must never contain credentials or payloads."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.param = param

    def payload(self) -> dict[str, object]:
        """Return the Responses-compatible error envelope."""
        return {
            "error": {
                "type": "adapter_error",
                "code": self.code,
                "message": str(self),
                "param": self.param,
            }
        }
