class SKALDError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        details: str | None = None,
        suggested_fix: str | None = None
    ):
        self.code = code
        self.message = message
        self.details = details
        self.suggested_fix = suggested_fix
        super().__init__(message)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "suggested_fix": self.suggested_fix
        }

    def __str__(self) -> str:
        base = f"[{self.code}] {self.message}"
        if self.details:
            base += f" | Details: {self.details}"
        return base

    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        code: str = "INTERNAL_ERROR",
        suggested_fix: str | None = None
    ):
        return cls(
            code=code,
            message=str(exc),
            details=repr(exc),
            suggested_fix=suggested_fix
        )
