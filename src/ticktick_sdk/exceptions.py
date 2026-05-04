"""
TickTick SDK Exception Hierarchy.

This module defines a comprehensive exception hierarchy for the TickTick SDK.
All exceptions inherit from TickTickError for easy catching at the top level.

Exception Hierarchy:
    TickTickError (base)
    ├── TickTickAuthenticationError
    │   ├── TickTickOAuthError (V1-specific)
    │   └── TickTickSessionError (V2-specific)
    ├── TickTickAPIError
    │   ├── TickTickRateLimitError
    │   ├── TickTickNotFoundError
    │   ├── TickTickForbiddenError
    │   └── TickTickServerError
    ├── TickTickValidationError
    └── TickTickConfigurationError
"""

from __future__ import annotations

from typing import Any


_REDACTED_DETAIL_KEYS = (
    "access_token",
    "authorization",
    "client_secret",
    "cookie",
    "password",
    "refresh_token",
    "response",
    "response_body",
    "secret",
    "session",
    "token",
)


def _should_redact_key(key: str) -> bool:
    """Check whether a detail key likely contains sensitive data."""
    key_lower = key.lower()
    return any(sensitive_key in key_lower for sensitive_key in _REDACTED_DETAIL_KEYS)


def _redact_detail_value(key: str, value: Any) -> Any:
    """Recursively redact sensitive exception detail values."""
    if _should_redact_key(key):
        if isinstance(value, str):
            return f"<redacted {len(value)} chars>"
        if isinstance(value, (dict, list, tuple, set)):
            return "<redacted structured data>"
        return "<redacted>"

    if isinstance(value, dict):
        return {nested_key: _redact_detail_value(nested_key, nested_value) for nested_key, nested_value in value.items()}
    if isinstance(value, list):
        return [_redact_detail_value(key, item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_detail_value(key, item) for item in value)
    return value


class TickTickError(Exception):
    """Base exception for all TickTick SDK errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.safe_details}"
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, details={self.safe_details!r})"

    @property
    def safe_details(self) -> dict[str, Any]:
        """Get a logging-safe view of the exception details."""
        return {key: _redact_detail_value(key, value) for key, value in self.details.items()}


# =============================================================================
# Authentication Errors
# =============================================================================


class TickTickAuthenticationError(TickTickError):
    """Base exception for authentication failures."""

    pass


class TickTickOAuthError(TickTickAuthenticationError):
    """V1 OAuth2-specific authentication error."""

    def __init__(
        self,
        message: str,
        oauth_error: str | None = None,
        oauth_error_description: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if oauth_error:
            details["oauth_error"] = oauth_error
        if oauth_error_description:
            details["oauth_error_description"] = oauth_error_description
        super().__init__(message, details)
        self.oauth_error = oauth_error
        self.oauth_error_description = oauth_error_description


class TickTickSessionError(TickTickAuthenticationError):
    """V2 Session-based authentication error."""

    def __init__(
        self,
        message: str,
        requires_2fa: bool = False,
        auth_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        details["requires_2fa"] = requires_2fa
        if auth_id:
            details["auth_id"] = auth_id
        super().__init__(message, details)
        self.requires_2fa = requires_2fa
        self.auth_id = auth_id


# =============================================================================
# API Errors
# =============================================================================


class TickTickAPIError(TickTickError):
    """Base exception for API call failures."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: str | None = None,
        api_version: str | None = None,
        endpoint: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if status_code is not None:
            details["status_code"] = status_code
        if response_body:
            details["response_body"] = response_body
        if api_version:
            details["api_version"] = api_version
        if endpoint:
            details["endpoint"] = endpoint
        super().__init__(message, details)
        self.status_code = status_code
        self.response_body = response_body
        self.api_version = api_version
        self.endpoint = endpoint


class TickTickRateLimitError(TickTickAPIError):
    """Rate limit exceeded error."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {}) or {}
        if retry_after is not None:
            details["retry_after"] = retry_after
        super().__init__(message, details=details, **kwargs)
        self.retry_after = retry_after


class TickTickNotFoundError(TickTickAPIError):
    """Resource not found error (404)."""

    def __init__(
        self,
        message: str = "Resource not found",
        resource_type: str | None = None,
        resource_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {}) or {}
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = resource_id
        super().__init__(message, status_code=404, details=details, **kwargs)
        self.resource_type = resource_type
        self.resource_id = resource_id


class TickTickForbiddenError(TickTickAPIError):
    """Access forbidden error (403)."""

    def __init__(
        self,
        message: str = "Access forbidden",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, status_code=403, **kwargs)


class TickTickServerError(TickTickAPIError):
    """Server-side error (5xx)."""

    def __init__(
        self,
        message: str = "Server error",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)


class TickTickQuotaExceededError(TickTickAPIError):
    """Account quota exceeded (free tier limits)."""

    def __init__(
        self,
        message: str = "Account quota exceeded",
        quota_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {}) or {}
        if quota_type:
            details["quota_type"] = quota_type
        super().__init__(message, details=details, **kwargs)
        self.quota_type = quota_type


# =============================================================================
# Validation & Configuration Errors
# =============================================================================


class TickTickValidationError(TickTickError):
    """Data validation error."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
        expected: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = repr(value)
        if expected:
            details["expected"] = expected
        super().__init__(message, details)
        self.field = field
        self.value = value
        self.expected = expected


class TickTickConfigurationError(TickTickError):
    """Configuration error (missing credentials, invalid settings)."""

    def __init__(
        self,
        message: str,
        missing_config: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if missing_config:
            details["missing_config"] = missing_config
        super().__init__(message, details)
        self.missing_config = missing_config or []


# =============================================================================
# Unified API Errors
# =============================================================================


class TickTickAPIUnavailableError(TickTickError):
    """Raised when neither V1 nor V2 API is available for an operation."""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        v1_error: TickTickError | None = None,
        v2_error: TickTickError | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if operation:
            details["operation"] = operation
        if v1_error:
            details["v1_error"] = str(v1_error)
        if v2_error:
            details["v2_error"] = str(v2_error)
        super().__init__(message, details)
        self.operation = operation
        self.v1_error = v1_error
        self.v2_error = v2_error
