from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ticktick_sdk.api.v1.auth import OAuth2Token
from ticktick_sdk.auth_cli import (
    OAuthCallbackHandler,
    print_env_instruction,
    print_token_expiry,
    reset_callback_state,
    run_manual_mode,
)


pytestmark = [pytest.mark.unit]


@pytest.mark.asyncio
async def test_run_manual_mode_extracts_code_and_state() -> None:
    with patch(
        "builtins.input",
        return_value="http://127.0.0.1:8080/callback?code=abc123&state=state456",
    ):
        code, state = await run_manual_mode(
            handler=object(),  # type: ignore[arg-type]
            auth_url="https://example.test",
        )

    assert code == "abc123"
    assert state == "state456"


@pytest.mark.asyncio
async def test_run_manual_mode_requires_state() -> None:
    with patch("builtins.input", return_value="abc123"):
        code, state = await run_manual_mode(
            handler=object(),  # type: ignore[arg-type]
            auth_url="https://example.test",
        )

    assert code is None
    assert state is None


@pytest.mark.asyncio
async def test_run_auth_flow_passes_returned_state_to_exchange() -> None:
    token = OAuth2Token(access_token="access-token", refresh_token="refresh-token")

    with (
        patch("ticktick_sdk.auth_cli.print_header"),
        patch("ticktick_sdk.auth_cli.print_success_token"),
        patch("ticktick_sdk.auth_cli.print_env_instruction"),
        patch("ticktick_sdk.auth_cli.print_token_expiry"),
        patch.dict(
            "os.environ",
            {
                "TICKTICK_CLIENT_ID": "client-id",
                "TICKTICK_CLIENT_SECRET": "client-secret",
                "TICKTICK_REDIRECT_URI": "http://127.0.0.1:8080/callback",
            },
            clear=False,
        ),
        patch("ticktick_sdk.auth_cli.OAuth2Handler") as mock_handler_class,
        patch(
            "ticktick_sdk.auth_cli.run_manual_mode",
            AsyncMock(return_value=("auth-code", "expected-state")),
        ),
    ):
        mock_handler = mock_handler_class.return_value
        mock_handler.get_authorization_url.return_value = ("https://example.test", "expected-state")
        mock_handler.exchange_code = AsyncMock(return_value=token)

        from ticktick_sdk.auth_cli import run_auth_flow

        exit_code = await run_auth_flow(manual=True)

    assert exit_code == 0
    mock_handler.exchange_code.assert_awaited_once_with(
        code="auth-code",
        state="expected-state",
    )


def test_print_env_instruction_uses_placeholders(capsys: pytest.CaptureFixture[str]) -> None:
    print_env_instruction("secret-access-token")
    output = capsys.readouterr().out

    assert "secret-access-token" not in output
    assert "YOUR_ACCESS_TOKEN" in output


def test_print_token_expiry_does_not_echo_refresh_token(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_token_expiry(3600, "refresh-token-secret")
    output = capsys.readouterr().out

    assert "refresh-token-secret" not in output
    assert "Refresh token was returned" in output


def test_callback_error_response_escapes_html() -> None:
    reset_callback_state()
    OAuthCallbackHandler.error = "<script>alert(1)</script>"

    handler = OAuthCallbackHandler.__new__(OAuthCallbackHandler)
    response: dict[str, object] = {}

    handler.send_response = lambda status: response.setdefault("status", status)  # type: ignore[method-assign]
    handler.send_header = lambda name, value: response.setdefault("headers", []).append((name, value))  # type: ignore[method-assign]
    handler.end_headers = lambda: None  # type: ignore[method-assign]

    class Writer:
        def __init__(self) -> None:
            self.body = b""

        def write(self, data: bytes) -> None:
            self.body += data

    writer = Writer()
    handler.wfile = writer  # type: ignore[attr-defined]

    handler._send_error_response()

    body = writer.body.decode()
    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body
