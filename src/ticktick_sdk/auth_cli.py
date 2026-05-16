#!/usr/bin/env python3
"""
TickTick OAuth2 Token Acquisition CLI.

This module provides the OAuth2 flow for obtaining an access token
for the TickTick V1 API. It can be run in two modes:

- Auto mode: Opens browser and runs a local callback server
- Manual mode: Prints URL for manual visit (SSH-friendly)

This module is used by the main CLI (`ticktick-sdk auth`) and can also
be imported and used programmatically.

Example:
    # From command line
    ticktick-sdk auth
    ticktick-sdk auth --manual

    # Programmatically
    from ticktick_sdk.auth_cli import run_auth_flow
    exit_code = await run_auth_flow(manual=True)
"""

from __future__ import annotations

import asyncio
import html
import os
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar
from urllib.parse import parse_qs, urlparse

from ticktick_sdk.api.v1.auth import OAuth2Handler


# =============================================================================
# Constants
# =============================================================================

# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def supports_color() -> bool:
    """Check if the terminal supports color output."""
    # Check for NO_COLOR environment variable (https://no-color.org/)
    if os.environ.get("NO_COLOR"):
        return False

    # Check if stdout is a TTY
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False

    # Check for TERM environment variable
    term = os.environ.get("TERM", "")
    if term == "dumb":
        return False

    return True


def colorize(text: str, color: str) -> str:
    """Apply color to text if terminal supports it."""
    if supports_color():
        return f"{color}{text}{Colors.END}"
    return text


# =============================================================================
# OAuth Callback Handler
# =============================================================================


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """
    HTTP request handler for the OAuth2 callback.

    This handler receives the authorization code from TickTick after the user
    authorizes the application. It serves a simple HTML page to inform the user
    of success or failure.
    """

    # Class variables to store the callback result
    authorization_code: ClassVar[str | None] = None
    state: ClassVar[str | None] = None
    error: ClassVar[str | None] = None

    def do_GET(self) -> None:
        """Handle GET request (the OAuth callback)."""
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if "code" in params:
            OAuthCallbackHandler.authorization_code = params["code"][0]
            OAuthCallbackHandler.state = params.get("state", [None])[0]
            self._send_success_response()
        elif "error" in params:
            OAuthCallbackHandler.error = params.get(
                "error_description", params["error"]
            )[0]
            self._send_error_response()
        else:
            self._send_error_response("Unknown response")

    def _send_success_response(self) -> None:
        """Send success HTML response."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Authorization Successful</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                    text-align: center;
                    padding: 50px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    margin: 0;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                .container {
                    background: white;
                    padding: 40px 60px;
                    border-radius: 16px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                }
                h1 { color: #22c55e; margin-bottom: 20px; }
                p { color: #666; margin: 10px 0; }
                .icon { font-size: 64px; margin-bottom: 20px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="icon">✓</div>
                <h1>Authorization Successful!</h1>
                <p>You can close this window and return to the terminal.</p>
                <p style="color: #999; font-size: 14px;">
                    The access token will be displayed in your terminal.
                </p>
            </div>
        </body>
        </html>
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())

    def _send_error_response(self, error: str | None = None) -> None:
        """Send error HTML response."""
        error_msg = html.escape(error or OAuthCallbackHandler.error or "Unknown error")
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Authorization Failed</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                    text-align: center;
                    padding: 50px;
                    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
                    min-height: 100vh;
                    margin: 0;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }}
                .container {{
                    background: white;
                    padding: 40px 60px;
                    border-radius: 16px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                }}
                h1 {{ color: #ef4444; margin-bottom: 20px; }}
                p {{ color: #666; margin: 10px 0; }}
                .error {{ color: #999; font-size: 14px; }}
                .icon {{ font-size: 64px; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="icon">✗</div>
                <h1>Authorization Failed</h1>
                <p class="error">{error_msg}</p>
                <p>Please try again.</p>
            </div>
        </body>
        </html>
        """
        self.send_response(400)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html_body.encode())

    def log_message(self, format: str, *args: object) -> None:
        """Suppress default HTTP server logging."""
        pass


# =============================================================================
# Helper Functions
# =============================================================================


def reset_callback_state() -> None:
    """Reset the callback handler state for a fresh OAuth flow."""
    OAuthCallbackHandler.authorization_code = None
    OAuthCallbackHandler.state = None
    OAuthCallbackHandler.error = None


def print_header(title: str, manual: bool = False) -> None:
    """Print the header banner."""
    width = 60
    print()
    print(colorize("=" * width, Colors.CYAN))
    print(colorize(f"  {title}", Colors.BOLD))
    if manual:
        print(colorize("  (Manual Mode - SSH-friendly)", Colors.YELLOW))
    print(colorize("=" * width, Colors.CYAN))
    print()


def print_success_token(token: str) -> None:
    """Print the success message with the token."""
    width = 60
    print()
    print(colorize("=" * width, Colors.GREEN))
    print(colorize("  SUCCESS! Here is your access token:", Colors.BOLD + Colors.GREEN))
    print(colorize("=" * width, Colors.GREEN))
    print()
    print(token)
    print()
    print(colorize("=" * width, Colors.GREEN))


def print_env_instruction(_token: str) -> None:
    """Print instructions for using the token."""
    print()
    print(colorize("NEXT STEPS:", Colors.BOLD))
    print()

    # Python library users
    print(colorize("For Python Library users:", Colors.BOLD))
    print("  Add to your .env file:")
    print(colorize("    TICKTICK_ACCESS_TOKEN=<paste the access token shown above>", Colors.CYAN))
    print()

    # Claude Code users
    print(colorize("For Claude Code users:", Colors.BOLD))
    print("  Run (replace YOUR_* placeholders):")
    print(colorize(
        f"    claude mcp add ticktick \\\n"
        f"      -e TICKTICK_CLIENT_ID=YOUR_CLIENT_ID \\\n"
        f"      -e TICKTICK_CLIENT_SECRET=YOUR_CLIENT_SECRET \\\n"
        f"      -e TICKTICK_ACCESS_TOKEN=YOUR_ACCESS_TOKEN \\\n"
        f"      -e TICKTICK_USERNAME=YOUR_EMAIL \\\n"
        f"      -e TICKTICK_PASSWORD=YOUR_PASSWORD \\\n"
        f"      -- ticktick-sdk",
        Colors.CYAN,
    ))
    print()

    # Claude Desktop users
    print(colorize("For Claude Desktop users:", Colors.BOLD))
    print("  See README.md for config file location and JSON format.")
    print()


def print_token_expiry(expires_in: int | None, refresh_token: str | None) -> None:
    """Print token expiration information."""
    if expires_in:
        hours = expires_in / 3600
        print(
            colorize(
                f"Note: This token expires in {hours:.1f} hours ({expires_in} seconds)",
                Colors.YELLOW,
            )
        )
    else:
        print(
            colorize(
                "Note: Token expiration not specified (may be long-lived)",
                Colors.YELLOW,
            )
        )

    if refresh_token:
        print()
        print(
            colorize(
                "Refresh token was returned. Store it securely if you plan to implement token refresh.",
                Colors.YELLOW,
            )
        )

    print()


# =============================================================================
# OAuth Flow Functions
# =============================================================================


async def run_auto_mode(
    handler: OAuth2Handler,
    auth_url: str,
    callback_port: int,
) -> tuple[str | None, str | None]:
    """
    Run OAuth flow with automatic browser and local callback server.

    This mode opens the user's browser to the authorization URL and starts
    a local HTTP server to receive the callback with the authorization code.

    Args:
        handler: The OAuth2Handler instance.
        auth_url: The authorization URL to open in the browser.
        callback_port: The port to listen on for the callback.

    Returns:
        The authorization code if successful, None otherwise.
    """
    print("Step 1: Opening browser for authorization...")
    print()
    print("If the browser doesn't open, visit this URL manually:")
    print()
    print(colorize(f"  {auth_url}", Colors.UNDERLINE))
    print()

    # Start local server to receive callback
    server = HTTPServer(("127.0.0.1", callback_port), OAuthCallbackHandler)
    server.timeout = 120  # 2 minute timeout

    # Open browser
    webbrowser.open(auth_url)

    print(f"Step 2: Waiting for callback on port {callback_port}...")
    print("        (Authorize the app in your browser)")
    print()

    # Wait for callback
    while (
        OAuthCallbackHandler.authorization_code is None
        and OAuthCallbackHandler.error is None
    ):
        server.handle_request()

    server.server_close()

    if OAuthCallbackHandler.error:
        print(
            colorize(
                f"ERROR: Authorization failed: {OAuthCallbackHandler.error}",
                Colors.RED,
            )
        )
        return None, None

    return OAuthCallbackHandler.authorization_code, OAuthCallbackHandler.state


async def run_manual_mode(
    handler: OAuth2Handler,
    auth_url: str,
) -> tuple[str | None, str | None]:
    """
    Run OAuth flow manually (SSH-friendly).

    This mode prints the authorization URL for the user to visit manually
    and prompts them to paste the authorization code from the callback URL.

    Args:
        handler: The OAuth2Handler instance.
        auth_url: The authorization URL to display.

    Returns:
        The authorization code if provided, None otherwise.
    """
    width = 70
    print(colorize("=" * width, Colors.CYAN))
    print(
        colorize(
            "  STEP 1: Open this URL in ANY browser (phone, tablet, other computer)",
            Colors.BOLD,
        )
    )
    print(colorize("=" * width, Colors.CYAN))
    print()
    print(auth_url)
    print()
    print(colorize("=" * width, Colors.CYAN))
    print()
    print(colorize("STEP 2:", Colors.BOLD) + " Authorize the app when prompted")
    print()
    print(colorize("STEP 3:", Colors.BOLD) + " After authorizing, you'll be redirected to a URL like:")
    print("        http://127.0.0.1:8080/callback?code=XXXXX&state=YYYYY")
    print()
    print("        (The page will show an error - that's OK!)")
    print("        Copy the full callback URL from your browser's address bar.")
    print()
    print(colorize("=" * width, Colors.CYAN))
    print()

    try:
        code = input("Paste the full callback URL here: ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        print(colorize("Cancelled by user.", Colors.YELLOW))
        return None, None

    if not code:
        print(colorize("ERROR: No code provided", Colors.RED))
        return None, None

    # Clean up the code if they pasted more than just the code
    state: str | None = None
    if "code=" in code:
        # They pasted the full URL or query string
        try:
            parsed = parse_qs(code.split("?")[-1] if "?" in code else code)
            if "code" in parsed:
                code = parsed["code"][0]
            if "state" in parsed:
                state = parsed["state"][0]
        except Exception:
            pass  # Use the code as-is

    if state is None:
        print(colorize("ERROR: Please paste the full callback URL so the OAuth state can be verified.", Colors.RED))
        return None, None

    return code, state


# =============================================================================
# Main Entry Points
# =============================================================================


async def run_auth_flow(manual: bool = False) -> int:
    """
    Run the OAuth2 authentication flow.

    This function orchestrates the entire OAuth2 flow:
    1. Reads credentials from environment variables
    2. Creates an OAuth2Handler
    3. Generates the authorization URL
    4. Runs either auto or manual mode to get the authorization code
    5. Exchanges the code for an access token
    6. Prints the token with instructions

    Args:
        manual: If True, use manual mode (SSH-friendly).
                If False, use auto mode (opens browser).

    Returns:
        Exit code: 0 for success, 1 for error.
    """
    # Reset state for fresh flow
    reset_callback_state()

    # Print header
    print_header("TickTick OAuth2 Token Acquisition", manual=manual)

    # Read credentials from environment
    client_id = os.environ.get("TICKTICK_CLIENT_ID", "")
    client_secret = os.environ.get("TICKTICK_CLIENT_SECRET", "")
    redirect_uri = os.environ.get(
        "TICKTICK_REDIRECT_URI", "http://127.0.0.1:8080/callback"
    )

    # Validate credentials
    if not client_id or not client_secret:
        print(colorize("ERROR: Missing credentials!", Colors.RED))
        print()
        print("Please set environment variables:")
        print(colorize("  export TICKTICK_CLIENT_ID='your_client_id'", Colors.CYAN))
        print(colorize("  export TICKTICK_CLIENT_SECRET='your_client_secret'", Colors.CYAN))
        print()
        print("Or create a .env file with these values.")
        print()
        print("Get credentials from: " + colorize(
            "https://developer.ticktick.com/manage",
            Colors.UNDERLINE,
        ))
        return 1

    # Parse callback port from redirect URI
    parsed_uri = urlparse(redirect_uri)
    callback_port = parsed_uri.port or 8080

    # Display configuration
    client_id_masked = f"{client_id[:8]}...{client_id[-4:]}" if len(client_id) > 12 else client_id
    print(f"Client ID: {colorize(client_id_masked, Colors.CYAN)}")
    print(f"Redirect URI: {colorize(redirect_uri, Colors.CYAN)}")
    print()

    # Create OAuth handler
    handler = OAuth2Handler(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
    )

    # Generate authorization URL
    auth_url, _state = handler.get_authorization_url()

    # Get authorization code
    if manual:
        code, returned_state = await run_manual_mode(handler, auth_url)
    else:
        code, returned_state = await run_auto_mode(handler, auth_url, callback_port)

    if not code:
        print(colorize("ERROR: No authorization code received", Colors.RED))
        return 1

    # Exchange code for token
    print()
    print("Exchanging authorization code for access token...")

    try:
        token = await handler.exchange_code(code=code, state=returned_state)
    except Exception as e:
        print()
        print(colorize(f"ERROR: Token exchange failed: {e}", Colors.RED))
        return 1

    # Success! Display the token
    print_success_token(token.access_token)
    print_env_instruction(token.access_token)
    print_token_expiry(token.expires_in, token.refresh_token)

    return 0


def main(manual: bool = False) -> int:
    """
    Synchronous entry point for the auth CLI.

    This function is called by the main CLI and wraps the async flow.

    Args:
        manual: If True, use manual mode (SSH-friendly).

    Returns:
        Exit code: 0 for success, 1 for error.
    """
    try:
        return asyncio.run(run_auth_flow(manual=manual))
    except KeyboardInterrupt:
        print()
        print(colorize("Cancelled by user.", Colors.YELLOW))
        return 130  # Standard exit code for SIGINT


if __name__ == "__main__":
    # Allow running directly for development/testing
    import argparse

    parser = argparse.ArgumentParser(description="Get TickTick OAuth2 access token")
    parser.add_argument(
        "--manual",
        "-m",
        action="store_true",
        help="Manual mode: prints URL for you to visit (SSH-friendly)",
    )
    args = parser.parse_args()
    sys.exit(main(manual=args.manual))
