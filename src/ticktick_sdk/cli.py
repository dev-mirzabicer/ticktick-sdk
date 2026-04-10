#!/usr/bin/env python3
"""
TickTick SDK Command Line Interface.

This module provides the main entry point for the ticktick-sdk command,
supporting multiple subcommands for different functionality.

Commands:
    ticktick-sdk              Run the MCP server (default)
    ticktick-sdk server       Run the MCP server (explicit)
    ticktick-sdk auth         Get OAuth2 access token (opens browser)
    ticktick-sdk auth --manual  Get OAuth2 access token (SSH-friendly)

Examples:
    # Start the MCP server for AI assistant integration
    ticktick-sdk

    # Get OAuth2 token (auto mode - opens browser)
    ticktick-sdk auth

    # Get OAuth2 token (manual mode - for SSH/headless environments)
    ticktick-sdk auth --manual

    # Show version
    ticktick-sdk --version

    # Show help
    ticktick-sdk --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn


import os

# Tool categories for --enabledModules flag
TOOL_MODULES = {
    "tasks": [
        "ticktick_create_tasks",
        "ticktick_get_task",
        "ticktick_list_tasks",
        "ticktick_update_tasks",
        "ticktick_complete_tasks",
        "ticktick_delete_tasks",
        "ticktick_move_tasks",
        "ticktick_set_task_parents",
        "ticktick_unparent_tasks",
        "ticktick_search_tasks",
        "ticktick_pin_tasks",
    ],
    "projects": [
        "ticktick_list_projects",
        "ticktick_get_project",
        "ticktick_create_project",
        "ticktick_update_project",
        "ticktick_delete_project",
    ],
    "folders": [
        "ticktick_list_folders",
        "ticktick_create_folder",
        "ticktick_rename_folder",
        "ticktick_delete_folder",
    ],
    "columns": [
        "ticktick_list_columns",
        "ticktick_create_column",
        "ticktick_update_column",
        "ticktick_delete_column",
    ],
    "tags": [
        "ticktick_list_tags",
        "ticktick_create_tag",
        "ticktick_update_tag",
        "ticktick_delete_tag",
        "ticktick_merge_tags",
    ],
    "habits": [
        "ticktick_habits",
        "ticktick_habit",
        "ticktick_habit_sections",
        "ticktick_create_habit",
        "ticktick_update_habit",
        "ticktick_delete_habit",
        "ticktick_checkin_habits",
        "ticktick_habit_checkins",
    ],
    "user": [
        "ticktick_get_profile",
        "ticktick_get_status",
        "ticktick_get_statistics",
        "ticktick_get_preferences",
    ],
    "checklist_items": [
        "ticktick_add_checklist_items",
        "ticktick_update_checklist_item",
        "ticktick_delete_checklist_items",
    ],
    "focus": [
        "ticktick_focus_heatmap",
        "ticktick_focus_by_tag",
    ],
}

# All tool names for validation
ALL_TOOLS = [tool for tools in TOOL_MODULES.values() for tool in tools]


def load_dotenv_if_available() -> None:
    """Load .env file if python-dotenv is available."""
    try:
        from dotenv import load_dotenv

        # Try current directory first, then walk up to find .env
        cwd = Path.cwd()
        for parent in [cwd, *cwd.parents]:
            env_file = parent / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                return
        # Fallback: let dotenv search for .env
        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed, skip


def get_version() -> str:
    """
    Get the package version.

    Uses importlib.metadata for Python 3.11+ to read the version
    from the installed package metadata.

    Returns:
        The package version string, or "unknown" if not found.
    """
    try:
        from importlib.metadata import version

        return version("ticktick-sdk")
    except Exception:
        return "unknown"


def resolve_enabled_tools(
    enabled_tools: str | None,
    enabled_modules: str | None,
) -> list[str] | None:
    """
    Resolve enabled tools from CLI arguments.

    Args:
        enabled_tools: Comma-separated list of specific tool names.
        enabled_modules: Comma-separated list of module names.

    Returns:
        List of enabled tool names, or None if all tools should be enabled.
    """
    if not enabled_tools and not enabled_modules:
        return None  # All tools enabled

    result = set()

    # Add tools from --enabledTools
    if enabled_tools:
        for tool in enabled_tools.split(","):
            tool = tool.strip()
            if tool:
                if tool not in ALL_TOOLS:
                    print(f"Warning: Unknown tool '{tool}', skipping", file=sys.stderr)
                else:
                    result.add(tool)

    # Add tools from --enabledModules
    if enabled_modules:
        for module in enabled_modules.split(","):
            module = module.strip().lower()
            if module:
                if module not in TOOL_MODULES:
                    print(
                        f"Warning: Unknown module '{module}'. "
                        f"Available: {', '.join(TOOL_MODULES.keys())}",
                        file=sys.stderr,
                    )
                else:
                    result.update(TOOL_MODULES[module])

    return list(result) if result else None


def run_server(
    enabled_tools: str | None = None,
    enabled_modules: str | None = None,
    host: str | None = None,
) -> int:
    """
    Run the MCP server.

    This starts the FastMCP server that exposes TickTick functionality
    as MCP tools for AI assistants.

    Args:
        enabled_tools: Comma-separated list of specific tools to enable.
        enabled_modules: Comma-separated list of modules to enable.
        host: API host ("ticktick.com" or "dida365.com").

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    # Set host if specified
    if host:
        host_lower = host.lower().strip()
        if host_lower in ("ticktick.com", "dida365.com"):
            os.environ["TICKTICK_HOST"] = host_lower
            print(f"Using API host: {host_lower}", file=sys.stderr)
        else:
            print(
                f"Warning: Invalid host '{host}'. "
                "Using default (ticktick.com). Valid: ticktick.com, dida365.com",
                file=sys.stderr,
            )

    # Resolve which tools to enable
    tools_to_enable = resolve_enabled_tools(enabled_tools, enabled_modules)

    # Pass to server via environment variable
    if tools_to_enable is not None:
        os.environ["TICKTICK_ENABLED_TOOLS"] = ",".join(tools_to_enable)
        print(
            f"Tool filtering enabled: {len(tools_to_enable)} of {len(ALL_TOOLS)} tools",
            file=sys.stderr,
        )

    from ticktick_sdk.server import main as server_main

    server_main()
    return 0


def run_auth(manual: bool = False) -> int:
    """
    Run the OAuth2 authentication flow.

    This guides the user through the OAuth2 flow to obtain an access token
    for the TickTick V1 API.

    Args:
        manual: If True, use manual mode (SSH-friendly).
                If False, use auto mode (opens browser).

    Returns:
        Exit code (0 for success, 1 for error).
    """
    from ticktick_sdk.auth_cli import main as auth_main

    return auth_main(manual=manual)


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    # Main parser
    parser = argparse.ArgumentParser(
        prog="ticktick-sdk",
        description="TickTick SDK - Async Python SDK and MCP Server for TickTick",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s                  Start the MCP server (default)
  %(prog)s server           Start the MCP server (explicit)
  %(prog)s auth             Get OAuth2 token (opens browser)
  %(prog)s auth --manual    Get OAuth2 token (SSH-friendly)

For more information, visit:
  https://github.com/dev-mirzabicer/ticktick-sdk
""",
    )

    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=f"%(prog)s {get_version()}",
    )

    # Subparsers for commands
    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available commands (default: server)",
        metavar="<command>",
    )

    # Server subcommand
    server_parser = subparsers.add_parser(
        "server",
        help="Run the MCP server for AI assistant integration",
        description="""\
Run the TickTick MCP server.

This starts the FastMCP server that exposes TickTick functionality
as tools for AI assistants like Claude. The server communicates
via stdio and implements the Model Context Protocol.

Before running the server, ensure your environment variables are set:
  - TICKTICK_CLIENT_ID
  - TICKTICK_CLIENT_SECRET
  - TICKTICK_ACCESS_TOKEN
  - TICKTICK_USERNAME
  - TICKTICK_PASSWORD

Tool Filtering (reduces context window usage):
  Use --enabledTools or --enabledModules to load only the tools you need.
  This can significantly reduce context usage from ~30-40% to ~5-10%.

Available modules: tasks, checklist_items, projects, folders, columns, tags, habits, user, focus
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    server_parser.add_argument(
        "--enabledTools",
        type=str,
        default=None,
        metavar="TOOLS",
        help=(
            "Comma-separated list of specific tools to enable. "
            "Example: --enabledTools ticktick_create_tasks,ticktick_list_tasks"
        ),
    )

    server_parser.add_argument(
        "--enabledModules",
        type=str,
        default=None,
        metavar="MODULES",
        help=(
            "Comma-separated list of tool modules to enable. "
            "Available: tasks, checklist_items, projects, folders, columns, tags, habits, user, focus. "
            "Example: --enabledModules tasks,projects"
        ),
    )

    server_parser.add_argument(
        "--host",
        type=str,
        default=None,
        metavar="HOST",
        help=(
            "API host to use. Options: ticktick.com (international, default), "
            "dida365.com (Chinese version). "
            "Can also be set via TICKTICK_HOST environment variable."
        ),
    )

    # Auth subcommand
    auth_parser = subparsers.add_parser(
        "auth",
        help="Get OAuth2 access token for TickTick API",
        description="""\
Get an OAuth2 access token for the TickTick V1 API.

This command guides you through the OAuth2 authorization flow:
1. Opens your browser to TickTick's authorization page
2. Waits for you to authorize the application
3. Exchanges the authorization code for an access token
4. Displays the token for you to copy to your .env file

Before running this command, ensure these environment variables are set:
  - TICKTICK_CLIENT_ID     (from developer.ticktick.com)
  - TICKTICK_CLIENT_SECRET (from developer.ticktick.com)

The redirect URI defaults to http://127.0.0.1:8080/callback
but can be customized with TICKTICK_REDIRECT_URI.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  ticktick-sdk auth             Opens browser for authorization
  ticktick-sdk auth --manual    Prints URL for manual authorization (SSH-friendly)

After obtaining the token, add it to your .env file:
  TICKTICK_ACCESS_TOKEN=your_token_here
""",
    )

    auth_parser.add_argument(
        "--manual",
        "-m",
        action="store_true",
        help="Manual mode: prints URL for you to visit (SSH-friendly)",
    )

    return parser


def main() -> int | NoReturn:
    """
    Main entry point for the CLI.

    Parses command line arguments and dispatches to the appropriate
    handler function.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    # Load .env file before doing anything else
    load_dotenv_if_available()

    parser = create_parser()
    args = parser.parse_args()

    # Default to server if no command specified
    if args.command is None:
        return run_server()
    elif args.command == "server":
        return run_server(
            enabled_tools=args.enabledTools,
            enabled_modules=args.enabledModules,
            host=args.host,
        )
    elif args.command == "auth":
        return run_auth(manual=args.manual)
    else:
        # This shouldn't happen with argparse, but handle it gracefully
        parser.print_help()
        return 1


def cli_main() -> NoReturn:
    """
    CLI entry point that exits with the appropriate code.

    This is the actual entry point referenced in pyproject.toml.
    It ensures proper exit code handling.
    """
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print()
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        # Catch unexpected errors and display them
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
