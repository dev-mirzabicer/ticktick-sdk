#!/usr/bin/env python3
"""
TickTick MCP Server - Comprehensive Task Management Integration.

This MCP server provides a complete interface for interacting with TickTick,
combining both V1 (OAuth2) and V2 (Session) APIs for maximum functionality.
It enables AI assistants to manage tasks, projects, tags, and track productivity.

=== CAPABILITIES ===

Task Management:
    - Create tasks with titles, due dates, priorities, tags, reminders, and recurrence
    - Create subtasks (parent-child relationships)
    - Update, complete, delete, and move tasks between projects
    - List active, completed, and overdue tasks
    - Search tasks by title or content

Project Management:
    - Create, read, update, and delete projects
    - Organize projects into folders
    - Get project details with all tasks

Tag Management:
    - Create, rename, merge, and delete tags
    - Tags support hierarchical nesting
    - Apply tags to tasks for organization

User Information:
    - Get user profile and account status
    - Access productivity statistics (completion rates, scores, levels)
    - Track focus/pomodoro sessions

=== TICKTICK API BEHAVIORS ===

IMPORTANT: TickTick has several unique API behaviors that tools account for:

1. SOFT DELETE: Deleting tasks moves them to trash (deleted=1) rather than
   permanently removing them. Deleted tasks remain accessible via get_task.

2. RECURRENCE REQUIRES START_DATE: Creating recurring tasks without a start_date
   silently ignores the recurrence rule. Always provide start_date with recurrence.

3. PARENT-CHILD RELATIONSHIPS: Setting parent_id during task creation is ignored
   by the API. Use the make_subtask tool to establish parent-child relationships.

4. DATE CLEARING: To clear a task's due_date or start_date, you must also clear
   both dates together (TickTick restores due_date from start_date otherwise).

5. TAG ORDER: The API does not preserve tag order - tags may be returned in
   any order regardless of how they were provided.

6. INBOX: The inbox is a special project that cannot be deleted. Its ID is
   available via get_status (inbox_id field).

=== AUTHENTICATION ===

This server requires BOTH V1 and V2 authentication for full functionality:

V1 (OAuth2) - Required for get_project_with_data:
    TICKTICK_CLIENT_ID      - OAuth2 client ID from developer portal
    TICKTICK_CLIENT_SECRET  - OAuth2 client secret
    TICKTICK_ACCESS_TOKEN   - Access token from OAuth2 flow

V2 (Session) - Required for most operations:
    TICKTICK_USERNAME       - TickTick account email
    TICKTICK_PASSWORD       - TickTick account password

Optional:
    TICKTICK_REDIRECT_URI   - OAuth2 redirect URI (default: http://localhost:8080/callback)
    TICKTICK_TIMEOUT        - Request timeout in seconds (default: 30)
    TICKTICK_DEVICE_ID      - Device identifier (auto-generated if not set)

=== RESPONSE FORMATS ===

All tools support two response formats via the `response_format` parameter:

- "markdown" (default): Human-readable formatted text with headers, lists, and
  timestamps in readable format. Best for displaying results to users.

- "json": Machine-readable structured data with all available fields.
  Best for programmatic processing or when specific field values are needed.

=== ERROR HANDLING ===

Tools return clear, actionable error messages:
- Authentication errors: Check credentials configuration
- Not found errors: Verify resource ID exists
- Validation errors: Check input parameters
- Rate limit errors: Wait before retrying
- Server errors: Retry or contact support
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from typing import Any, AsyncIterator

from mcp.server.fastmcp import FastMCP, Context

from ticktick_sdk.client import TickTickClient
from ticktick_sdk.settings import get_settings
from ticktick_sdk.tools.inputs import (
    ResponseFormat,
    # Task inputs - list-based for batch operations
    CreateTasksInput,
    TaskGetInput,
    UpdateTasksInput,
    CompleteTasksInput,
    DeleteTasksInput,
    MoveTasksInput,
    SetTaskParentsInput,
    UnparentTasksInput,
    TaskListInput,
    PinTasksInput,
    SearchInput,
    # Project inputs
    ProjectCreateInput,
    ProjectGetInput,
    ProjectDeleteInput,
    ProjectUpdateInput,
    # Folder inputs
    FolderCreateInput,
    FolderDeleteInput,
    FolderRenameInput,
    # Column inputs
    ColumnListInput,
    ColumnCreateInput,
    ColumnUpdateInput,
    ColumnDeleteInput,
    # Tag inputs
    TagCreateInput,
    TagDeleteInput,
    TagMergeInput,
    TagUpdateInput,
    # Focus inputs
    FocusStatsInput,
    # Habit inputs
    HabitListInput,
    HabitGetInput,
    HabitCreateInput,
    HabitUpdateInput,
    HabitDeleteInput,
    CheckinHabitsInput,
    HabitCheckinsInput,
)
from ticktick_sdk.models import Habit, HabitSection
from ticktick_sdk.tools.formatting import (
    format_task_markdown,
    format_task_json,
    format_tasks_markdown,
    format_tasks_json,
    format_project_markdown,
    format_project_json,
    format_projects_markdown,
    format_projects_json,
    format_tag_markdown,
    format_tag_json,
    format_tags_markdown,
    format_tags_json,
    format_folders_markdown,
    format_folders_json,
    format_column_markdown,
    format_column_json,
    format_columns_markdown,
    format_columns_json,
    format_user_markdown,
    format_user_status_markdown,
    format_statistics_markdown,
    format_response,
    success_message,
    error_message,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Maximum response size in characters to prevent overwhelming context
CHARACTER_LIMIT = 25000

# Default pagination limits
DEFAULT_TASK_LIMIT = 50
DEFAULT_PROJECT_LIMIT = 100
MAX_TASK_LIMIT = 200


# =============================================================================
# Truncation Helper
# =============================================================================


def truncate_response(
    result: str,
    items_count: int,
    truncated_count: int | None = None,
) -> str:
    """
    Truncate response if it exceeds CHARACTER_LIMIT.

    Args:
        result: The formatted response string
        items_count: Total number of items before truncation
        truncated_count: Number of items after truncation (if different)

    Returns:
        Truncated response with guidance message if needed
    """
    if len(result) <= CHARACTER_LIMIT:
        return result

    # Find a good truncation point (after a complete item)
    truncate_at = CHARACTER_LIMIT - 500  # Leave room for message
    truncate_point = result.rfind("\n\n", 0, truncate_at)
    if truncate_point == -1:
        truncate_point = result.rfind("\n", 0, truncate_at)
    if truncate_point == -1:
        truncate_point = truncate_at

    truncated = result[:truncate_point]

    # Add truncation message
    message = (
        f"\n\n---\n"
        f"⚠️ **Response truncated** (exceeded {CHARACTER_LIMIT:,} characters)\n\n"
        f"Showing partial results. To see more:\n"
        f"- Use filters (project_id, tag, priority) to narrow results\n"
        f"- Use the 'limit' parameter to reduce the number of items\n"
        f"- Request response_format='json' for more compact output"
    )

    return truncated + message


# =============================================================================
# Lifespan Management
# =============================================================================


@asynccontextmanager
async def lifespan(mcp: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """
    Manage the TickTick client lifecycle.

    Initializes the client on startup and closes it on shutdown.
    """
    logger.info("Initializing TickTick MCP Server...")

    try:
        client = TickTickClient.from_settings()
        await client.connect()
        logger.info("TickTick client connected successfully")
        yield {"client": client}
    except Exception as e:
        logger.error("Failed to initialize TickTick client: %s", e)
        raise
    finally:
        if "client" in locals():
            await client.disconnect()
            logger.info("TickTick client disconnected")


# Initialize FastMCP server
mcp = FastMCP(
    "ticktick_sdk",
    lifespan=lifespan,
)


def get_client(ctx: Context) -> TickTickClient:
    """Get the TickTick client from context."""
    return ctx.request_context.lifespan_context["client"]


# =============================================================================
# Error Handling
# =============================================================================


def handle_error(e: Exception, operation: str) -> str:
    """
    Handle exceptions and return user-friendly, actionable error messages.

    Error messages include:
    1. What went wrong
    2. Why it might have happened
    3. Specific steps to resolve the issue
    """
    logger.exception("Error in %s: %s", operation, e)

    error_type = type(e).__name__
    error_str = str(e)

    if "Authentication" in error_type:
        return error_message(
            "Authentication failed",
            "NEXT STEPS:\n"
            "1. Verify environment variables are set:\n"
            "   - TICKTICK_CLIENT_ID (OAuth2 client ID)\n"
            "   - TICKTICK_CLIENT_SECRET (OAuth2 client secret)\n"
            "   - TICKTICK_ACCESS_TOKEN (OAuth2 access token)\n"
            "   - TICKTICK_USERNAME (TickTick account email)\n"
            "   - TICKTICK_PASSWORD (TickTick account password)\n"
            "2. Check that credentials are not expired\n"
            "3. Re-run OAuth2 flow if access token is invalid"
        )
    elif "NotFound" in error_type:
        resource_hint = ""
        if "task" in error_str.lower():
            resource_hint = (
                "HINTS:\n"
                "- Task may have been permanently deleted (not just trashed)\n"
                "- Use ticktick_list_tasks to see available tasks\n"
                "- Check if the task ID is correct"
            )
        elif "project" in error_str.lower():
            resource_hint = (
                "HINTS:\n"
                "- Use ticktick_list_projects to see available projects\n"
                "- The inbox project ID can be obtained from ticktick_get_status"
            )
        elif "tag" in error_str.lower():
            resource_hint = (
                "HINTS:\n"
                "- Use ticktick_list_tags to see available tags\n"
                "- Tag names are case-insensitive"
            )
        elif "folder" in error_str.lower() or "group" in error_str.lower():
            resource_hint = (
                "HINTS:\n"
                "- Use ticktick_list_folders to see available folders"
            )
        return error_message(
            f"Resource not found: {error_str}",
            resource_hint or "Verify the ID is correct and the resource exists."
        )
    elif "Validation" in error_type:
        return error_message(
            f"Invalid input: {error_str}",
            "Check the parameter types and constraints in the tool documentation."
        )
    elif "Configuration" in error_type:
        if "recurrence" in error_str.lower() and "start_date" in error_str.lower():
            return error_message(
                f"Configuration error: {error_str}",
                "TICKTICK REQUIREMENT: Recurring tasks require a start_date.\n"
                "Add a start_date parameter when setting recurrence rules."
            )
        return error_message(
            f"Configuration error: {error_str}",
            "Check your environment variables and tool parameters."
        )
    elif "RateLimit" in error_type:
        return error_message(
            "Rate limit exceeded",
            "NEXT STEPS:\n"
            "1. Wait 30-60 seconds before retrying\n"
            "2. Reduce the frequency of API calls\n"
            "3. Batch operations where possible"
        )
    elif "Quota" in error_type:
        return error_message(
            "Account quota exceeded",
            "HINTS:\n"
            "- Free accounts have limited projects/tasks\n"
            "- Delete unused projects or upgrade to Pro"
        )
    elif "Forbidden" in error_type:
        return error_message(
            f"Access denied: {error_str}",
            "You don't have permission to access this resource.\n"
            "Check if you're the owner or have appropriate sharing permissions."
        )
    elif "Server" in error_type:
        return error_message(
            f"TickTick server error: {error_str}",
            "NEXT STEPS:\n"
            "1. Wait a moment and retry the operation\n"
            "2. Check if TickTick service is operational\n"
            "3. Try with different parameters if the issue persists"
        )
    else:
        return error_message(
            f"Unexpected error: {error_str}",
            f"Error type: {error_type}\n"
            "If this persists, check the server logs for more details."
        )


# =============================================================================
# Task Tools
# =============================================================================


@mcp.tool(
    name="ticktick_create_tasks",
    annotations={
        "title": "Create Tasks",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def ticktick_create_tasks(params: CreateTasksInput, ctx: Context) -> str:
    """
    Create one or more tasks in TickTick.

    Creates tasks with specified properties. Supports batch creation (1-50 tasks).
    Each task can include title, due date, priority, tags, reminders, and recurrence.

    IMPORTANT BEHAVIORS:
    - If no project_id is specified, tasks are created in the inbox
    - RECURRENCE REQUIRES start_date: Recurrence rules without start_date are ignored
    - parent_id is supported - the SDK handles parent assignment after creation
    - Tags are created automatically if they don't exist

    Args:
        params: Task creation parameters:
            - tasks (list, required): List of task specifications (1-50 tasks)
              Each task must contain:
                - title (str, required): Task title
              Optional fields:
                - project_id (str): Project ID (defaults to inbox)
                - content (str): Task notes/description
                - priority (str): 'none', 'low', 'medium', 'high'
                - start_date (str): Start date in ISO format (REQUIRED for recurrence)
                - due_date (str): Due date in ISO format
                - tags (list[str]): Tag names to apply
                - reminders (list[str]): Reminder triggers in iCal format
                - recurrence (str): RRULE format
                - parent_id (str): Parent task ID for subtasks
            - response_format (str): 'markdown' (default) or 'json'

    Returns:
        On success: Summary of created tasks with details
        On error: Error message with hints for resolution

    Examples:
        Single task:
            tasks=[{"title": "Buy groceries"}]

        Multiple tasks:
            tasks=[
                {"title": "Task 1", "priority": "high"},
                {"title": "Task 2", "tags": ["work"]},
                {"title": "Task 3", "due_date": "2026-01-20"}
            ]

        Task with parent (subtask):
            tasks=[{"title": "Subtask", "parent_id": "parent_task_id", "project_id": "proj_id"}]
    """
    try:
        client = get_client(ctx)

        # Build task specifications
        task_specs = []
        for task_item in params.tasks:
            spec: dict[str, Any] = {"title": task_item.title}

            if task_item.project_id:
                spec["project_id"] = task_item.project_id
            if task_item.content:
                spec["content"] = task_item.content
            if task_item.description:
                spec["description"] = task_item.description
            if task_item.priority:
                spec["priority"] = task_item.priority
            if task_item.start_date:
                spec["start_date"] = task_item.start_date
            if task_item.due_date:
                spec["due_date"] = task_item.due_date
            if task_item.time_zone:
                spec["time_zone"] = task_item.time_zone
            if task_item.all_day is not None:
                spec["all_day"] = task_item.all_day
            if task_item.tags:
                spec["tags"] = task_item.tags
            if task_item.reminders:
                spec["reminders"] = task_item.reminders
            if task_item.recurrence:
                spec["recurrence"] = task_item.recurrence
            if task_item.parent_id:
                spec["parent_id"] = task_item.parent_id

            task_specs.append(spec)

        created_tasks = await client.create_tasks(task_specs)

        if params.response_format == ResponseFormat.MARKDOWN:
            if len(created_tasks) == 1:
                return f"# Task Created\n\n{format_task_markdown(created_tasks[0])}"
            else:
                return f"# {len(created_tasks)} Tasks Created\n\n{format_tasks_markdown(created_tasks, 'Created Tasks')}"
        else:
            return json.dumps({
                "success": True,
                "count": len(created_tasks),
                "tasks": [format_task_json(t) for t in created_tasks]
            }, indent=2)

    except Exception as e:
        return handle_error(e, "create_tasks")


@mcp.tool(
    name="ticktick_get_task",
    annotations={
        "title": "Get Task",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_get_task(params: TaskGetInput, ctx: Context) -> str:
    """
    Get a task by its ID.

    Retrieves full details of a specific task including title, description,
    due date, priority, tags, subtasks, and completion status.

    Args:
        params: Query parameters including:
            - task_id (str): Task identifier (required)
            - project_id (str): Project ID (needed for V1 fallback)

    Returns:
        Formatted task details or error message.
    """
    try:
        client = get_client(ctx)
        task = await client.get_task(params.task_id, params.project_id)

        if params.response_format == ResponseFormat.MARKDOWN:
            return format_task_markdown(task)
        else:
            return json.dumps(format_task_json(task), indent=2)

    except Exception as e:
        return handle_error(e, "get_task")


@mcp.tool(
    name="ticktick_list_tasks",
    annotations={
        "title": "List Tasks",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_list_tasks(params: TaskListInput, ctx: Context) -> str:
    """
    List tasks with flexible filtering.

    This unified tool handles all task listing scenarios:
    - Active tasks (default)
    - Completed tasks (with date range)
    - Abandoned tasks (with date range)
    - Deleted/trashed tasks

    Args:
        params: Filter parameters:
            - status (str): 'active' (default), 'completed', 'abandoned', 'deleted'
            - project_id (str): Filter by project
            - column_id (str): Filter by kanban column (active only, use with project_id)
            - tag (str): Filter by tag name
            - priority (str): Filter by priority level
            - due_today (bool): Only tasks due today (active only)
            - overdue (bool): Only overdue tasks (active only)
            - from_date (str): Start date for completed/abandoned (YYYY-MM-DD)
            - to_date (str): End date for completed/abandoned (YYYY-MM-DD)
            - days (int): Days to look back for completed/abandoned (default 7)
            - limit (int): Maximum results (default 50)

    Returns:
        Formatted list of tasks or error message.

    Examples:
        - Active tasks: status="active" (or just omit status)
        - Completed last 7 days: status="completed"
        - Completed in range: status="completed", from_date="2026-01-01", to_date="2026-01-15"
        - Abandoned tasks: status="abandoned", days=30
        - Deleted tasks: status="deleted"
        - Active + project: status="active", project_id="..."
        - Tasks in column: project_id="...", column_id="..." (kanban workflow)
    """
    try:
        client = get_client(ctx)

        # Handle different status types
        if params.status == "active":
            tasks = await client.get_all_tasks()

            # Apply active-only filters
            if params.project_id:
                tasks = [t for t in tasks if t.project_id == params.project_id]

            if params.column_id:
                tasks = [t for t in tasks if t.column_id == params.column_id]

            if params.tag:
                tag_lower = params.tag.lower()
                tasks = [t for t in tasks if any(tag.lower() == tag_lower for tag in t.tags)]

            if params.priority:
                priority_map = {"none": 0, "low": 1, "medium": 3, "high": 5}
                target_priority = priority_map.get(params.priority, 0)
                tasks = [t for t in tasks if t.priority == target_priority]

            if params.due_today:
                today = date.today()
                tasks = [t for t in tasks if t.due_date and t.due_date.date() == today]

            if params.overdue:
                today = date.today()
                tasks = [t for t in tasks if t.due_date and t.due_date.date() < today and not t.is_completed]

        elif params.status == "completed":
            # Completed tasks require date range
            if params.from_date and params.to_date:
                from_dt = datetime.fromisoformat(params.from_date)
                to_dt = datetime.fromisoformat(params.to_date)
            else:
                to_dt = datetime.now()
                from_dt = to_dt - timedelta(days=params.days)

            tasks = await client.get_completed_tasks(days=params.days, limit=params.limit)

        elif params.status == "abandoned":
            # Abandoned tasks require date range
            tasks = await client.get_abandoned_tasks(days=params.days, limit=params.limit)

        elif params.status == "deleted":
            tasks = await client.get_deleted_tasks(limit=params.limit)

        else:
            tasks = await client.get_all_tasks()

        # Apply limit
        total_count = len(tasks)
        tasks = tasks[: params.limit]

        if params.response_format == ResponseFormat.MARKDOWN:
            title = f"{params.status.capitalize()} Tasks" if params.status else "Tasks"
            result = format_tasks_markdown(tasks, title)
        else:
            result = json.dumps(format_tasks_json(tasks), indent=2)

        # Apply truncation if response is too large
        return truncate_response(result, total_count, len(tasks))

    except Exception as e:
        return handle_error(e, "list_tasks")


@mcp.tool(
    name="ticktick_update_tasks",
    annotations={
        "title": "Update Tasks",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_update_tasks(params: UpdateTasksInput, ctx: Context) -> str:
    """
    Update one or more tasks.

    Updates specified fields of tasks. Supports batch updates (1-100 tasks).
    Each update preserves unspecified fields.

    Also supports column assignment for kanban boards via column_id field.

    Args:
        params: Update parameters:
            - tasks (list, required): List of update specifications (1-100 tasks)
              Each update must contain:
                - task_id (str, required): Task to update
                - project_id (str, required): Project containing the task
              Optional update fields:
                - title (str): New title
                - content (str): New content/notes
                - priority (str): New priority ('none', 'low', 'medium', 'high')
                - start_date (str): New start date
                - due_date (str): New due date
                - tags (list): New tags (replaces existing)
                - column_id (str): Kanban column ID (empty string to remove from column)
            - response_format (str): 'markdown' (default) or 'json'

    Returns:
        Summary of updated tasks or error message.

    Examples:
        Single task update:
            tasks=[{"task_id": "abc123", "project_id": "proj1", "priority": "high"}]

        Multiple tasks:
            tasks=[
                {"task_id": "abc1", "project_id": "proj1", "title": "Updated 1"},
                {"task_id": "abc2", "project_id": "proj1", "priority": "low"}
            ]

        Move to kanban column:
            tasks=[{"task_id": "abc1", "project_id": "proj1", "column_id": "col123"}]
    """
    try:
        client = get_client(ctx)

        # Build update specifications
        update_specs = []
        for task_item in params.tasks:
            spec: dict[str, Any] = {
                "task_id": task_item.task_id,
                "project_id": task_item.project_id,
            }

            if task_item.title is not None:
                spec["title"] = task_item.title
            if task_item.content is not None:
                spec["content"] = task_item.content
            if task_item.priority is not None:
                spec["priority"] = task_item.priority
            if task_item.start_date is not None:
                spec["start_date"] = task_item.start_date
            if task_item.due_date is not None:
                spec["due_date"] = task_item.due_date
            if task_item.all_day is not None:
                spec["all_day"] = task_item.all_day
            if task_item.time_zone is not None:
                spec["time_zone"] = task_item.time_zone
            if task_item.tags is not None:
                spec["tags"] = task_item.tags
            if task_item.recurrence is not None:
                spec["recurrence"] = task_item.recurrence
            if task_item.column_id is not None:
                spec["column_id"] = task_item.column_id

            update_specs.append(spec)

        response = await client.update_tasks(update_specs)

        if params.response_format == ResponseFormat.MARKDOWN:
            count = len(update_specs)
            if count == 1:
                return f"# Task Updated\n\nSuccessfully updated task `{update_specs[0]['task_id']}`"
            else:
                return f"# {count} Tasks Updated\n\nSuccessfully updated {count} tasks."
        else:
            return json.dumps({
                "success": True,
                "count": len(update_specs),
                "response": response
            }, indent=2)

    except Exception as e:
        return handle_error(e, "update_tasks")


@mcp.tool(
    name="ticktick_complete_tasks",
    annotations={
        "title": "Complete Tasks",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_complete_tasks(params: CompleteTasksInput, ctx: Context) -> str:
    """
    Complete one or more tasks.

    Changes task status to completed and records completion time.
    Supports batch completion (1-100 tasks).

    Args:
        params: Completion parameters:
            - tasks (list, required): List of task identifiers (1-100 tasks)
              Each task must contain:
                - task_id (str): Task to complete
                - project_id (str): Project containing the task
            - response_format (str): 'markdown' (default) or 'json'

    Returns:
        Success confirmation or error message.

    Examples:
        Single task:
            tasks=[{"task_id": "abc123", "project_id": "proj1"}]

        Multiple tasks:
            tasks=[
                {"task_id": "abc1", "project_id": "proj1"},
                {"task_id": "abc2", "project_id": "proj1"}
            ]
    """
    try:
        client = get_client(ctx)
        task_ids = [(t.task_id, t.project_id) for t in params.tasks]
        await client.complete_tasks(task_ids)

        count = len(task_ids)
        if count == 1:
            return success_message(f"Task `{task_ids[0][0]}` marked as complete.")
        else:
            return success_message(f"{count} tasks marked as complete.")

    except Exception as e:
        return handle_error(e, "complete_tasks")


@mcp.tool(
    name="ticktick_delete_tasks",
    annotations={
        "title": "Delete Tasks",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_delete_tasks(params: DeleteTasksInput, ctx: Context) -> str:
    """
    Delete one or more tasks.

    Moves tasks to trash. Supports batch deletion (1-100 tasks).
    Can be undone from the TickTick trash.

    Args:
        params: Deletion parameters:
            - tasks (list, required): List of task identifiers (1-100 tasks)
              Each task must contain:
                - task_id (str): Task to delete
                - project_id (str): Project containing the task
            - response_format (str): 'markdown' (default) or 'json'

    Returns:
        Success confirmation or error message.
    """
    try:
        client = get_client(ctx)
        task_ids = [(t.task_id, t.project_id) for t in params.tasks]
        await client.delete_tasks(task_ids)

        count = len(task_ids)
        if count == 1:
            return success_message(f"Task `{task_ids[0][0]}` deleted.")
        else:
            return success_message(f"{count} tasks deleted.")

    except Exception as e:
        return handle_error(e, "delete_tasks")


@mcp.tool(
    name="ticktick_move_tasks",
    annotations={
        "title": "Move Tasks",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_move_tasks(params: MoveTasksInput, ctx: Context) -> str:
    """
    Move one or more tasks to different projects.

    Transfers tasks between projects while preserving all properties.
    Supports batch moves (1-100 tasks).

    Args:
        params: Move parameters:
            - moves (list, required): List of move specifications (1-100)
              Each move must contain:
                - task_id (str): Task to move
                - from_project_id (str): Source project
                - to_project_id (str): Destination project
            - response_format (str): 'markdown' (default) or 'json'

    Returns:
        Success confirmation or error message.

    Examples:
        Single task:
            moves=[{"task_id": "abc123", "from_project_id": "proj1", "to_project_id": "proj2"}]

        Multiple tasks to same project:
            moves=[
                {"task_id": "abc1", "from_project_id": "proj1", "to_project_id": "proj2"},
                {"task_id": "abc2", "from_project_id": "proj1", "to_project_id": "proj2"}
            ]
    """
    try:
        client = get_client(ctx)
        move_specs = [{
            "task_id": m.task_id,
            "from_project_id": m.from_project_id,
            "to_project_id": m.to_project_id,
        } for m in params.moves]
        await client.move_tasks(move_specs)

        count = len(move_specs)
        if count == 1:
            return success_message(f"Task `{move_specs[0]['task_id']}` moved to project `{move_specs[0]['to_project_id']}`.")
        else:
            return success_message(f"{count} tasks moved.")

    except Exception as e:
        return handle_error(e, "move_tasks")


@mcp.tool(
    name="ticktick_set_task_parents",
    annotations={
        "title": "Set Task Parents",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_set_task_parents(params: SetTaskParentsInput, ctx: Context) -> str:
    """
    Make one or more tasks into subtasks.

    Creates parent-child relationships between tasks. Child tasks will
    appear nested under their parent. Supports batch operations (1-50 tasks).

    Args:
        params: Parent assignment parameters:
            - tasks (list, required): List of parent assignments (1-50)
              Each assignment must contain:
                - task_id (str): Task to make a subtask
                - project_id (str): Project containing both tasks
                - parent_id (str): Parent task ID
            - response_format (str): 'markdown' (default) or 'json'

    Returns:
        Success confirmation or error message.

    Examples:
        Single subtask:
            tasks=[{"task_id": "child1", "project_id": "proj1", "parent_id": "parent1"}]

        Multiple subtasks under same parent:
            tasks=[
                {"task_id": "child1", "project_id": "proj1", "parent_id": "parent1"},
                {"task_id": "child2", "project_id": "proj1", "parent_id": "parent1"}
            ]
    """
    try:
        client = get_client(ctx)
        assignments = [{
            "task_id": t.task_id,
            "project_id": t.project_id,
            "parent_id": t.parent_id,
        } for t in params.tasks]
        await client.set_task_parents(assignments)

        count = len(assignments)
        if count == 1:
            return success_message(f"Task `{assignments[0]['task_id']}` is now a subtask of `{assignments[0]['parent_id']}`.")
        else:
            return success_message(f"{count} tasks assigned as subtasks.")

    except Exception as e:
        return handle_error(e, "set_task_parents")


@mcp.tool(
    name="ticktick_unparent_tasks",
    annotations={
        "title": "Unparent Tasks",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_unparent_tasks(params: UnparentTasksInput, ctx: Context) -> str:
    """
    Remove one or more tasks from their parents.

    Converts subtasks back into top-level tasks. Supports batch
    operations (1-50 tasks).

    Args:
        params: Unparent parameters:
            - tasks (list, required): List of tasks to unparent (1-50)
              Each task must contain:
                - task_id (str): Subtask to unparent
                - project_id (str): Project containing the task
            - response_format (str): 'markdown' (default) or 'json'

    Returns:
        Success confirmation or error message.

    Raises:
        Error if a task is not a subtask (has no parent).
    """
    try:
        client = get_client(ctx)
        unparent_specs = [{
            "task_id": t.task_id,
            "project_id": t.project_id,
        } for t in params.tasks]
        await client.unparent_tasks(unparent_specs)

        count = len(unparent_specs)
        if count == 1:
            return success_message(f"Task `{unparent_specs[0]['task_id']}` is now a top-level task.")
        else:
            return success_message(f"{count} tasks are now top-level tasks.")

    except Exception as e:
        return handle_error(e, "unparent_tasks")


@mcp.tool(
    name="ticktick_search_tasks",
    annotations={
        "title": "Search Tasks",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_search_tasks(params: SearchInput, ctx: Context) -> str:
    """
    Search for tasks by title or content.

    Performs a text search across all active tasks, matching the query
    against task titles and content.

    Args:
        params: Search parameters:
            - query (str): Search query (required)
            - limit (int): Maximum results (default 20)

    Returns:
        Formatted list of matching tasks or error message.

    Examples:
        - Search by keyword: query="meeting"
        - Search by phrase: query="quarterly report"
    """
    try:
        client = get_client(ctx)
        tasks = await client.search_tasks(params.query)
        tasks = tasks[: params.limit]

        title = f"Search Results: '{params.query}'"

        if params.response_format == ResponseFormat.MARKDOWN:
            return format_tasks_markdown(tasks, title)
        else:
            return json.dumps(format_tasks_json(tasks), indent=2)

    except Exception as e:
        return handle_error(e, "search_tasks")


# =============================================================================
# Task Pinning Tools
# =============================================================================


@mcp.tool(
    name="ticktick_pin_tasks",
    annotations={
        "title": "Pin/Unpin Tasks",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def ticktick_pin_tasks(params: PinTasksInput, ctx: Context) -> str:
    """
    Pin or unpin one or more tasks.

    Pinned tasks appear at the top of task lists in TickTick.
    Supports batch operations (1-50 tasks).

    Args:
        params: Pin operation parameters:
            - tasks (list, required): List of pin operations (1-50 tasks)
              Each operation must contain:
                - task_id (str): Task to pin/unpin
                - project_id (str): Project containing the task
                - pin (bool): True to pin, False to unpin (default True)
            - response_format (str): 'markdown' (default) or 'json'

    Returns:
        Success confirmation with updated task details.

    Examples:
        Pin single task:
            tasks=[{"task_id": "abc123", "project_id": "proj1", "pin": true}]

        Unpin task:
            tasks=[{"task_id": "abc123", "project_id": "proj1", "pin": false}]

        Pin multiple:
            tasks=[
                {"task_id": "abc1", "project_id": "proj1", "pin": true},
                {"task_id": "abc2", "project_id": "proj1", "pin": true}
            ]
    """
    try:
        client = get_client(ctx)
        pin_specs = [{
            "task_id": t.task_id,
            "project_id": t.project_id,
            "pin": t.pin,
        } for t in params.tasks]

        updated_tasks = await client.pin_tasks(pin_specs)

        count = len(updated_tasks)
        if count == 1:
            task = updated_tasks[0]
            action = "pinned" if params.tasks[0].pin else "unpinned"
            if params.response_format == ResponseFormat.MARKDOWN:
                return f"**Success**: Task '{task.title}' has been {action}.\n\n" + format_task_markdown(task)
            else:
                result = format_task_json(task)
                result["action"] = action
                return json.dumps(result, indent=2, default=str)
        else:
            if params.response_format == ResponseFormat.MARKDOWN:
                return f"**Success**: {count} tasks updated.\n\n{format_tasks_markdown(updated_tasks)}"
            else:
                return json.dumps({
                    "success": True,
                    "count": count,
                    "tasks": [format_task_json(t) for t in updated_tasks]
                }, indent=2, default=str)

    except Exception as e:
        return handle_error(e, "pin_tasks")


# =============================================================================
# Kanban Column Tools
# =============================================================================


@mcp.tool(
    name="ticktick_list_columns",
    annotations={
        "title": "List Kanban Columns",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def ticktick_list_columns(params: ColumnListInput, ctx: Context) -> str:
    """
    List all kanban columns for a project.

    Returns the columns in a kanban-view project, sorted by display order.
    Only kanban-view projects have columns.

    Args:
        params: Column list input with project_id

    Returns:
        List of columns
    """
    try:
        client = get_client(ctx)
        columns = await client.get_columns(params.project_id)

        if params.response_format == ResponseFormat.MARKDOWN:
            return format_columns_markdown(columns)
        else:
            return json.dumps(format_columns_json(columns), indent=2, default=str)

    except Exception as e:
        return handle_error(e, "list_columns")


@mcp.tool(
    name="ticktick_create_column",
    annotations={
        "title": "Create Kanban Column",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def ticktick_create_column(params: ColumnCreateInput, ctx: Context) -> str:
    """
    Create a new kanban column in a project.

    Columns organize tasks in kanban-view projects. Common column names
    include "To Do", "In Progress", "Review", and "Done".

    Args:
        params: Column create input with project_id, name, and optional sort_order

    Returns:
        Created column details
    """
    try:
        client = get_client(ctx)
        column = await client.create_column(
            project_id=params.project_id,
            name=params.name,
            sort_order=params.sort_order,
        )

        if params.response_format == ResponseFormat.MARKDOWN:
            return f"**Success**: Created column '{column.name}'\n\n" + format_column_markdown(column)
        else:
            result = format_column_json(column)
            result["action"] = "created"
            return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return handle_error(e, "create_column")


@mcp.tool(
    name="ticktick_update_column",
    annotations={
        "title": "Update Kanban Column",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def ticktick_update_column(params: ColumnUpdateInput, ctx: Context) -> str:
    """
    Update a kanban column's name or sort order.

    Args:
        params: Column update input with column_id, project_id, and optional name/sort_order

    Returns:
        Updated column details
    """
    try:
        client = get_client(ctx)
        column = await client.update_column(
            column_id=params.column_id,
            project_id=params.project_id,
            name=params.name,
            sort_order=params.sort_order,
        )

        if params.response_format == ResponseFormat.MARKDOWN:
            return f"**Success**: Updated column '{column.name}'\n\n" + format_column_markdown(column)
        else:
            result = format_column_json(column)
            result["action"] = "updated"
            return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return handle_error(e, "update_column")


@mcp.tool(
    name="ticktick_delete_column",
    annotations={
        "title": "Delete Kanban Column",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def ticktick_delete_column(params: ColumnDeleteInput, ctx: Context) -> str:
    """
    Delete a kanban column.

    Warning: Tasks in this column will become unassigned to any column.
    This operation cannot be undone.

    Args:
        params: Column delete input with column_id and project_id

    Returns:
        Confirmation message
    """
    try:
        client = get_client(ctx)
        await client.delete_column(params.column_id, params.project_id)
        return f"**Success**: Deleted column `{params.column_id}`"

    except Exception as e:
        return handle_error(e, "delete_column")


# =============================================================================
# Project Tools
# =============================================================================


@mcp.tool(
    name="ticktick_list_projects",
    annotations={
        "title": "List Projects",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_list_projects(ctx: Context, response_format: ResponseFormat = ResponseFormat.MARKDOWN) -> str:
    """
    List all projects.

    Retrieves all user projects including their IDs, names, colors,
    and organization settings.

    Returns:
        Formatted list of projects or error message.
    """
    try:
        client = get_client(ctx)
        projects = await client.get_all_projects()

        if response_format == ResponseFormat.MARKDOWN:
            return format_projects_markdown(projects)
        else:
            return json.dumps(format_projects_json(projects), indent=2)

    except Exception as e:
        return handle_error(e, "list_projects")


@mcp.tool(
    name="ticktick_get_project",
    annotations={
        "title": "Get Project",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_get_project(params: ProjectGetInput, ctx: Context) -> str:
    """
    Get a project by ID, optionally with its tasks.

    Retrieves project details and optionally all tasks within the project.

    Args:
        params: Query parameters:
            - project_id (str): Project identifier (required)
            - include_tasks (bool): Include project tasks (default False)

    Returns:
        Formatted project details or error message.
    """
    try:
        client = get_client(ctx)

        if params.include_tasks:
            project_data = await client.get_project_tasks(params.project_id)

            if params.response_format == ResponseFormat.MARKDOWN:
                lines = [format_project_markdown(project_data.project)]
                lines.append("")
                lines.append(format_tasks_markdown(project_data.tasks, "Tasks"))
                return "\n".join(lines)
            else:
                return json.dumps({
                    "project": format_project_json(project_data.project),
                    "tasks": format_tasks_json(project_data.tasks),
                }, indent=2)
        else:
            project = await client.get_project(params.project_id)

            if params.response_format == ResponseFormat.MARKDOWN:
                return format_project_markdown(project)
            else:
                return json.dumps(format_project_json(project), indent=2)

    except Exception as e:
        return handle_error(e, "get_project")


@mcp.tool(
    name="ticktick_create_project",
    annotations={
        "title": "Create Project",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def ticktick_create_project(params: ProjectCreateInput, ctx: Context) -> str:
    """
    Create a new project.

    Creates a new project/list for organizing tasks. Projects can have
    different view modes and be organized in folders.

    Args:
        params: Project creation parameters:
            - name (str): Project name (required)
            - color (str): Hex color code (e.g., '#F18181')
            - kind (str): 'TASK' or 'NOTE'
            - view_mode (str): 'list', 'kanban', or 'timeline'
            - folder_id (str): Parent folder ID

    Returns:
        Formatted project details or error message.
    """
    try:
        client = get_client(ctx)

        project = await client.create_project(
            name=params.name,
            color=params.color,
            kind=params.kind,
            view_mode=params.view_mode,
            folder_id=params.folder_id,
        )

        if params.response_format == ResponseFormat.MARKDOWN:
            return f"# Project Created\n\n{format_project_markdown(project)}"
        else:
            return json.dumps({"success": True, "project": format_project_json(project)}, indent=2)

    except Exception as e:
        return handle_error(e, "create_project")


@mcp.tool(
    name="ticktick_update_project",
    annotations={
        "title": "Update Project",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_update_project(params: ProjectUpdateInput, ctx: Context) -> str:
    """
    Update a project's properties.

    Updates project name, color, or folder assignment.

    Args:
        params: Update parameters:
            - project_id (str): Project to update (required)
            - name (str): New project name
            - color (str): New hex color code (e.g., '#F18181')
            - folder_id (str): New folder ID (use 'NONE' to remove from folder)

    Returns:
        Formatted updated project or error message.
    """
    try:
        client = get_client(ctx)

        # Handle "NONE" to remove from folder
        folder_id = params.folder_id
        if folder_id and folder_id.upper() == "NONE":
            folder_id = ""  # Empty string removes from folder

        project = await client.update_project(
            project_id=params.project_id,
            name=params.name,
            color=params.color,
            folder_id=folder_id,
        )

        if params.response_format == ResponseFormat.MARKDOWN:
            return f"# Project Updated\n\n{format_project_markdown(project)}"
        else:
            return json.dumps({"success": True, "project": format_project_json(project)}, indent=2)

    except Exception as e:
        return handle_error(e, "update_project")


@mcp.tool(
    name="ticktick_delete_project",
    annotations={
        "title": "Delete Project",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_delete_project(params: ProjectDeleteInput, ctx: Context) -> str:
    """
    Delete a project.

    Permanently deletes a project and all its tasks. This is a destructive
    operation that cannot be undone.

    Args:
        params: Deletion parameters:
            - project_id (str): Project to delete (required)

    Returns:
        Success confirmation or error message.
    """
    try:
        client = get_client(ctx)
        await client.delete_project(params.project_id)
        return success_message(f"Project `{params.project_id}` deleted.")

    except Exception as e:
        return handle_error(e, "delete_project")


# =============================================================================
# Folder Tools
# =============================================================================


@mcp.tool(
    name="ticktick_list_folders",
    annotations={
        "title": "List Folders",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_list_folders(ctx: Context, response_format: ResponseFormat = ResponseFormat.MARKDOWN) -> str:
    """
    List all folders (project groups).

    Retrieves all folders used to organize projects.

    Returns:
        Formatted list of folders or error message.
    """
    try:
        client = get_client(ctx)
        folders = await client.get_all_folders()

        if response_format == ResponseFormat.MARKDOWN:
            return format_folders_markdown(folders)
        else:
            return json.dumps(format_folders_json(folders), indent=2)

    except Exception as e:
        return handle_error(e, "list_folders")


@mcp.tool(
    name="ticktick_create_folder",
    annotations={
        "title": "Create Folder",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def ticktick_create_folder(params: FolderCreateInput, ctx: Context) -> str:
    """
    Create a new folder for organizing projects.

    Args:
        params: Folder creation parameters:
            - name (str): Folder name (required)

    Returns:
        Formatted folder details or error message.
    """
    try:
        client = get_client(ctx)
        folder = await client.create_folder(params.name)

        if params.response_format == ResponseFormat.MARKDOWN:
            return f"# Folder Created\n\n- **{folder.name}** (`{folder.id}`)"
        else:
            return json.dumps({"success": True, "folder": {"id": folder.id, "name": folder.name}}, indent=2)

    except Exception as e:
        return handle_error(e, "create_folder")


@mcp.tool(
    name="ticktick_rename_folder",
    annotations={
        "title": "Rename Folder",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_rename_folder(params: FolderRenameInput, ctx: Context) -> str:
    """
    Rename a folder.

    Args:
        params: Rename parameters:
            - folder_id (str): Folder to rename (required)
            - name (str): New folder name (required)

    Returns:
        Formatted updated folder or error message.
    """
    try:
        client = get_client(ctx)
        folder = await client.rename_folder(params.folder_id, params.name)

        if params.response_format == ResponseFormat.MARKDOWN:
            return f"# Folder Renamed\n\n- **{folder.name}** (`{folder.id}`)"
        else:
            return json.dumps({"success": True, "folder": {"id": folder.id, "name": folder.name}}, indent=2)

    except Exception as e:
        return handle_error(e, "rename_folder")


@mcp.tool(
    name="ticktick_delete_folder",
    annotations={
        "title": "Delete Folder",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_delete_folder(params: FolderDeleteInput, ctx: Context) -> str:
    """
    Delete a folder.

    Deletes a folder. Projects in the folder are not deleted but become
    ungrouped.

    Args:
        params: Deletion parameters:
            - folder_id (str): Folder to delete (required)

    Returns:
        Success confirmation or error message.
    """
    try:
        client = get_client(ctx)
        await client.delete_folder(params.folder_id)
        return success_message(f"Folder `{params.folder_id}` deleted.")

    except Exception as e:
        return handle_error(e, "delete_folder")


# =============================================================================
# Tag Tools
# =============================================================================


@mcp.tool(
    name="ticktick_list_tags",
    annotations={
        "title": "List Tags",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_list_tags(ctx: Context, response_format: ResponseFormat = ResponseFormat.MARKDOWN) -> str:
    """
    List all tags.

    Retrieves all tags with their colors and hierarchy.

    Returns:
        Formatted list of tags or error message.
    """
    try:
        client = get_client(ctx)
        tags = await client.get_all_tags()

        if response_format == ResponseFormat.MARKDOWN:
            return format_tags_markdown(tags)
        else:
            return json.dumps(format_tags_json(tags), indent=2)

    except Exception as e:
        return handle_error(e, "list_tags")


@mcp.tool(
    name="ticktick_create_tag",
    annotations={
        "title": "Create Tag",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def ticktick_create_tag(params: TagCreateInput, ctx: Context) -> str:
    """
    Create a new tag.

    Tags are used to categorize and filter tasks across projects.

    Args:
        params: Tag creation parameters:
            - name (str): Tag name (required)
            - color (str): Hex color code
            - parent (str): Parent tag name for nesting

    Returns:
        Formatted tag details or error message.
    """
    try:
        client = get_client(ctx)
        tag = await client.create_tag(params.name, color=params.color, parent=params.parent)

        if params.response_format == ResponseFormat.MARKDOWN:
            return f"# Tag Created\n\n{format_tag_markdown(tag)}"
        else:
            return json.dumps({"success": True, "tag": format_tag_json(tag)}, indent=2)

    except Exception as e:
        return handle_error(e, "create_tag")


@mcp.tool(
    name="ticktick_update_tag",
    annotations={
        "title": "Update Tag",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_update_tag(params: TagUpdateInput, ctx: Context) -> str:
    """
    Update a tag's properties.

    Updates tag color, parent, or label (rename). If label is provided,
    the tag is renamed first, then other updates are applied.

    Args:
        params: Update parameters:
            - name (str): Current tag name to update (required)
            - color (str): New hex color code (e.g., '#F18181')
            - parent (str): New parent tag name (empty string to remove parent)
            - label (str): New display name/label for the tag (rename)

    Returns:
        Formatted updated tag or error message.

    Examples:
        Change color:
            name="work", color="#F18181"

        Rename tag:
            name="old-name", label="new-name"

        Rename and change color:
            name="old-name", label="new-name", color="#FF0000"
    """
    try:
        client = get_client(ctx)

        # Handle rename if label is provided
        if params.label:
            await client.rename_tag(params.name, params.label)
            # After rename, update the name we use for subsequent operations
            tag_name = params.label
        else:
            tag_name = params.name

        # Handle empty string as None to remove parent
        parent = params.parent
        if parent == "":
            parent = None

        # Apply other updates if any
        if params.color or parent is not None:
            tag = await client.update_tag(tag_name, color=params.color, parent=parent)
        else:
            # If only renamed, get the tag to return it
            tags = await client.get_all_tags()
            tag = next((t for t in tags if t.name.lower() == tag_name.lower()), None)
            if not tag:
                return success_message(f"Tag renamed to `{tag_name}`")

        if params.response_format == ResponseFormat.MARKDOWN:
            return f"# Tag Updated\n\n{format_tag_markdown(tag)}"
        else:
            return json.dumps({"success": True, "tag": format_tag_json(tag)}, indent=2)

    except Exception as e:
        return handle_error(e, "update_tag")


@mcp.tool(
    name="ticktick_delete_tag",
    annotations={
        "title": "Delete Tag",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_delete_tag(params: TagDeleteInput, ctx: Context) -> str:
    """
    Delete a tag.

    Removes the tag. Tasks with this tag will no longer have it.

    Args:
        params: Deletion parameters:
            - name (str): Tag name to delete (required)

    Returns:
        Success confirmation or error message.
    """
    try:
        client = get_client(ctx)
        await client.delete_tag(params.name)
        return success_message(f"Tag `{params.name}` deleted.")

    except Exception as e:
        return handle_error(e, "delete_tag")


@mcp.tool(
    name="ticktick_merge_tags",
    annotations={
        "title": "Merge Tags",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_merge_tags(params: TagMergeInput, ctx: Context) -> str:
    """
    Merge one tag into another.

    Moves all tasks from the source tag to the target tag, then deletes
    the source tag.

    Args:
        params: Merge parameters:
            - source (str): Tag to merge from (will be deleted)
            - target (str): Tag to merge into (will remain)

    Returns:
        Success confirmation or error message.
    """
    try:
        client = get_client(ctx)
        await client.merge_tags(params.source, params.target)
        return success_message(f"Tag `{params.source}` merged into `{params.target}`.")

    except Exception as e:
        return handle_error(e, "merge_tags")


# =============================================================================
# User Tools
# =============================================================================


@mcp.tool(
    name="ticktick_get_profile",
    annotations={
        "title": "Get User Profile",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_get_profile(ctx: Context, response_format: ResponseFormat = ResponseFormat.MARKDOWN) -> str:
    """
    Get user profile information.

    Retrieves the current user's profile including username, display name,
    and account settings.

    Returns:
        Formatted user profile or error message.
    """
    try:
        client = get_client(ctx)
        user = await client.get_profile()

        if response_format == ResponseFormat.MARKDOWN:
            return format_user_markdown(user)
        else:
            return json.dumps({
                "username": user.username,
                "display_name": user.display_name,
                "name": user.name,
                "email": user.email,
                "locale": user.locale,
                "verified_email": user.verified_email,
            }, indent=2)

    except Exception as e:
        return handle_error(e, "get_profile")


@mcp.tool(
    name="ticktick_get_status",
    annotations={
        "title": "Get Account Status",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_get_status(ctx: Context, response_format: ResponseFormat = ResponseFormat.MARKDOWN) -> str:
    """
    Get account status and subscription information.

    Retrieves subscription status, Pro account details, and team membership.

    Returns:
        Formatted account status or error message.
    """
    try:
        client = get_client(ctx)
        status = await client.get_status()

        if response_format == ResponseFormat.MARKDOWN:
            return format_user_status_markdown(status)
        else:
            return json.dumps({
                "user_id": status.user_id,
                "username": status.username,
                "inbox_id": status.inbox_id,
                "is_pro": status.is_pro,
                "pro_end_date": status.pro_end_date,
                "team_user": status.team_user,
            }, indent=2)

    except Exception as e:
        return handle_error(e, "get_status")


@mcp.tool(
    name="ticktick_get_statistics",
    annotations={
        "title": "Get Productivity Statistics",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_get_statistics(ctx: Context, response_format: ResponseFormat = ResponseFormat.MARKDOWN) -> str:
    """
    Get productivity statistics.

    Retrieves task completion statistics, scores, levels, and focus/pomodoro data.

    Returns:
        Formatted statistics or error message.
    """
    try:
        client = get_client(ctx)
        stats = await client.get_statistics()

        if response_format == ResponseFormat.MARKDOWN:
            return format_statistics_markdown(stats)
        else:
            return json.dumps({
                "level": stats.level,
                "score": stats.score,
                "today_completed": stats.today_completed,
                "yesterday_completed": stats.yesterday_completed,
                "total_completed": stats.total_completed,
                "today_pomo_count": stats.today_pomo_count,
                "total_pomo_count": stats.total_pomo_count,
                "total_pomo_duration_hours": stats.total_pomo_duration_hours,
            }, indent=2)

    except Exception as e:
        return handle_error(e, "get_statistics")


@mcp.tool(
    name="ticktick_get_preferences",
    annotations={
        "title": "Get User Preferences",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_get_preferences(ctx: Context) -> str:
    """
    Get user preferences and settings.

    Retrieves user-configurable settings including:
    - timeZone: User's timezone
    - weekStartDay: First day of week (0=Sunday, 1=Monday)
    - startOfDay: Hour when day starts
    - dateFormat: Date display format
    - timeFormat: Time display format (12h/24h)
    - defaultReminder: Default reminder setting
    - And other user preferences

    Returns:
        JSON object with all user preferences.
    """
    try:
        client = get_client(ctx)
        preferences = await client.get_preferences()
        return json.dumps(preferences, indent=2)

    except Exception as e:
        return handle_error(e, "get_preferences")


# =============================================================================
# Focus Tools
# =============================================================================


@mcp.tool(
    name="ticktick_focus_heatmap",
    annotations={
        "title": "Get Focus Heatmap",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_focus_heatmap(params: FocusStatsInput, ctx: Context) -> str:
    """
    Get focus/pomodoro heatmap data.

    Retrieves focus time data for visualization as a heatmap.

    Args:
        params: Query parameters:
            - start_date (str): Start date (YYYY-MM-DD)
            - end_date (str): End date (YYYY-MM-DD)
            - days (int): Days to look back if dates not specified

    Returns:
        Focus heatmap data or error message.
    """
    try:
        client = get_client(ctx)

        end_date = date.fromisoformat(params.end_date) if params.end_date else date.today()
        start_date = date.fromisoformat(params.start_date) if params.start_date else end_date - timedelta(days=params.days)

        data = await client.get_focus_heatmap(start_date, end_date)

        if params.response_format == ResponseFormat.MARKDOWN:
            lines = ["# Focus Heatmap", "", f"Period: {start_date} to {end_date}", ""]
            total_duration = sum(d.get("duration", 0) for d in data)
            hours = total_duration / 3600
            lines.append(f"Total Focus Time: {hours:.1f} hours")
            return "\n".join(lines)
        else:
            return json.dumps({
                "start_date": str(start_date),
                "end_date": str(end_date),
                "data": data,
            }, indent=2)

    except Exception as e:
        return handle_error(e, "focus_heatmap")


@mcp.tool(
    name="ticktick_focus_by_tag",
    annotations={
        "title": "Get Focus Time by Tag",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_focus_by_tag(params: FocusStatsInput, ctx: Context) -> str:
    """
    Get focus time distribution by tag.

    Shows how focus time is distributed across different tags.

    Args:
        params: Query parameters:
            - start_date (str): Start date (YYYY-MM-DD)
            - end_date (str): End date (YYYY-MM-DD)
            - days (int): Days to look back if dates not specified

    Returns:
        Focus distribution by tag or error message.
    """
    try:
        client = get_client(ctx)

        end_date = date.fromisoformat(params.end_date) if params.end_date else date.today()
        start_date = date.fromisoformat(params.start_date) if params.start_date else end_date - timedelta(days=params.days)

        data = await client.get_focus_by_tag(start_date, end_date)

        if params.response_format == ResponseFormat.MARKDOWN:
            lines = ["# Focus Time by Tag", "", f"Period: {start_date} to {end_date}", ""]

            if not data:
                lines.append("No focus data for this period.")
            else:
                for tag, seconds in sorted(data.items(), key=lambda x: x[1], reverse=True):
                    hours = seconds / 3600
                    lines.append(f"- **{tag}**: {hours:.1f} hours")

            return "\n".join(lines)
        else:
            return json.dumps({
                "start_date": str(start_date),
                "end_date": str(end_date),
                "tag_durations": data,
            }, indent=2)

    except Exception as e:
        return handle_error(e, "focus_by_tag")


# =============================================================================
# Habit Tools
# =============================================================================


def format_habit_markdown(habit: Habit) -> str:
    """Format a habit for markdown display."""
    lines = [
        f"## {habit.name}",
        f"- **ID**: `{habit.id}`",
        f"- **Type**: {habit.habit_type}",
        f"- **Goal**: {habit.goal} {habit.unit}",
    ]

    if habit.is_numeric:
        lines.append(f"- **Step**: +{habit.step}")

    lines.append(f"- **Status**: {'Archived' if habit.is_archived else 'Active'}")
    lines.append(f"- **Total Check-ins**: {habit.total_checkins}")
    lines.append(f"- **Current Streak**: {habit.current_streak}")

    if habit.target_days > 0:
        lines.append(f"- **Target**: {habit.target_days} days")

    if habit.color:
        lines.append(f"- **Color**: {habit.color}")

    if habit.repeat_rule:
        lines.append(f"- **Repeat**: `{habit.repeat_rule}`")

    if habit.reminders:
        lines.append(f"- **Reminders**: {', '.join(habit.reminders)}")

    if habit.encouragement:
        lines.append(f"- **Encouragement**: {habit.encouragement}")

    return "\n".join(lines)


def format_habit_json(habit: Habit) -> dict[str, Any]:
    """Format a habit for JSON output."""
    return {
        "id": habit.id,
        "name": habit.name,
        "type": habit.habit_type,
        "goal": habit.goal,
        "step": habit.step,
        "unit": habit.unit,
        "status": "archived" if habit.is_archived else "active",
        "total_checkins": habit.total_checkins,
        "current_streak": habit.current_streak,
        "target_days": habit.target_days,
        "color": habit.color,
        "icon": habit.icon,
        "repeat_rule": habit.repeat_rule,
        "reminders": habit.reminders,
        "section_id": habit.section_id,
        "encouragement": habit.encouragement,
        "created_time": habit.created_time.isoformat() if habit.created_time else None,
        "modified_time": habit.modified_time.isoformat() if habit.modified_time else None,
    }


def format_habits_markdown(habits: list[Habit], title: str = "Habits") -> str:
    """Format multiple habits for markdown display."""
    if not habits:
        return f"# {title}\n\nNo habits found."

    lines = [f"# {title}", f"*{len(habits)} habits*", ""]

    for habit in habits:
        status = "📦" if habit.is_archived else "✅" if habit.current_streak > 0 else "⏳"
        lines.append(f"### {status} {habit.name}")
        lines.append(f"- **ID**: `{habit.id}`")
        lines.append(f"- **Type**: {habit.habit_type}")
        lines.append(f"- **Streak**: {habit.current_streak} | Total: {habit.total_checkins}")
        if habit.target_days > 0:
            lines.append(f"- **Target**: {habit.target_days} days")
        lines.append("")

    return "\n".join(lines)


def format_habits_json(habits: list[Habit]) -> list[dict[str, Any]]:
    """Format multiple habits for JSON output."""
    return [format_habit_json(h) for h in habits]


def format_section_markdown(section: HabitSection) -> str:
    """Format a habit section for markdown display."""
    return f"- **{section.display_name}** (`{section.id}`)"


def format_sections_json(sections: list[HabitSection]) -> list[dict[str, Any]]:
    """Format habit sections for JSON output."""
    return [
        {"id": s.id, "name": s.name, "display_name": s.display_name}
        for s in sections
    ]


@mcp.tool(
    name="ticktick_habits",
    annotations={
        "title": "List Habits",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_habits(params: HabitListInput, ctx: Context) -> str:
    """
    List all habits.

    Retrieves all habits including their status, streaks, and goals.

    Args:
        params: Query parameters:
            - include_archived (bool): Include archived habits (default: False)
            - response_format (str): Output format ("markdown" or "json")

    Returns:
        List of habits with their details.
    """
    try:
        client = get_client(ctx)
        habits = await client.get_all_habits()

        if not params.include_archived:
            habits = [h for h in habits if h.is_active]

        if params.response_format == ResponseFormat.MARKDOWN:
            return format_habits_markdown(habits)
        else:
            return json.dumps(format_habits_json(habits), indent=2)

    except Exception as e:
        return handle_error(e, "list_habits")


@mcp.tool(
    name="ticktick_habit",
    annotations={
        "title": "Get Habit",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_habit(params: HabitGetInput, ctx: Context) -> str:
    """
    Get a specific habit by ID.

    Args:
        params: Query parameters:
            - habit_id (str): Habit ID (required)
            - response_format (str): Output format

    Returns:
        Habit details.
    """
    try:
        client = get_client(ctx)
        habit = await client.get_habit(params.habit_id)

        if params.response_format == ResponseFormat.MARKDOWN:
            return format_habit_markdown(habit)
        else:
            return json.dumps(format_habit_json(habit), indent=2)

    except Exception as e:
        return handle_error(e, "get_habit")


@mcp.tool(
    name="ticktick_habit_sections",
    annotations={
        "title": "List Habit Sections",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_habit_sections(ctx: Context, response_format: ResponseFormat = ResponseFormat.MARKDOWN) -> str:
    """
    List habit sections (time-of-day groupings).

    Sections organize habits by time of day: Morning, Afternoon, Night.

    Returns:
        List of habit sections with their IDs.
    """
    try:
        client = get_client(ctx)
        sections = await client.get_habit_sections()

        if response_format == ResponseFormat.MARKDOWN:
            lines = ["# Habit Sections", ""]
            for section in sections:
                lines.append(format_section_markdown(section))
            return "\n".join(lines)
        else:
            return json.dumps(format_sections_json(sections), indent=2)

    except Exception as e:
        return handle_error(e, "habit_sections")


@mcp.tool(
    name="ticktick_create_habit",
    annotations={
        "title": "Create Habit",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def ticktick_create_habit(params: HabitCreateInput, ctx: Context) -> str:
    """
    Create a new habit.

    Creates a habit with the specified configuration. Habits can be boolean
    (yes/no) or numeric (count/measure).

    Args:
        params: Habit parameters:
            - name (str): Habit name (required)
            - habit_type (str): "Boolean" or "Real" (default: Boolean)
            - goal (float): Target value (default: 1.0)
            - step (float): Increment for numeric (default: 1.0)
            - unit (str): Unit of measurement (default: Count)
            - color (str): Hex color (optional)
            - section_id (str): Time-of-day section (optional)
            - repeat_rule (str): RRULE pattern (default: daily)
            - reminders (list[str]): Times in HH:MM format
            - target_days (int): Goal in days (0 = no target)
            - encouragement (str): Motivational message

    Returns:
        Created habit details.
    """
    try:
        client = get_client(ctx)
        habit = await client.create_habit(
            name=params.name,
            habit_type=params.habit_type,
            goal=params.goal,
            step=params.step,
            unit=params.unit,
            color=params.color or "#97E38B",
            section_id=params.section_id,
            repeat_rule=params.repeat_rule,
            reminders=params.reminders,
            target_days=params.target_days,
            encouragement=params.encouragement,
        )

        if params.response_format == ResponseFormat.MARKDOWN:
            return f"# Habit Created\n\n{format_habit_markdown(habit)}"
        else:
            return json.dumps({"success": True, "habit": format_habit_json(habit)}, indent=2)

    except Exception as e:
        return handle_error(e, "create_habit")


@mcp.tool(
    name="ticktick_update_habit",
    annotations={
        "title": "Update Habit",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_update_habit(params: HabitUpdateInput, ctx: Context) -> str:
    """
    Update a habit's properties.

    Includes archive/unarchive functionality via the archived field.

    Args:
        params: Update parameters:
            - habit_id (str): Habit ID (required)
            - name (str): New name
            - goal (float): New goal
            - step (float): New step
            - unit (str): New unit
            - color (str): New hex color
            - section_id (str): New section ID
            - repeat_rule (str): New RRULE pattern
            - reminders (list[str]): New reminders
            - target_days (int): New target days
            - encouragement (str): New message
            - archived (bool): Set true to archive, false to unarchive

    Returns:
        Updated habit details.

    Examples:
        Update name:
            habit_id="abc123", name="New Habit Name"

        Archive habit:
            habit_id="abc123", archived=true

        Unarchive habit:
            habit_id="abc123", archived=false
    """
    try:
        client = get_client(ctx)

        # Handle archive/unarchive via the archived field
        if params.archived is not None:
            if params.archived:
                habit = await client.archive_habit(params.habit_id)
                action = "archived"
            else:
                habit = await client.unarchive_habit(params.habit_id)
                action = "unarchived"

            # If only archiving/unarchiving (no other updates), return early
            if not any([params.name, params.goal, params.step, params.unit, params.color,
                       params.section_id, params.repeat_rule, params.reminders,
                       params.target_days, params.encouragement]):
                if params.response_format == ResponseFormat.MARKDOWN:
                    return f"# Habit {action.capitalize()}\n\n**{habit.name}** has been {action}."
                else:
                    return json.dumps({"success": True, "action": action, "habit": format_habit_json(habit)}, indent=2)

        # Apply other updates
        habit = await client.update_habit(
            habit_id=params.habit_id,
            name=params.name,
            goal=params.goal,
            step=params.step,
            unit=params.unit,
            color=params.color,
            section_id=params.section_id,
            repeat_rule=params.repeat_rule,
            reminders=params.reminders,
            target_days=params.target_days,
            encouragement=params.encouragement,
        )

        if params.response_format == ResponseFormat.MARKDOWN:
            return f"# Habit Updated\n\n{format_habit_markdown(habit)}"
        else:
            return json.dumps({"success": True, "habit": format_habit_json(habit)}, indent=2)

    except Exception as e:
        return handle_error(e, "update_habit")


@mcp.tool(
    name="ticktick_delete_habit",
    annotations={
        "title": "Delete Habit",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_delete_habit(params: HabitDeleteInput, ctx: Context) -> str:
    """
    Delete a habit.

    Permanently removes the habit and all its check-in history.

    Args:
        params: Delete parameters:
            - habit_id (str): Habit ID to delete (required)

    Returns:
        Confirmation message.
    """
    try:
        client = get_client(ctx)
        await client.delete_habit(params.habit_id)
        return success_message(f"Habit `{params.habit_id}` deleted successfully.")

    except Exception as e:
        return handle_error(e, "delete_habit")


@mcp.tool(
    name="ticktick_checkin_habits",
    annotations={
        "title": "Check In Habits",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def ticktick_checkin_habits(params: CheckinHabitsInput, ctx: Context) -> str:
    """
    Check in one or more habits for today or past dates.

    Records check-ins for habits. For each check-in:
    - If no date is provided, checks in for today (increments both total and streak)
    - If a past date is provided (backdating), only increments total (streak unaffected)

    This is useful for:
    - Checking in multiple habits at once
    - Migrating habit history from another app
    - Backdating missed check-ins

    Args:
        params: Check-in parameters:
            - checkins (list): List of check-ins, each containing:
                - habit_id (str): Habit ID (required)
                - value (float): Check-in value (default: 1.0)
                - checkin_date (str): Date to check in for (YYYY-MM-DD, optional)

    Returns:
        Updated habits with new totals.
    """
    try:
        client = get_client(ctx)

        # Build checkins list for batch operation
        checkin_data = []
        for checkin in params.checkins:
            checkin_data.append({
                "habit_id": checkin.habit_id,
                "value": checkin.value,
                "checkin_date": checkin.checkin_date,
            })

        # Call batch method
        results = await client.checkin_habits(checkin_data)

        if params.response_format == ResponseFormat.MARKDOWN:
            lines = [f"# {len(results)} Habit Check-in(s) Recorded", ""]
            today_str = date.today().isoformat()

            for habit_id, habit in results.items():
                # Find the corresponding checkin data
                checkin_info = next(
                    (c for c in params.checkins if c.habit_id == habit_id),
                    None
                )
                date_str = (
                    checkin_info.checkin_date
                    if checkin_info and checkin_info.checkin_date
                    else "today"
                )

                lines.append(f"## {habit.name}")
                lines.append(f"- **Date**: {date_str}")
                lines.append(f"- **Total Check-ins**: {habit.total_checkins}")
                lines.append(f"- **Current Streak**: {habit.current_streak}")
                lines.append("")

            # Add backdating note if any past dates
            has_backdated = any(
                c.checkin_date and c.checkin_date < today_str
                for c in params.checkins
            )
            if has_backdated:
                lines.append("*Note: Backdated check-ins don't affect the current streak.*")

            return "\n".join(lines)
        else:
            return json.dumps({
                "success": True,
                "count": len(results),
                "habits": {
                    habit_id: format_habit_json(habit)
                    for habit_id, habit in results.items()
                },
            }, indent=2)

    except Exception as e:
        return handle_error(e, "checkin_habits")


@mcp.tool(
    name="ticktick_habit_checkins",
    annotations={
        "title": "Get Habit Check-in History",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ticktick_habit_checkins(params: HabitCheckinsInput, ctx: Context) -> str:
    """
    Get habit check-in history.

    Retrieves check-in records for the specified habits.

    Args:
        params: Query parameters:
            - habit_ids (list[str]): List of habit IDs to query (required)
            - after_stamp (int): Date stamp (YYYYMMDD) to get check-ins after

    Returns:
        Check-in history for each habit.
    """
    try:
        client = get_client(ctx)
        data = await client.get_habit_checkins(
            habit_ids=params.habit_ids,
            after_stamp=params.after_stamp,
        )

        if params.response_format == ResponseFormat.MARKDOWN:
            lines = ["# Habit Check-in History", ""]
            for habit_id, checkins in data.items():
                lines.append(f"## Habit `{habit_id}`")
                if not checkins:
                    lines.append("No check-ins found.")
                else:
                    for checkin in checkins:
                        lines.append(f"- {checkin.checkin_stamp}: {checkin.value}")
                lines.append("")
            return "\n".join(lines)
        else:
            # Convert HabitCheckin objects to dicts
            result = {}
            for habit_id, checkins in data.items():
                result[habit_id] = [
                    {
                        "checkin_stamp": c.checkin_stamp,
                        "value": c.value,
                        "goal": c.goal,
                        "status": c.status,
                    }
                    for c in checkins
                ]
            return json.dumps(result, indent=2)

    except Exception as e:
        return handle_error(e, "habit_checkins")


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point for the TickTick MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
