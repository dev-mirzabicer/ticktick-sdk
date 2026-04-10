"""
Pydantic Input Models for TickTick SDK Tools.

This module defines all input validation models used by MCP tools.
Each model includes proper field constraints, descriptions, and examples.

=== Batch 2 Consolidation (v0.4.0) ===

This version introduces list-based mutations for batch operations:
- Every create/update/delete operation accepts a list of items
- A list of 1 = single operation, list of 50 = batch operation
- Consolidated tools: list_tasks absorbs completed/abandoned/deleted queries
- update_tasks includes column assignment (replaces move_task_to_column)
- update_tag includes label rename (replaces rename_tag)
- update_habit includes archived field (replaces archive/unarchive_habit)
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional, List, Literal

from pydantic import BaseModel, Field, ConfigDict, field_validator


class ResponseFormat(str, Enum):
    """Output format for tool responses."""

    MARKDOWN = "markdown"
    JSON = "json"


class BaseMCPInput(BaseModel):
    """Base input model with common configuration."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )


# =============================================================================
# Task Input Models - List-Based for Batch Operations
# =============================================================================


class TaskCreateItem(BaseModel):
    """Single task creation specification for batch operations."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    title: str = Field(
        ...,
        description="Task title (required)",
        min_length=1,
        max_length=500,
    )
    project_id: Optional[str] = Field(
        default=None,
        description="Project ID to create the task in. If not provided, uses inbox.",
        pattern=r"^(inbox\d+|[a-f0-9]{24})$",
    )
    content: Optional[str] = Field(
        default=None,
        description="Task notes/content (supports markdown)",
        max_length=10000,
    )
    description: Optional[str] = Field(
        default=None,
        description="Checklist description",
        max_length=5000,
    )
    priority: Optional[str] = Field(
        default=None,
        description="Priority level: 'none' (0), 'low' (1), 'medium' (3), 'high' (5)",
        pattern=r"^(none|low|medium|high|0|1|3|5)$",
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Start date in ISO format (e.g., '2025-01-15T09:00:00' or '2025-01-15')",
    )
    due_date: Optional[str] = Field(
        default=None,
        description="Due date in ISO format (e.g., '2025-01-15T17:00:00' or '2025-01-15')",
    )
    all_day: Optional[bool] = Field(
        default=None,
        description="Whether this is an all-day task (no specific time)",
    )
    time_zone: Optional[str] = Field(
        default=None,
        description="IANA timezone (e.g., 'America/New_York', 'UTC')",
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="List of tag names to apply (e.g., ['work', 'urgent'])",
        max_length=20,
    )
    reminders: Optional[List[str]] = Field(
        default=None,
        description="Reminder triggers in iCal format (e.g., 'TRIGGER:-PT30M' for 30 min before)",
        max_length=10,
    )
    recurrence: Optional[str] = Field(
        default=None,
        description="Recurrence rule in RRULE format (e.g., 'RRULE:FREQ=DAILY;INTERVAL=1'). Requires start_date.",
    )
    parent_id: Optional[str] = Field(
        default=None,
        description="Parent task ID to make this a subtask. The SDK handles parent assignment after creation.",
        pattern=r"^[a-f0-9]{24}$",
    )
    kind: Optional[str] = Field(
        default=None,
        description=(
            "Task type/kind: 'TEXT' (standard task, default), 'NOTE' (note), "
            "'CHECKLIST' (checklist with items). Must be uppercase."
        ),
        pattern=r"^(TEXT|NOTE|CHECKLIST)$",
    )
    items: Optional[List[str]] = Field(
        default=None,
        description=(
            "Checklist item titles. Only used when kind='CHECKLIST'. "
            "Each string becomes a checklist item. "
            "Example: ['Buy milk', 'Buy eggs', 'Buy bread']"
        ),
        max_length=100,
    )

    @field_validator("priority")
    @classmethod
    def normalize_priority(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return v.lower()

    @field_validator("items")
    @classmethod
    def validate_items(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            for item in v:
                if not item.strip():
                    raise ValueError("Checklist item titles must not be empty")
        return v


class CreateTasksInput(BaseMCPInput):
    """Create one or more tasks."""

    tasks: List[TaskCreateItem] = Field(
        ...,
        description="Tasks to create (1-50). Each task requires at least a title.",
        min_length=1,
        max_length=50,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable, 'json' for machine-readable",
    )


class TaskUpdateItem(BaseModel):
    """Single task update specification for batch operations."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task_id: str = Field(
        ...,
        description="Task identifier to update (required)",
        pattern=r"^[a-f0-9]{24}$",
    )
    project_id: str = Field(
        ...,
        description="Project ID the task belongs to (required)",
        pattern=r"^(inbox\d+|[a-f0-9]{24})$",
    )
    title: Optional[str] = Field(
        default=None,
        description="New task title",
        min_length=1,
        max_length=500,
    )
    content: Optional[str] = Field(
        default=None,
        description="New task content",
        max_length=10000,
    )
    priority: Optional[str] = Field(
        default=None,
        description="New priority: 'none', 'low', 'medium', 'high'",
        pattern=r"^(none|low|medium|high|0|1|3|5)$",
    )
    start_date: Optional[str] = Field(
        default=None,
        description="New start date in ISO format",
    )
    due_date: Optional[str] = Field(
        default=None,
        description="New due date in ISO format",
    )
    all_day: Optional[bool] = Field(
        default=None,
        description="All-day flag",
    )
    time_zone: Optional[str] = Field(
        default=None,
        description="New timezone",
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="New list of tags (replaces existing)",
        max_length=20,
    )
    recurrence: Optional[str] = Field(
        default=None,
        description="New recurrence rule in RRULE format",
    )
    # Column assignment (absorbs move_task_to_column)
    column_id: Optional[str] = Field(
        default=None,
        description="Kanban column ID to assign task to. Use empty string '' to remove from column.",
    )
    kind: Optional[str] = Field(
        default=None,
        description=(
            "Task type/kind: 'TEXT', 'NOTE', or 'CHECKLIST'. "
            "Changing kind may affect task behavior."
        ),
        pattern=r"^(TEXT|NOTE|CHECKLIST)$",
    )


class UpdateTasksInput(BaseMCPInput):
    """Update one or more tasks."""

    tasks: List[TaskUpdateItem] = Field(
        ...,
        description="Tasks to update (1-100). Each requires task_id and project_id.",
        min_length=1,
        max_length=100,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format",
    )


class TaskIdentifier(BaseModel):
    """Task identifier for operations requiring task_id and project_id."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task_id: str = Field(
        ...,
        description="Task identifier",
        pattern=r"^[a-f0-9]{24}$",
    )
    project_id: str = Field(
        ...,
        description="Project ID the task belongs to",
        pattern=r"^(inbox\d+|[a-f0-9]{24})$",
    )


class CompleteTasksInput(BaseMCPInput):
    """Complete one or more tasks."""

    tasks: List[TaskIdentifier] = Field(
        ...,
        description="Tasks to mark as completed (1-100)",
        min_length=1,
        max_length=100,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format",
    )


class DeleteTasksInput(BaseMCPInput):
    """Delete one or more tasks (moves to trash)."""

    tasks: List[TaskIdentifier] = Field(
        ...,
        description="Tasks to delete (1-100)",
        min_length=1,
        max_length=100,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format",
    )


class TaskMoveItem(BaseModel):
    """Single task move specification for batch operations."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task_id: str = Field(
        ...,
        description="Task identifier to move",
        pattern=r"^[a-f0-9]{24}$",
    )
    from_project_id: str = Field(
        ...,
        description="Source project ID",
        pattern=r"^(inbox\d+|[a-f0-9]{24})$",
    )
    to_project_id: str = Field(
        ...,
        description="Destination project ID",
        pattern=r"^(inbox\d+|[a-f0-9]{24})$",
    )


class MoveTasksInput(BaseMCPInput):
    """Move one or more tasks between projects."""

    moves: List[TaskMoveItem] = Field(
        ...,
        description="Move operations (1-100)",
        min_length=1,
        max_length=100,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format",
    )


class TaskParentItem(BaseModel):
    """Single parent assignment specification for batch operations."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task_id: str = Field(
        ...,
        description="Task identifier to make a subtask",
        pattern=r"^[a-f0-9]{24}$",
    )
    project_id: str = Field(
        ...,
        description="Project ID containing both tasks",
        pattern=r"^(inbox\d+|[a-f0-9]{24})$",
    )
    parent_id: str = Field(
        ...,
        description="Parent task identifier",
        pattern=r"^[a-f0-9]{24}$",
    )


class SetTaskParentsInput(BaseMCPInput):
    """Make one or more tasks into subtasks of other tasks."""

    tasks: List[TaskParentItem] = Field(
        ...,
        description="Parent assignments (1-50)",
        min_length=1,
        max_length=50,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format",
    )


class TaskUnparentItem(BaseModel):
    """Single unparent specification for batch operations."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task_id: str = Field(
        ...,
        description="Task identifier to unparent",
        pattern=r"^[a-f0-9]{24}$",
    )
    project_id: str = Field(
        ...,
        description="Project ID containing the task",
        pattern=r"^(inbox\d+|[a-f0-9]{24})$",
    )


class UnparentTasksInput(BaseMCPInput):
    """Remove one or more tasks from their parent (make top-level)."""

    tasks: List[TaskUnparentItem] = Field(
        ...,
        description="Tasks to unparent (1-50)",
        min_length=1,
        max_length=50,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format",
    )


class TaskPinItem(BaseModel):
    """Single pin/unpin specification for batch operations."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task_id: str = Field(
        ...,
        description="Task identifier to pin/unpin",
        pattern=r"^[a-f0-9]{24}$",
    )
    project_id: str = Field(
        ...,
        description="Project ID the task belongs to",
        pattern=r"^(inbox\d+|[a-f0-9]{24})$",
    )
    pin: bool = Field(
        default=True,
        description="True to pin the task, False to unpin it",
    )


class PinTasksInput(BaseMCPInput):
    """Pin or unpin one or more tasks."""

    tasks: List[TaskPinItem] = Field(
        ...,
        description="Pin operations (1-50)",
        min_length=1,
        max_length=50,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format",
    )


class TaskGetInput(BaseMCPInput):
    """Input for getting a task by ID."""

    task_id: str = Field(
        ...,
        description="Task identifier (24-character hex string)",
        pattern=r"^[a-f0-9]{24}$",
    )
    project_id: Optional[str] = Field(
        default=None,
        description="Project ID (required for V1 API fallback)",
        pattern=r"^(inbox\d+|[a-f0-9]{24})$",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format",
    )


class TaskListInput(BaseMCPInput):
    """
    List tasks with flexible filtering.

    This unified tool handles all task listing scenarios:
    - Active tasks (default)
    - Completed tasks (requires from_date/to_date or days)
    - Abandoned tasks (requires from_date/to_date or days)
    - Deleted/trashed tasks
    """

    # Status filter (absorbs completed/abandoned/deleted tools)
    status: Literal["active", "completed", "abandoned", "deleted"] = Field(
        default="active",
        description=(
            "Task status filter:\n"
            "- 'active': Current/pending tasks (default)\n"
            "- 'completed': Completed tasks (use days or from_date/to_date)\n"
            "- 'abandoned': Abandoned/won't-do tasks (use days or from_date/to_date)\n"
            "- 'deleted': Trashed tasks"
        ),
    )
    # Existing filters
    project_id: Optional[str] = Field(
        default=None,
        description="Filter by project ID",
        pattern=r"^(inbox\d+|[a-f0-9]{24})$",
    )
    column_id: Optional[str] = Field(
        default=None,
        description=(
            "Filter by kanban column ID. "
            "Use with project_id for best results. "
            "Only applies to 'active' status."
        ),
        pattern=r"^[a-f0-9]{24}$",
    )
    tag: Optional[str] = Field(
        default=None,
        description="Filter by tag name",
    )
    priority: Optional[str] = Field(
        default=None,
        description="Filter by priority: 'none', 'low', 'medium', 'high'",
        pattern=r"^(none|low|medium|high)$",
    )
    due_today: Optional[bool] = Field(
        default=None,
        description="Filter to only tasks due today (for active status)",
    )
    overdue: Optional[bool] = Field(
        default=None,
        description="Filter to only overdue tasks (for active status)",
    )
    # Date range (for completed/abandoned status)
    from_date: Optional[str] = Field(
        default=None,
        description="Start date for completed/abandoned queries (YYYY-MM-DD format)",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    to_date: Optional[str] = Field(
        default=None,
        description="End date for completed/abandoned queries (YYYY-MM-DD format)",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    days: int = Field(
        default=7,
        description="Number of days to look back for completed/abandoned queries (if from_date/to_date not specified)",
        ge=1,
        le=90,
    )
    # Pagination
    limit: int = Field(
        default=50,
        description="Maximum number of tasks to return",
        ge=1,
        le=500,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format",
    )


class SearchInput(BaseMCPInput):
    """Input for searching tasks."""

    query: str = Field(
        ...,
        description="Search query to match against task titles and content",
        min_length=1,
        max_length=200,
    )
    limit: int = Field(
        default=20,
        description="Maximum number of results to return",
        ge=1,
        le=100,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()


# =============================================================================
# Project Input Models
# =============================================================================


class ProjectCreateInput(BaseMCPInput):
    """Input for creating a project."""

    name: str = Field(
        ...,
        description="Project name (e.g., 'Work', 'Personal', 'Shopping')",
        min_length=1,
        max_length=100,
    )
    color: Optional[str] = Field(
        default=None,
        description="Hex color code (e.g., '#F18181', '#86BB6D')",
        pattern=r"^#[0-9A-Fa-f]{6}$",
    )
    kind: Optional[str] = Field(
        default="TASK",
        description="Project type: 'TASK' for tasks, 'NOTE' for notes",
        pattern=r"^(TASK|NOTE)$",
    )
    view_mode: Optional[str] = Field(
        default="list",
        description="View mode: 'list', 'kanban', 'timeline'",
        pattern=r"^(list|kanban|timeline)$",
    )
    folder_id: Optional[str] = Field(
        default=None,
        description="Parent folder ID to place project in",
        pattern=r"^[a-f0-9]{24}$",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format",
    )


class ProjectGetInput(BaseMCPInput):
    """Input for getting a project."""

    project_id: str = Field(
        ...,
        description="Project identifier",
        pattern=r"^(inbox\d+|[a-f0-9]{24})$",
    )
    include_tasks: bool = Field(
        default=False,
        description="Whether to include tasks in the response",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format",
    )


class ProjectDeleteInput(BaseMCPInput):
    """Input for deleting a project."""

    project_id: str = Field(
        ...,
        description="Project identifier to delete",
        pattern=r"^[a-f0-9]{24}$",
    )


class ProjectUpdateInput(BaseMCPInput):
    """Input for updating a project."""

    project_id: str = Field(
        ...,
        description="Project identifier to update",
        pattern=r"^[a-f0-9]{24}$",
    )
    name: Optional[str] = Field(
        default=None,
        description="New project name",
        min_length=1,
        max_length=100,
    )
    color: Optional[str] = Field(
        default=None,
        description="New hex color code (e.g., '#F18181')",
        pattern=r"^#[0-9A-Fa-f]{6}$",
    )
    folder_id: Optional[str] = Field(
        default=None,
        description="New folder ID (use 'NONE' to remove from folder)",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format",
    )


# =============================================================================
# Folder Input Models
# =============================================================================


class FolderCreateInput(BaseMCPInput):
    """Input for creating a folder (project group)."""

    name: str = Field(
        ...,
        description="Folder name",
        min_length=1,
        max_length=100,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format",
    )


class FolderDeleteInput(BaseMCPInput):
    """Input for deleting a folder."""

    folder_id: str = Field(
        ...,
        description="Folder identifier to delete",
        pattern=r"^[a-f0-9]{24}$",
    )


class FolderRenameInput(BaseMCPInput):
    """Input for renaming a folder (project group)."""

    folder_id: str = Field(
        ...,
        description="Folder identifier to rename",
        pattern=r"^[a-f0-9]{24}$",
    )
    name: str = Field(
        ...,
        description="New folder name",
        min_length=1,
        max_length=100,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format",
    )


# =============================================================================
# Column Input Models (Kanban)
# =============================================================================


class ColumnListInput(BaseMCPInput):
    """Input for listing kanban columns."""

    project_id: str = Field(
        ...,
        description="Project ID to get columns for (must be a kanban-view project)",
        pattern=r"^(inbox\d+|[a-f0-9]{24})$",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format",
    )


class ColumnCreateInput(BaseMCPInput):
    """Input for creating a kanban column."""

    project_id: str = Field(
        ...,
        description="Project ID to create column in",
        pattern=r"^[a-f0-9]{24}$",
    )
    name: str = Field(
        ...,
        description="Column name (e.g., 'To Do', 'In Progress', 'Done')",
        min_length=1,
        max_length=100,
    )
    sort_order: Optional[int] = Field(
        default=None,
        description="Display order (lower numbers appear first)",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format",
    )


class ColumnUpdateInput(BaseMCPInput):
    """Input for updating a kanban column."""

    column_id: str = Field(
        ...,
        description="Column identifier to update",
        pattern=r"^[a-f0-9]{24}$",
    )
    project_id: str = Field(
        ...,
        description="Project ID the column belongs to",
        pattern=r"^[a-f0-9]{24}$",
    )
    name: Optional[str] = Field(
        default=None,
        description="New column name",
        min_length=1,
        max_length=100,
    )
    sort_order: Optional[int] = Field(
        default=None,
        description="New display order",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format",
    )


class ColumnDeleteInput(BaseMCPInput):
    """Input for deleting a kanban column."""

    column_id: str = Field(
        ...,
        description="Column identifier to delete",
        pattern=r"^[a-f0-9]{24}$",
    )
    project_id: str = Field(
        ...,
        description="Project ID the column belongs to",
        pattern=r"^[a-f0-9]{24}$",
    )


# =============================================================================
# Tag Input Models
# =============================================================================


class TagCreateInput(BaseMCPInput):
    """Input for creating a tag."""

    name: str = Field(
        ...,
        description="Tag name/label (e.g., 'work', 'personal', 'urgent')",
        min_length=1,
        max_length=50,
    )
    color: Optional[str] = Field(
        default=None,
        description="Hex color code (e.g., '#F18181')",
        pattern=r"^#[0-9A-Fa-f]{6}$",
    )
    parent: Optional[str] = Field(
        default=None,
        description="Parent tag name for nesting",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format",
    )


class TagDeleteInput(BaseMCPInput):
    """Input for deleting a tag."""

    name: str = Field(
        ...,
        description="Tag name to delete (lowercase identifier)",
        min_length=1,
        max_length=50,
    )


class TagMergeInput(BaseMCPInput):
    """Input for merging tags."""

    source: str = Field(
        ...,
        description="Tag to merge from (will be deleted)",
        min_length=1,
        max_length=50,
    )
    target: str = Field(
        ...,
        description="Tag to merge into (will remain)",
        min_length=1,
        max_length=50,
    )


class TagUpdateInput(BaseMCPInput):
    """
    Update a tag's properties.

    This tool handles color, parent, and rename (label) operations.
    If label is provided, the tag is renamed first, then other updates are applied.
    """

    name: str = Field(
        ...,
        description="Current tag name (lowercase identifier) to update",
        min_length=1,
        max_length=50,
    )
    color: Optional[str] = Field(
        default=None,
        description="New hex color code (e.g., '#F18181')",
        pattern=r"^#[0-9A-Fa-f]{6}$",
    )
    parent: Optional[str] = Field(
        default=None,
        description="New parent tag name (or empty string to remove parent)",
    )
    # NEW: Label/rename (absorbs rename_tag tool)
    label: Optional[str] = Field(
        default=None,
        description="New display label to rename the tag to",
        min_length=1,
        max_length=50,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format",
    )


# =============================================================================
# Focus/Pomodoro Input Models
# =============================================================================


class FocusStatsInput(BaseMCPInput):
    """Input for focus/pomodoro statistics."""

    start_date: Optional[str] = Field(
        default=None,
        description="Start date in YYYY-MM-DD format",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    end_date: Optional[str] = Field(
        default=None,
        description="End date in YYYY-MM-DD format",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    days: int = Field(
        default=30,
        description="Number of days to look back (if dates not specified)",
        ge=1,
        le=365,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format",
    )


# =============================================================================
# Habit Input Models
# =============================================================================


class HabitListInput(BaseMCPInput):
    """Input for listing all habits."""

    include_archived: bool = Field(
        default=False,
        description="Include archived habits in the list",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format",
    )


class HabitGetInput(BaseMCPInput):
    """Input for getting a specific habit."""

    habit_id: str = Field(
        ...,
        description="Habit ID (24-character hex string)",
        pattern=r"^[a-f0-9]{24}$",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format",
    )


class HabitCreateInput(BaseMCPInput):
    """Input for creating a new habit."""

    name: str = Field(
        ...,
        description="Habit name (e.g., 'Exercise', 'Read 30 minutes')",
        min_length=1,
        max_length=100,
    )
    habit_type: str = Field(
        default="Boolean",
        description="Habit type: 'Boolean' for yes/no, 'Real' for numeric counter",
        pattern=r"^(Boolean|Real)$",
    )
    goal: float = Field(
        default=1.0,
        description="Target goal value (1.0 for boolean, custom for numeric)",
        ge=0.1,
        le=10000,
    )
    step: float = Field(
        default=1.0,
        description="Increment step for numeric habits (e.g., 1.0 for +1 button)",
        ge=0.1,
        le=1000,
    )
    unit: str = Field(
        default="Count",
        description="Unit of measurement (e.g., 'Count', 'Minutes', 'Pages')",
        max_length=20,
    )
    color: Optional[str] = Field(
        default=None,
        description="Hex color code (e.g., '#97E38B'). Defaults to green.",
        pattern=r"^#[0-9A-Fa-f]{6}$",
    )
    section_id: Optional[str] = Field(
        default=None,
        description="Time-of-day section ID (_morning, _afternoon, _night). Get from ticktick_habit_sections.",
        pattern=r"^[a-f0-9]{24}$",
    )
    repeat_rule: str = Field(
        default="RRULE:FREQ=WEEKLY;BYDAY=SU,MO,TU,WE,TH,FR,SA",
        description="RRULE recurrence pattern. Daily: 'RRULE:FREQ=WEEKLY;BYDAY=SU,MO,TU,WE,TH,FR,SA'. 5x/week: 'RRULE:FREQ=WEEKLY;TT_TIMES=5'",
    )
    reminders: Optional[List[str]] = Field(
        default=None,
        description="List of reminder times in HH:MM format (e.g., ['09:00', '21:00'])",
        max_length=5,
    )
    target_days: int = Field(
        default=0,
        description="Goal in days (e.g., 100 for '100-day challenge'). 0 = no target.",
        ge=0,
        le=1000,
    )
    encouragement: str = Field(
        default="",
        description="Motivational message to display",
        max_length=200,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format",
    )

    @field_validator("reminders")
    @classmethod
    def validate_reminders(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return v
        import re
        time_pattern = re.compile(r"^([01]?[0-9]|2[0-3]):[0-5][0-9]$")
        for reminder in v:
            if not time_pattern.match(reminder):
                raise ValueError(f"Invalid time format: {reminder}. Use HH:MM format.")
        return v


class HabitUpdateInput(BaseMCPInput):
    """
    Update a habit's properties.

    This tool handles all habit updates including archive/unarchive.
    Set archived=true to archive, archived=false to unarchive.
    """

    habit_id: str = Field(
        ...,
        description="Habit ID to update",
        pattern=r"^[a-f0-9]{24}$",
    )
    name: Optional[str] = Field(
        default=None,
        description="New habit name",
        min_length=1,
        max_length=100,
    )
    goal: Optional[float] = Field(
        default=None,
        description="New target goal value",
        ge=0.1,
        le=10000,
    )
    step: Optional[float] = Field(
        default=None,
        description="New increment step for numeric habits",
        ge=0.1,
        le=1000,
    )
    unit: Optional[str] = Field(
        default=None,
        description="New unit of measurement",
        max_length=20,
    )
    color: Optional[str] = Field(
        default=None,
        description="New hex color code",
        pattern=r"^#[0-9A-Fa-f]{6}$",
    )
    section_id: Optional[str] = Field(
        default=None,
        description="New time-of-day section ID",
        pattern=r"^[a-f0-9]{24}$",
    )
    repeat_rule: Optional[str] = Field(
        default=None,
        description="New RRULE recurrence pattern",
    )
    reminders: Optional[List[str]] = Field(
        default=None,
        description="New list of reminder times in HH:MM format",
        max_length=5,
    )
    target_days: Optional[int] = Field(
        default=None,
        description="New goal in days",
        ge=0,
        le=1000,
    )
    encouragement: Optional[str] = Field(
        default=None,
        description="New motivational message",
        max_length=200,
    )
    # NEW: Archive control (absorbs archive_habit/unarchive_habit)
    archived: Optional[bool] = Field(
        default=None,
        description="Set to true to archive the habit, false to unarchive it",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format",
    )


class HabitDeleteInput(BaseMCPInput):
    """Input for deleting a habit."""

    habit_id: str = Field(
        ...,
        description="Habit ID to delete",
        pattern=r"^[a-f0-9]{24}$",
    )


class HabitCheckinItem(BaseModel):
    """Single habit check-in specification for batch operations."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    habit_id: str = Field(
        ...,
        description="Habit ID to check in",
        pattern=r"^[a-f0-9]{24}$",
    )
    value: float = Field(
        default=1.0,
        description="Check-in value. 1.0 for boolean habits, custom for numeric.",
        ge=0.1,
        le=10000,
    )
    checkin_date: Optional[str] = Field(
        default=None,
        description=(
            "Date to check in for (YYYY-MM-DD format). "
            "If not provided, checks in for today. "
            "Use a past date to backdate the check-in."
        ),
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )


class CheckinHabitsInput(BaseMCPInput):
    """Record one or more habit check-ins."""

    checkins: List[HabitCheckinItem] = Field(
        ...,
        description="Check-ins to record (1-100). Useful for backdating multiple days.",
        min_length=1,
        max_length=100,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format",
    )


class HabitCheckinsInput(BaseMCPInput):
    """Input for querying habit check-in history."""

    habit_ids: List[str] = Field(
        ...,
        description="List of habit IDs to query. Get habit IDs from ticktick_habits.",
        min_length=1,
    )
    after_stamp: int = Field(
        default=0,
        description="Date stamp (YYYYMMDD format, e.g., 20251201) to get check-ins after. 0 for all.",
        ge=0,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.JSON,
        description="Output format (JSON recommended for habit data)",
    )


# =============================================================================
# Backward Compatibility Aliases (Deprecated - will be removed in v0.5.0)
# =============================================================================

# These maintain compatibility with existing code that uses old names
TaskCreateInput = CreateTasksInput  # Alias for migration
TaskUpdateInput = UpdateTasksInput  # Alias for migration
TaskCompleteInput = CompleteTasksInput  # Alias for migration
TaskDeleteInput = DeleteTasksInput  # Alias for migration
TaskMoveInput = MoveTasksInput  # Alias for migration
TaskParentInput = SetTaskParentsInput  # Alias for migration
TaskUnparentInput = UnparentTasksInput  # Alias for migration
TaskPinInput = PinTasksInput  # Alias for migration
HabitCheckinInput = CheckinHabitsInput  # Alias for migration
