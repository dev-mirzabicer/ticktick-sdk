# TickTick SDK - Data Models Documentation

> **Version**: 0.2.1
> **Last Updated**: January 2026
> **Audience**: Developers, AI Agents, System Integrators

This document provides comprehensive documentation for all data models in the TickTick SDK. It is designed to be a complete reference that enables developers and AI agents to work with the SDK's models without needing to read source code.

---

## Table of Contents

1. [Introduction to the Unified Model System](#section-1-introduction-to-the-unified-model-system)
2. [Base Model (TickTickModel)](#section-2-base-model-ticktickmodel)
3. [Task Model](#section-3-task-model)
4. [Project Models](#section-4-project-models)
5. [Tag Model](#section-5-tag-model)
6. [Habit Models](#section-6-habit-models)
7. [User Models](#section-7-user-models)
8. [Model Relationships Diagram](#section-8-model-relationships-diagram)
9. [Field Type Reference](#section-9-field-type-reference)
10. [V1 vs V2 Model Differences](#section-10-v1-vs-v2-model-differences)
11. [Model Validation Rules](#section-11-model-validation-rules)
12. [Working with Models (Practical Guide)](#section-12-working-with-models-practical-guide)

---

## Section 1: Introduction to the Unified Model System

### What is the Unified Model Strategy?

The TickTick SDK implements a **unified model strategy** where a single set of Pydantic models serves as the canonical representation for all data, regardless of which API version (V1 or V2) the data originates from.

This strategy solves a fundamental challenge: TickTick exposes two distinct APIs with different:
- **Field naming conventions** (V1 uses snake_case in responses, V2 uses camelCase)
- **Date/time formats** (V1: `%Y-%m-%dT%H:%M:%S%z`, V2: `%Y-%m-%dT%H:%M:%S.000+0000`)
- **Feature availability** (V2 has tags, habits, subtasks; V1 does not)
- **Response structures** (V2 uses batch operations with `id2etag`/`id2error` patterns)

### Why Unified Models Are Necessary

| Challenge | Without Unified Models | With Unified Models |
|-----------|----------------------|---------------------|
| **Field Names** | Two classes per entity (TaskV1, TaskV2) | One `Task` class with aliases |
| **Date Parsing** | Duplicate datetime logic | Centralized `parse_datetime()` |
| **Feature Gaps** | Conditional logic everywhere | Optional fields with None defaults |
| **API Changes** | Breaks user code | SDK absorbs changes |
| **User Experience** | Users must know API versions | Users work with clean models |

### How Unified Models Abstract API Differences

The SDK uses several techniques to provide a seamless experience:

1. **Field Aliases**: Pydantic's `alias` parameter maps camelCase API fields to snake_case Python attributes
2. **Datetime Validators**: Custom validators automatically parse multiple datetime formats
3. **Optional Fields**: V2-only fields are optional with None defaults for V1 compatibility
4. **Conversion Methods**: `to_v1_dict()`, `to_v2_dict()`, `from_v1()`, `from_v2()` handle format differences
5. **Populate by Name**: The `populate_by_name=True` config accepts both field name and alias

### Overview of All Models

The SDK provides **14 unified models** across 6 source files:

| Model | Purpose | File | V1 Support | V2 Support |
|-------|---------|------|------------|------------|
| **Task** | Task/todo item | `models/task.py` | Yes | Yes |
| **ChecklistItem** | Subtask within a task | `models/task.py` | Yes | Yes |
| **TaskReminder** | Task reminder configuration | `models/task.py` | Yes | Yes |
| **Project** | Project/list container | `models/project.py` | Yes | Yes |
| **ProjectGroup** | Folder for organizing projects | `models/project.py` | No | Yes |
| **Column** | Kanban board column | `models/project.py` | Yes | Yes |
| **ProjectData** | Project with tasks and columns | `models/project.py` | Yes | Yes |
| **SortOption** | Sorting configuration | `models/project.py` | Yes | Yes |
| **Tag** | Task tag/label | `models/tag.py` | No | Yes |
| **Habit** | Recurring habit tracker | `models/habit.py` | No | Yes |
| **HabitSection** | Time-of-day habit grouping | `models/habit.py` | No | Yes |
| **HabitCheckin** | Habit completion record | `models/habit.py` | No | Yes |
| **HabitPreferences** | Habit feature settings | `models/habit.py` | No | Yes |
| **User** | User profile information | `models/user.py` | No | Yes |
| **UserStatus** | Subscription and account status | `models/user.py` | No | Yes |
| **UserStatistics** | Productivity statistics | `models/user.py` | No | Yes |
| **TaskCount** | Task completion counts | `models/user.py` | No | Yes |

---

## Section 2: Base Model (TickTickModel)

**Source File**: `/src/ticktick_sdk/models/base.py`

### Overview

`TickTickModel` is the abstract base class that all unified models inherit from. It provides common configuration, datetime handling, and API conversion methods.

### Class Definition

```python
class TickTickModel(BaseModel):
    """Base model for all TickTick data models."""

    model_config = ConfigDict(
        populate_by_name=True,      # Accept field name OR alias
        use_enum_values=True,       # Serialize enums as values
        validate_assignment=True,   # Validate on attribute assignment
        extra="ignore",             # Allow unknown fields from API
        alias_generator=lambda s: s,  # No automatic alias generation
    )
```

### Pydantic v2 Configuration Settings

| Setting | Value | Purpose |
|---------|-------|---------|
| `populate_by_name` | `True` | Allows model instantiation using either the Python field name (`project_id`) or the JSON alias (`projectId`) |
| `use_enum_values` | `True` | When serializing, outputs the enum's value (e.g., `2`) rather than the enum instance |
| `validate_assignment` | `True` | Re-validates the model whenever an attribute is modified after creation |
| `extra` | `"ignore"` | Silently ignores extra fields from API responses that aren't defined in the model (V1/V2 APIs may return different fields) |
| `alias_generator` | Identity function | Disables automatic alias generation; aliases are manually defined per-field |

### Common Methods

#### `parse_datetime(value)` - Parse Datetime Strings

Parses datetime strings from either V1 or V2 API formats into Python `datetime` objects.

**Signature**:
```python
@classmethod
def parse_datetime(cls, value: str | datetime | None) -> datetime | None
```

**Supported Formats** (tried in order):
1. `%Y-%m-%dT%H:%M:%S.%f%z` - ISO 8601 with microseconds and timezone
2. `%Y-%m-%dT%H:%M:%S.000+0000` - V2 format with milliseconds
3. `%Y-%m-%dT%H:%M:%S%z` - ISO 8601 without microseconds
4. `%Y-%m-%dT%H:%M:%S+0000` - V2 format without milliseconds
5. `%Y-%m-%dT%H:%M:%SZ` - ISO 8601 with Z suffix
6. ISO format via `datetime.fromisoformat()` - Fallback

**Behavior**:
- Returns `None` if input is `None`
- Returns input unchanged if already a `datetime` object
- Handles the `+0000` timezone format by converting to `+00:00`
- Returns `None` if parsing fails (does not raise exceptions)

**Example**:
```python
# V2 format
dt = TickTickModel.parse_datetime("2024-12-15T14:30:00.000+0000")
# Returns: datetime(2024, 12, 15, 14, 30, 0, tzinfo=UTC)

# V1 format
dt = TickTickModel.parse_datetime("2024-12-15T14:30:00+00:00")
# Returns: datetime(2024, 12, 15, 14, 30, 0, tzinfo=UTC)

# Already a datetime
dt = TickTickModel.parse_datetime(existing_datetime)
# Returns: existing_datetime unchanged
```

#### `format_datetime(value, for_api)` - Format Datetime for API

Formats a Python `datetime` object for API submission.

**Signature**:
```python
@classmethod
def format_datetime(cls, value: datetime | None, for_api: str = "v2") -> str | None
```

**Parameters**:
- `value`: The datetime to format (or None)
- `for_api`: Target API version - `"v1"` or `"v2"` (default: `"v2"`)

**Output Formats**:
- V1: `%Y-%m-%dT%H:%M:%S%z` (e.g., `"2024-12-15T14:30:00+00:00"`)
- V2: `%Y-%m-%dT%H:%M:%S.000+0000` (e.g., `"2024-12-15T14:30:00.000+0000"`)

**Behavior**:
- Returns `None` if input is `None`
- Adds UTC timezone if the datetime is naive (no timezone info)

**Example**:
```python
from datetime import datetime, timezone

dt = datetime(2024, 12, 15, 14, 30, 0, tzinfo=timezone.utc)

# For V2 API
v2_str = Task.format_datetime(dt, "v2")
# Returns: "2024-12-15T14:30:00.000+0000"

# For V1 API
v1_str = Task.format_datetime(dt, "v1")
# Returns: "2024-12-15T14:30:00+00:00"
```

#### `to_v1_dict()` - Convert to V1 API Format

Converts the model to a dictionary suitable for V1 API requests.

**Signature**:
```python
def to_v1_dict(self) -> dict[str, Any]
```

**Behavior**:
- Uses Pydantic's `model_dump()` with `by_alias=True` to output camelCase keys
- Excludes `None` values to avoid sending null fields
- Subclasses override this method to handle V1-specific field transformations

#### `to_v2_dict()` - Convert to V2 API Format

Converts the model to a dictionary suitable for V2 API requests.

**Signature**:
```python
def to_v2_dict(self) -> dict[str, Any]
```

**Behavior**:
- Uses Pydantic's `model_dump()` with `by_alias=True`
- Excludes `None` values
- Subclasses override this method to handle V2-specific transformations

#### `from_v1(data)` - Create from V1 Response

Class method to create a model instance from V1 API response data.

**Signature**:
```python
@classmethod
def from_v1(cls, data: dict[str, Any]) -> Self
```

**Behavior**:
- Uses Pydantic's `model_validate()` for parsing
- Field aliases allow camelCase keys to map to snake_case attributes
- Datetime validators automatically parse string dates

#### `from_v2(data)` - Create from V2 Response

Class method to create a model instance from V2 API response data.

**Signature**:
```python
@classmethod
def from_v2(cls, data: dict[str, Any]) -> Self
```

**Behavior**:
- Identical to `from_v1()` in the base class
- Subclasses may override for V2-specific parsing

---

## Section 3: Task Model

**Source File**: `/src/ticktick_sdk/models/task.py` (339 lines)

The `Task` model is the most complex and important model in the SDK. It represents a task/todo item and combines fields from both V1 and V2 APIs.

### Task Class

```python
class Task(TickTickModel):
    """
    Unified Task model.

    V1-only fields: (none - V2 is superset)
    V2-only fields: tags, parent_id, child_ids, etag, progress, deleted,
                    is_floating, creator, assignee, focus_summaries,
                    pomodoro_summaries, attachments, column_id, comment_count
    """
```

### Task Fields - Complete Reference

#### Core Identifiers

| Field | Type | Default | Alias | Description | V1 | V2 |
|-------|------|---------|-------|-------------|----|----|
| `id` | `str` | Required | - | Unique task identifier (24-char hex, MongoDB ObjectId format) | Yes | Yes |
| `project_id` | `str` | Required | `projectId` | ID of the project containing this task | Yes | Yes |
| `etag` | `str \| None` | `None` | - | Version tag for optimistic concurrency control (8-char alphanumeric) | No | Yes |

#### Content Fields

| Field | Type | Default | Alias | Description | V1 | V2 |
|-------|------|---------|-------|-------------|----|----|
| `title` | `str \| None` | `None` | - | Task title/name (main text displayed) | Yes | Yes |
| `content` | `str \| None` | `None` | - | Task notes/description (rich text, may contain markdown) | Yes | Yes |
| `desc` | `str \| None` | `None` | - | Checklist description (used when kind=CHECKLIST) | Yes | Yes |
| `kind` | `str` | `"TEXT"` | - | Task type: `"TEXT"`, `"NOTE"`, or `"CHECKLIST"` | Yes | Yes |

#### Status Fields

| Field | Type | Default | Alias | Description | V1 | V2 |
|-------|------|---------|-------|-------------|----|----|
| `status` | `int` | `0` | - | Task status: `-1`=Abandoned, `0`=Active, `1`=Completed(alt), `2`=Completed | Yes | Yes |
| `priority` | `int` | `0` | - | Priority level: `0`=None, `1`=Low, `3`=Medium, `5`=High | Yes | Yes |
| `progress` | `int \| None` | `None` | - | Completion percentage (0-100) for checklist tasks | No | Yes |
| `deleted` | `int` | `0` | - | Soft delete flag: `0`=Active, `1`=Deleted (in trash) | No | Yes |

#### Date/Time Fields

| Field | Type | Default | Alias | Description | V1 | V2 |
|-------|------|---------|-------|-------------|----|----|
| `start_date` | `datetime \| None` | `None` | `startDate` | Task start date/time | Yes | Yes |
| `due_date` | `datetime \| None` | `None` | `dueDate` | Task due date/time | Yes | Yes |
| `created_time` | `datetime \| None` | `None` | `createdTime` | When the task was created | Yes | Yes |
| `modified_time` | `datetime \| None` | `None` | `modifiedTime` | When the task was last modified | Yes | Yes |
| `completed_time` | `datetime \| None` | `None` | `completedTime` | When the task was completed | Yes | Yes |
| `time_zone` | `str \| None` | `None` | `timeZone` | IANA timezone name (e.g., "America/New_York") | Yes | Yes |
| `is_all_day` | `bool \| None` | `None` | `isAllDay` | True if task has no specific time (date only) | Yes | Yes |
| `is_floating` | `bool` | `False` | `isFloating` | True if task has no timezone (floating time) | No | Yes |

#### Recurrence Fields

| Field | Type | Default | Alias | Description | V1 | V2 |
|-------|------|---------|-------|-------------|----|----|
| `repeat_flag` | `str \| None` | `None` | `repeatFlag` | iCalendar RRULE format (e.g., "RRULE:FREQ=DAILY;INTERVAL=1") | Yes | Yes |
| `repeat_from` | `int \| None` | `None` | `repeatFrom` | Recurrence base: `0`=Due date, `1`=Completed date, `2`=Unknown | No | Yes |
| `repeat_first_date` | `datetime \| None` | `None` | `repeatFirstDate` | Original first occurrence date | No | Yes |
| `repeat_task_id` | `str \| None` | `None` | `repeatTaskId` | ID of the recurring task template | No | Yes |
| `ex_date` | `list[str] \| None` | `None` | `exDate` | List of excluded dates (skipped occurrences) | No | Yes |

#### Reminder Fields

| Field | Type | Default | Alias | Description | V1 | V2 |
|-------|------|---------|-------|-------------|----|----|
| `reminder` | `str \| None` | `None` | - | Legacy single reminder field | No | Yes |
| `reminders` | `list[TaskReminder]` | `[]` | - | List of reminder configurations | Yes | Yes |
| `remind_time` | `datetime \| None` | `None` | `remindTime` | Computed reminder time | No | Yes |

#### Hierarchy Fields (V2 Only)

| Field | Type | Default | Alias | Description | V1 | V2 |
|-------|------|---------|-------|-------------|----|----|
| `parent_id` | `str \| None` | `None` | `parentId` | Parent task ID (makes this a subtask) | No | Yes |
| `child_ids` | `list[str] \| None` | `None` | `childIds` | List of child task IDs (subtasks) | No | Yes |

#### Checklist Items

| Field | Type | Default | Alias | Description | V1 | V2 |
|-------|------|---------|-------|-------------|----|----|
| `items` | `list[ChecklistItem]` | `[]` | - | Checklist/subtask items within the task | Yes | Yes |

#### Organization Fields

| Field | Type | Default | Alias | Description | V1 | V2 |
|-------|------|---------|-------|-------------|----|----|
| `tags` | `list[str]` | `[]` | - | List of tag names assigned to the task | No | Yes |
| `column_id` | `str \| None` | `None` | `columnId` | Kanban column ID (if project uses kanban view) | No | Yes |
| `sort_order` | `int \| None` | `None` | `sortOrder` | Manual sort position within project | Yes | Yes |

#### Collaboration Fields (V2 Only)

| Field | Type | Default | Alias | Description | V1 | V2 |
|-------|------|---------|-------|-------------|----|----|
| `assignee` | `Any \| None` | `None` | - | Assigned user (structure varies) | No | Yes |
| `creator` | `int \| None` | `None` | - | User ID of task creator | No | Yes |
| `completed_user_id` | `int \| None` | `None` | `completedUserId` | User ID who completed the task | No | Yes |
| `comment_count` | `int \| None` | `None` | `commentCount` | Number of comments on the task | No | Yes |

#### Attachment Fields (V2 Only)

| Field | Type | Default | Alias | Description | V1 | V2 |
|-------|------|---------|-------|-------------|----|----|
| `attachments` | `list[Any]` | `[]` | - | File attachments (structure varies) | No | Yes |

#### Focus/Pomodoro Fields (V2 Only)

| Field | Type | Default | Alias | Description | V1 | V2 |
|-------|------|---------|-------|-------------|----|----|
| `focus_summaries` | `list[Any]` | `[]` | `focusSummaries` | Focus session summaries for this task | No | Yes |
| `pomodoro_summaries` | `list[Any]` | `[]` | `pomodoroSummaries` | Pomodoro session summaries | No | Yes |

### Task Status Values

The `TaskStatus` enum (from `constants.py`) defines valid status values:

| Constant | Value | Description |
|----------|-------|-------------|
| `TaskStatus.ABANDONED` | `-1` | "Won't do" - task intentionally not completed (V2 only) |
| `TaskStatus.ACTIVE` | `0` | Open/in-progress task |
| `TaskStatus.COMPLETED_ALT` | `1` | Completed (alternative value, V2 only) |
| `TaskStatus.COMPLETED` | `2` | Completed (standard value) |

**Important**: Both `1` and `2` indicate completion. The SDK's `TaskStatus.is_completed()` checks for both.

### Task Priority Values

The `TaskPriority` enum defines valid priority levels:

| Constant | Value | Description | UI Display |
|----------|-------|-------------|------------|
| `TaskPriority.NONE` | `0` | No priority set | No indicator |
| `TaskPriority.LOW` | `1` | Low priority | Blue indicator |
| `TaskPriority.MEDIUM` | `3` | Medium priority | Yellow indicator |
| `TaskPriority.HIGH` | `5` | High priority | Red indicator |

**Note**: Values `2` and `4` are NOT valid priorities. The SDK's `TaskPriority.from_string()` maps string names to these specific values.

### Task Kind Values

The `TaskKind` enum defines task types:

| Constant | Value | Description |
|----------|-------|-------------|
| `TaskKind.TEXT` | `"TEXT"` | Standard task |
| `TaskKind.NOTE` | `"NOTE"` | Note/memo (no completion status) |
| `TaskKind.CHECKLIST` | `"CHECKLIST"` | Task with checklist items |

### Task Computed Properties

The `Task` class provides these read-only properties:

| Property | Type | Description |
|----------|------|-------------|
| `is_completed` | `bool` | True if status is COMPLETED or COMPLETED_ALT |
| `is_closed` | `bool` | True if completed OR abandoned |
| `is_abandoned` | `bool` | True if status is ABANDONED |
| `is_active` | `bool` | True if status is ACTIVE |
| `is_subtask` | `bool` | True if parent_id is set |
| `has_subtasks` | `bool` | True if child_ids is non-empty |
| `priority_label` | `str` | Human-readable priority ("none", "low", "medium", "high") |

### Task Field Validators

The Task model includes these validators:

#### Datetime Field Validator
```python
@field_validator(
    "start_date", "due_date", "created_time", "modified_time",
    "completed_time", "remind_time", "repeat_first_date",
    mode="before",
)
@classmethod
def parse_datetime_field(cls, v: Any) -> datetime | None:
    return cls.parse_datetime(v)
```
Automatically parses datetime strings from API responses.

#### Reminders Validator
```python
@field_validator("reminders", mode="before")
@classmethod
def parse_reminders(cls, v: Any) -> list[TaskReminder]:
```
Handles multiple reminder formats:
- V1 format: List of strings (`["TRIGGER:-PT30M"]`)
- V2 format: List of dicts (`[{"id": "...", "trigger": "TRIGGER:-PT30M"}]`)

#### Items Validator
```python
@field_validator("items", mode="before")
@classmethod
def parse_items(cls, v: Any) -> list[ChecklistItem]:
```
Parses checklist items from dict or ChecklistItem instances.

### Task Conversion Methods

#### `to_v1_dict()` - V1 API Format

Returns a dictionary for V1 API task creation/update:

```python
{
    "id": "...",
    "projectId": "...",
    "title": "...",
    "content": "...",
    "desc": "...",
    "isAllDay": True,
    "startDate": "2024-12-15T14:30:00+00:00",  # V1 format
    "dueDate": "2024-12-16T14:30:00+00:00",
    "timeZone": "America/New_York",
    "reminders": ["TRIGGER:-PT30M"],  # Strings only
    "repeatFlag": "RRULE:FREQ=DAILY",
    "priority": 5,
    "sortOrder": 123,
    "items": [{"id": "...", "title": "...", "status": 0}]
}
```

**V1 Limitations**:
- No `tags` field
- No `parentId` field
- Reminders are strings, not objects

#### `to_v2_dict(for_update=False)` - V2 API Format

Returns a dictionary for V2 API batch operations:

```python
{
    "id": "...",
    "projectId": "...",
    "title": "...",
    "content": "...",
    "desc": "...",
    "kind": "TEXT",
    "status": 0,
    "priority": 5,
    "isAllDay": True,
    "startDate": "2024-12-15T14:30:00.000+0000",  # V2 format
    "dueDate": "2024-12-16T14:30:00.000+0000",
    "timeZone": "America/New_York",
    "reminders": [{"id": "...", "trigger": "TRIGGER:-PT30M"}],
    "repeatFlag": "RRULE:FREQ=DAILY",
    "tags": ["work", "urgent"],
    "sortOrder": 123,
    "items": [...],
    "parentId": "...",  # Only if set
    "completedTime": "..."  # Only if set
}
```

**The `for_update` parameter**:
- When `for_update=False` (default): Omits None date fields (for task creation)
- When `for_update=True`: Sends empty strings for None date fields (to clear dates)
- Same logic applies to `tags`: empty list sent on update to clear tags

### Task API Quirks and Important Notes

1. **Recurrence requires start_date**: TickTick silently ignores `repeat_flag` if `start_date` is not set. The SDK validates this.

2. **parent_id is ignored on create**: Setting `parent_id` during task creation does NOT create a subtask. You must call `set_task_parent()` separately after creation.

3. **Date clearing requires both dates**: To clear dates, you must clear both `start_date` AND `due_date` together.

4. **Subtasks via V2 only**: The parent/child task hierarchy is only available through the V2 API.

5. **etag for concurrency**: The `etag` field should be preserved and sent back on updates to prevent conflicts.

---

### ChecklistItem Class

**Source File**: `/src/ticktick_sdk/models/task.py`

Represents a subtask/checklist item within a task (when `kind="CHECKLIST"`).

```python
class ChecklistItem(TickTickModel):
    """Subtask/checklist item model."""
```

#### ChecklistItem Fields

| Field | Type | Default | Alias | Description | V1 | V2 |
|-------|------|---------|-------|-------------|----|----|
| `id` | `str` | Required | - | Unique item identifier | Yes | Yes |
| `title` | `str \| None` | `None` | - | Item text/description | Yes | Yes |
| `status` | `int` | `0` | - | Completion status: `0`=Normal, `1`=Completed | Yes | Yes |
| `completed_time` | `datetime \| None` | `None` | `completedTime` | When item was completed | Yes | Yes |
| `start_date` | `datetime \| None` | `None` | `startDate` | Item start date | Yes | Yes |
| `time_zone` | `str \| None` | `None` | `timeZone` | IANA timezone | No | Yes |
| `is_all_day` | `bool \| None` | `None` | `isAllDay` | Date-only flag | Yes | Yes |
| `sort_order` | `int \| None` | `None` | `sortOrder` | Position within checklist | Yes | Yes |

#### SubtaskStatus Values

| Constant | Value | Description |
|----------|-------|-------------|
| `SubtaskStatus.NORMAL` | `0` | Incomplete |
| `SubtaskStatus.COMPLETED` | `1` | Completed |

**Note**: Subtask status uses `1` for completed, unlike Task status which uses `2`.

#### ChecklistItem Properties

| Property | Type | Description |
|----------|------|-------------|
| `is_completed` | `bool` | True if status equals `SubtaskStatus.COMPLETED` (1) |

---

### TaskReminder Class

**Source File**: `/src/ticktick_sdk/models/task.py`

Represents a reminder/alarm configuration for a task.

```python
class TaskReminder(TickTickModel):
    """Task reminder configuration."""
```

#### TaskReminder Fields

| Field | Type | Default | Alias | Description |
|-------|------|---------|-------|-------------|
| `id` | `str \| None` | `None` | - | Reminder identifier (V2 only) |
| `trigger` | `str` | Required | - | iCalendar TRIGGER format |

#### Trigger Format

The `trigger` field uses RFC 5545 TRIGGER format:

```
TRIGGER:-PT{duration}{unit}
```

**Examples**:
- `"TRIGGER:-PT30M"` - 30 minutes before
- `"TRIGGER:-PT1H"` - 1 hour before
- `"TRIGGER:-PT1D"` - 1 day before
- `"TRIGGER:PT0S"` - At the exact time

**Duration Units**:
- `S` - Seconds
- `M` - Minutes
- `H` - Hours
- `D` - Days
- `W` - Weeks

#### TaskReminder Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `from_v1_string` | `classmethod(trigger: str) -> TaskReminder` | Create from V1 string |
| `to_v1_string` | `() -> str` | Convert to V1 format (returns trigger) |

---

## Section 4: Project Models

**Source File**: `/src/ticktick_sdk/models/project.py` (273 lines)

### Project Class

Represents a TickTick project/list that contains tasks.

```python
class Project(TickTickModel):
    """
    Unified Project model.

    V1 fields: id, name, color, sortOrder, closed, groupId, viewMode, permission, kind
    V2 additional: etag, inAll, sortOption, sortType, modifiedTime, isOwner,
                   userCount, muted, transferred, teamId, notificationOptions,
                   openToTeam, teamMemberPermission, background, etc.
    """
```

#### Project Fields - Complete Reference

| Field | Type | Default | Alias | Description | V1 | V2 |
|-------|------|---------|-------|-------------|----|----|
| `id` | `str` | Required | - | Unique project identifier | Yes | Yes |
| `etag` | `str \| None` | `None` | - | Version tag for concurrency | No | Yes |
| `name` | `str` | Required | - | Project name/title | Yes | Yes |
| `color` | `str \| None` | `None` | - | Hex color (e.g., "#F18181") | Yes | Yes |
| `kind` | `str \| None` | `"TASK"` | - | Project type: `"TASK"` or `"NOTE"` | Yes | Yes |
| `group_id` | `str \| None` | `None` | `groupId` | Parent folder ID | Yes | Yes |
| `in_all` | `bool \| None` | `None` | `inAll` | Show in "All" view | No | Yes |
| `view_mode` | `str \| None` | `"list"` | `viewMode` | View: `"list"`, `"kanban"`, `"timeline"` | Yes | Yes |
| `sort_option` | `SortOption \| None` | `None` | `sortOption` | Sorting configuration | No | Yes |
| `sort_order` | `int \| None` | `None` | `sortOrder` | Display position | Yes | Yes |
| `sort_type` | `str \| None` | `None` | `sortType` | Sort type identifier | No | Yes |
| `modified_time` | `datetime \| None` | `None` | `modifiedTime` | Last modification time | No | Yes |
| `is_owner` | `bool \| None` | `None` | `isOwner` | True if current user owns project | No | Yes |
| `user_count` | `int \| None` | `None` | `userCount` | Number of users with access | No | Yes |
| `closed` | `bool \| None` | `None` | - | True if project is archived | Yes | Yes |
| `muted` | `bool \| None` | `None` | - | True if notifications muted | No | Yes |
| `permission` | `str \| None` | `None` | - | User's permission: `"read"`, `"write"`, `"comment"` | Yes | Yes |
| `team_id` | `Any \| None` | `None` | `teamId` | Team ID if shared with team | No | Yes |
| `open_to_team` | `bool \| None` | `None` | `openToTeam` | True if open to team members | No | Yes |

#### ProjectKind Values

| Constant | Value | Description |
|----------|-------|-------------|
| `ProjectKind.TASK` | `"TASK"` | Standard task project |
| `ProjectKind.NOTE` | `"NOTE"` | Notes project |

#### ViewMode Values

| Constant | Value | Description |
|----------|-------|-------------|
| `ViewMode.LIST` | `"list"` | List view (default) |
| `ViewMode.KANBAN` | `"kanban"` | Kanban board view |
| `ViewMode.TIMELINE` | `"timeline"` | Timeline/calendar view |

#### Permission Values

| Constant | Value | Description |
|----------|-------|-------------|
| `Permission.READ` | `"read"` | Read-only access |
| `Permission.WRITE` | `"write"` | Read and write access |
| `Permission.COMMENT` | `"comment"` | Read and comment access |

#### Project Properties

| Property | Type | Description |
|----------|------|-------------|
| `is_inbox` | `bool` | True if ID starts with "inbox" |
| `is_closed` | `bool` | True if `closed` is True |
| `is_note_project` | `bool` | True if kind is NOTE |
| `is_task_project` | `bool` | True if kind is TASK |
| `view_mode_enum` | `ViewMode` | View mode as enum (defaults to LIST) |

#### Project Conversion Methods

**`to_v1_dict()`** - For V1 API:
```python
{
    "name": "Project Name",
    "color": "#F18181",
    "sortOrder": 123,
    "viewMode": "list",
    "kind": "TASK"
}
```

**`to_v2_create_dict()`** - For V2 batch create:
```python
{
    "name": "Project Name",
    "color": "#F18181",
    "kind": "TASK",
    "viewMode": "list",
    "groupId": "folder_id",
    "sortOrder": 123
}
```

**`to_v2_update_dict()`** - For V2 batch update:
```python
{
    "id": "project_id",
    "name": "Project Name",
    "color": "#F18181",
    "groupId": "folder_id"  # Use "NONE" to remove from folder
}
```

---

### ProjectGroup Class (Folder)

Represents a folder for organizing projects. This is a **V2-only** feature.

```python
class ProjectGroup(TickTickModel):
    """
    Project group/folder model.
    This is a V2-only feature for organizing projects.
    """
```

#### ProjectGroup Fields

| Field | Type | Default | Alias | Description |
|-------|------|---------|-------|-------------|
| `id` | `str` | Required | - | Unique folder identifier |
| `etag` | `str \| None` | `None` | - | Version tag |
| `name` | `str` | Required | - | Folder name |
| `view_mode` | `str \| None` | `None` | `viewMode` | Default view mode |
| `sort_option` | `SortOption \| None` | `None` | `sortOption` | Sorting config |
| `sort_order` | `int \| None` | `None` | `sortOrder` | Display position |
| `sort_type` | `str \| None` | `None` | `sortType` | Sort type |
| `deleted` | `int` | `0` | - | Soft delete: `0`=Active, `1`=Deleted |
| `show_all` | `bool` | `False` | `showAll` | Show all projects flag |
| `team_id` | `Any \| None` | `None` | `teamId` | Team ID |
| `user_id` | `int \| None` | `None` | `userId` | Owner user ID |

#### ProjectGroup Conversion Methods

**`to_v2_create_dict()`**:
```python
{
    "name": "Folder Name",
    "listType": "group"  # Required for folders
}
```

**`to_v2_update_dict()`**:
```python
{
    "id": "folder_id",
    "name": "New Name",
    "listType": "group"
}
```

---

### Column Class (Kanban)

Represents a column in a Kanban board view.

```python
class Column(TickTickModel):
    """Kanban column model."""
```

#### Column Fields

| Field | Type | Default | Alias | Description | V1 | V2 |
|-------|------|---------|-------|-------------|----|----|
| `id` | `str` | Required | - | Unique column identifier | Yes | Yes |
| `project_id` | `str` | Required | `projectId` | Parent project ID | Yes | Yes |
| `name` | `str` | Required | - | Column name | Yes | Yes |
| `sort_order` | `int \| None` | `None` | `sortOrder` | Display position | Yes | Yes |

---

### ProjectData Class

A container model that holds a project with its tasks and columns. Primarily returned by the V1 `get_project_with_data` endpoint.

```python
class ProjectData(TickTickModel):
    """Project with its tasks and columns."""
```

#### ProjectData Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `project` | `Project` | Required | The project object |
| `tasks` | `list[Task]` | `[]` | Tasks belonging to this project |
| `columns` | `list[Column]` | `[]` | Kanban columns (if applicable) |

#### ProjectData Factory Methods

**`from_v1(data)`** - Create from V1 API response:
```python
ProjectData.from_v1({
    "project": {...},
    "tasks": [...],
    "columns": [...]
})
```

**`from_v2(project, tasks)`** - Create from V2 data:
```python
ProjectData.from_v2(project=project_obj, tasks=task_list)
# Note: V2 doesn't provide column data, so columns will be empty
```

---

### SortOption Class

Configuration for sorting tasks within a project or tag view.

```python
class SortOption(TickTickModel):
    """Sorting configuration."""
```

#### SortOption Fields

| Field | Type | Default | Alias | Description |
|-------|------|---------|-------|-------------|
| `group_by` | `str \| None` | `None` | `groupBy` | Field to group by |
| `order_by` | `str \| None` | `None` | `orderBy` | Field to order by |

---

## Section 5: Tag Model

**Source File**: `/src/ticktick_sdk/models/tag.py` (107 lines)

Tags are a **V2-only** feature for categorizing and organizing tasks. Tags can be nested under parent tags.

### Tag Class

```python
class Tag(TickTickModel):
    """
    Tag model.
    Tags are a V2-only feature for organizing tasks.
    """
```

### Tag Fields - Complete Reference

| Field | Type | Default | Alias | Description |
|-------|------|---------|-------|-------------|
| `name` | `str` | Required | - | Lowercase identifier used in API calls (no spaces/special chars) |
| `label` | `str` | Required | - | Display name shown to users (can contain spaces) |
| `raw_name` | `str \| None` | `None` | `rawName` | Original name before normalization |
| `etag` | `str \| None` | `None` | - | Version tag for concurrency |
| `color` | `str \| None` | `None` | - | Hex color (e.g., "#F18181") |
| `parent` | `str \| None` | `None` | - | Parent tag name (for nesting) |
| `sort_option` | `SortOption \| None` | `None` | `sortOption` | Sorting configuration |
| `sort_type` | `str \| None` | `None` | `sortType` | Sort type identifier |
| `sort_order` | `int \| None` | `None` | `sortOrder` | Display position |
| `type` | `int \| None` | `None` | - | Tag type (internal use) |

### Tag Naming Conventions

Understanding the three name fields is critical:

| Field | Purpose | Example |
|-------|---------|---------|
| `label` | Display name | `"Work Projects"` |
| `name` | API identifier | `"workprojects"` |
| `raw_name` | Original submission | `"Work Projects"` or `"workprojects"` |

**Rules**:
- `name` is always lowercase with no spaces
- `label` is the human-readable display name
- When creating a tag, `name` is auto-generated from `label` by lowercasing and removing spaces

### Tag Hierarchy (Nested Tags)

Tags can be nested under parent tags using the `parent` field:

```
work
├── projects
│   ├── client-a
│   └── client-b
└── meetings
```

To create nested tags:
```python
# Create parent tag
parent_tag = Tag.create(label="Work")

# Create child tag
child_tag = Tag.create(label="Projects", parent="work")  # parent is the name, not label
```

### Tag Properties

| Property | Type | Description |
|----------|------|-------------|
| `is_nested` | `bool` | True if `parent` is not None |

### Tag Factory Method

```python
@classmethod
def create(
    cls,
    label: str,
    color: str | None = None,
    parent: str | None = None,
) -> Self:
    """Create a new tag with auto-generated name."""
    name = label.lower().replace(" ", "")
    return cls(
        name=name,
        label=label,
        color=color,
        parent=parent,
    )
```

### Tag Conversion Methods

**`to_v2_create_dict()`**:
```python
{
    "label": "Work Projects",
    "name": "workprojects",
    "color": "#F18181",      # Optional
    "parent": "work",         # Optional
    "sortType": "...",        # Optional
    "sortOrder": 123          # Optional
}
```

**`to_v2_update_dict()`**:
```python
{
    "name": "workprojects",
    "label": "Work Projects",
    "rawName": "workprojects",
    "color": "#F18181",
    "parent": "work",
    "sortType": "...",
    "sortOrder": 123
}
```

### Tag Usage Notes

1. **V2 Only**: Tags are not available through the V1 API
2. **Case Insensitive**: Tag names are case-insensitive in API operations
3. **Order Not Preserved**: The API does not guarantee tag order on tasks
4. **Merging Tags**: Tags can be merged using the dedicated `merge_tags()` endpoint

---

## Section 6: Habit Models

**Source File**: `/src/ticktick_sdk/models/habit.py` (285 lines)

Habits are a **V2-only** feature for tracking recurring activities with check-in functionality.

### Habit Class

```python
class Habit(BaseModel):
    """
    Unified Habit model.

    Represents a recurring habit that can be checked in daily.
    Supports both boolean (yes/no) and numeric (count/measure) habits.
    """
```

**Note**: `Habit` inherits from `BaseModel` directly, not `TickTickModel`.

### Habit Fields - Complete Reference

#### Core Identifiers

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | Required | Unique habit identifier |
| `etag` | `str \| None` | `None` | Version tag for concurrency |

#### Basic Information

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | Required | Habit name/title |
| `icon` | `str` | `"habit_daily_check_in"` | Icon resource name |
| `color` | `str` | `"#97E38B"` | Hex color |
| `sort_order` | `int` | `0` | Display position |
| `status` | `int` | `0` | Status: `0`=Active, `2`=Archived |
| `encouragement` | `str` | `""` | Motivational message |

#### Habit Type and Goal

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `habit_type` | `str` | `"Boolean"` | Type: `"Boolean"` or `"Real"` |
| `goal` | `float` | `1.0` | Target value (1.0 for boolean, custom for numeric) |
| `step` | `float` | `0.0` | Increment step for numeric habits |
| `unit` | `str` | `"Count"` | Unit of measurement (e.g., "Pages", "km", "glasses") |
| `record_enable` | `bool` | `False` | Enable recording custom values |

#### Schedule and Reminders

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `repeat_rule` | `str \| None` | `None` | RRULE format (e.g., "RRULE:FREQ=DAILY") |
| `reminders` | `list[str]` | `[]` | Reminder times in HH:MM format |
| `section_id` | `str \| None` | `None` | Time-of-day section ID |

#### Target Tracking

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target_days` | `int` | `0` | Goal in days (0 = no target) |
| `target_start_date` | `int \| None` | `None` | Target start date (YYYYMMDD format) |
| `completed_cycles` | `int` | `0` | Number of completed cycles |
| `ex_dates` | `list[str]` | `[]` | Excluded dates |
| `style` | `int` | `1` | Display style |

#### Streak Tracking

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `current_streak` | `int` | `0` | Current consecutive day streak |
| `total_checkins` | `int` | `0` | Total number of check-ins |

#### Timestamps

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `created_time` | `datetime \| None` | `None` | Creation timestamp |
| `modified_time` | `datetime \| None` | `None` | Last modification timestamp |
| `archived_time` | `datetime \| None` | `None` | Archival timestamp |

### Habit Types

| Type | Value | Description | Goal |
|------|-------|-------------|------|
| **Boolean** | `"Boolean"` | Yes/no completion | Always `1.0` |
| **Real/Numeric** | `"Real"` | Count or measurement | Custom value |

**Boolean Habits**: For habits like "Exercise", "Meditate", "Read" where you either did it or didn't.

**Numeric Habits**: For habits like "Drink 8 glasses of water", "Walk 10,000 steps", "Read 30 pages".

### Habit Properties

| Property | Type | Description |
|----------|------|-------------|
| `is_boolean` | `bool` | True if `habit_type == "Boolean"` |
| `is_numeric` | `bool` | True if `habit_type == "Real"` |
| `is_active` | `bool` | True if `status == 0` |
| `is_archived` | `bool` | True if `status == 2` |

### Habit Conversion Methods

**`from_v2(data)`** - Create from V2 API response:

Maps V2 API field names to model fields:
- `iconRes` -> `icon`
- `type` -> `habit_type`
- `totalCheckIns` -> `total_checkins`
- `recordEnable` -> `record_enable`
- etc.

**`to_v2_dict(for_update=False)`** - For V2 API operations:

```python
{
    "id": "...",
    "name": "Drink Water",
    "iconRes": "habit_water",
    "color": "#97E38B",
    "sortOrder": 0,
    "status": 0,
    "encouragement": "Stay hydrated!",
    "totalCheckIns": 0,
    "type": "Real",
    "goal": 8.0,
    "step": 1.0,
    "unit": "glasses",
    "reminders": ["09:00", "12:00", "18:00"],
    "recordEnable": True,
    "targetDays": 30,
    "completedCycles": 0,
    "exDates": [],
    "currentStreak": 0,
    "style": 1,
    "createdTime": "...",   # Only on create
    "modifiedTime": "...",  # Always included
    "repeatRule": "RRULE:FREQ=DAILY",  # If set
    "sectionId": "...",                 # If set
    "etag": "..."                       # If set
}
```

---

### HabitSection Class

Represents a time-of-day grouping for habits (morning, afternoon, night).

```python
class HabitSection(BaseModel):
    """
    Habit section (time of day grouping).
    TickTick organizes habits into sections like morning, afternoon, night.
    """
```

#### HabitSection Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | Required | Section unique identifier |
| `name` | `str` | Required | Section name (e.g., `"_morning"`, `"_afternoon"`, `"_night"`) |
| `sort_order` | `int` | `0` | Display order |
| `created_time` | `datetime \| None` | `None` | Creation timestamp |
| `modified_time` | `datetime \| None` | `None` | Last modification |
| `etag` | `str \| None` | `None` | Version tag |

#### HabitSection Names

| Internal Name | Display Name |
|---------------|--------------|
| `_morning` | Morning |
| `_afternoon` | Afternoon |
| `_night` | Night |

#### HabitSection Properties

| Property | Type | Description |
|----------|------|-------------|
| `display_name` | `str` | Human-readable name (strips underscore, title-cases) |

---

### HabitCheckin Class

Represents a single check-in record for a habit on a specific date.

```python
class HabitCheckin(BaseModel):
    """
    Habit check-in record.
    Represents a single check-in for a habit on a specific date.
    """
```

#### HabitCheckin Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `habit_id` | `str` | Required | Parent habit ID |
| `checkin_stamp` | `int` | Required | Check-in date in YYYYMMDD format (e.g., `20241215`) |
| `checkin_time` | `datetime \| None` | `None` | Timestamp when check-in was recorded |
| `value` | `float` | `1.0` | Check-in value (1.0 for boolean, custom for numeric) |
| `goal` | `float` | `1.0` | Goal at time of check-in |
| `status` | `int` | `2` | Check-in status: `2` = Completed |

#### Checkin Date Format

The `checkin_stamp` field uses integer date format:
- `20241215` = December 15, 2024
- `20250101` = January 1, 2025

This format enables efficient date comparison and storage.

---

### HabitPreferences Class

User preferences for the habit feature.

```python
class HabitPreferences(BaseModel):
    """Habit preferences and settings."""
```

#### HabitPreferences Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `show_in_calendar` | `bool` | `True` | Show habits in calendar view |
| `show_in_today` | `bool` | `True` | Show habits in today view |
| `enabled` | `bool` | `True` | Habits feature enabled |
| `default_section_order` | `int` | `0` | Default section order |

---

## Section 7: User Models

**Source File**: `/src/ticktick_sdk/models/user.py` (143 lines)

User models are **V2-only** and provide account, subscription, and productivity information.

### User Class

User profile information.

```python
class User(TickTickModel):
    """User profile information."""
```

#### User Fields

| Field | Type | Default | Alias | Description |
|-------|------|---------|-------|-------------|
| `username` | `str` | Required | - | Account username |
| `display_name` | `str \| None` | `None` | `displayName` | Display name |
| `name` | `str \| None` | `None` | - | Full name |
| `picture` | `str \| None` | `None` | - | Avatar URL |
| `locale` | `str \| None` | `None` | - | User locale |
| `site_domain` | `str \| None` | `None` | `siteDomain` | Site domain |
| `user_code` | `str \| None` | `None` | `userCode` | User code |
| `verified_email` | `bool` | `False` | `verifiedEmail` | Email verified flag |
| `filled_password` | `bool` | `False` | `filledPassword` | Password set flag |
| `email` | `str \| None` | `None` | - | Email address |

---

### UserStatus Class

User subscription and account status information.

```python
class UserStatus(TickTickModel):
    """User subscription and account status."""
```

#### UserStatus Fields

| Field | Type | Default | Alias | Description |
|-------|------|---------|-------|-------------|
| `user_id` | `str` | Required | `userId` | User unique identifier |
| `user_code` | `str \| None` | `None` | `userCode` | User code |
| `username` | `str` | Required | - | Account username |
| `inbox_id` | `str` | Required | `inboxId` | **Inbox project ID** (critical for creating tasks) |
| `is_pro` | `bool` | `False` | `pro` | True if user has Pro subscription |
| `pro_start_date` | `str \| None` | `None` | `proStartDate` | Pro subscription start date |
| `pro_end_date` | `str \| None` | `None` | `proEndDate` | Pro subscription end date |
| `subscribe_type` | `str \| None` | `None` | `subscribeType` | Subscription type |
| `subscribe_freq` | `str \| None` | `None` | `subscribeFreq` | Subscription frequency |
| `need_subscribe` | `bool` | `False` | `needSubscribe` | Subscription needed flag |
| `free_trial` | `bool` | `False` | `freeTrial` | On free trial |
| `grace_period` | `bool` | `False` | `gracePeriod` | In grace period |
| `team_user` | `bool` | `False` | `teamUser` | Is team user |
| `team_pro` | `bool` | `False` | `teamPro` | Has team Pro |
| `active_team_user` | `bool` | `False` | `activeTeamUser` | Active team membership |

#### Important: inbox_id

The `inbox_id` field is **critical** for the SDK. When creating a task without specifying a `project_id`, the SDK uses the inbox as the default project. This ID has the format `inbox{user_id}`.

---

### UserStatistics Class

User productivity statistics and metrics.

```python
class UserStatistics(TickTickModel):
    """User productivity statistics."""
```

#### UserStatistics Fields

##### Score and Level

| Field | Type | Default | Alias | Description |
|-------|------|---------|-------|-------------|
| `score` | `int` | `0` | - | Total productivity score |
| `level` | `int` | `0` | - | User level based on score |

##### Task Completion

| Field | Type | Default | Alias | Description |
|-------|------|---------|-------|-------------|
| `yesterday_completed` | `int` | `0` | `yesterdayCompleted` | Tasks completed yesterday |
| `today_completed` | `int` | `0` | `todayCompleted` | Tasks completed today |
| `total_completed` | `int` | `0` | `totalCompleted` | Total tasks ever completed |

##### Task History

| Field | Type | Default | Alias | Description |
|-------|------|---------|-------|-------------|
| `score_by_day` | `dict[str, int]` | `{}` | `scoreByDay` | Daily scores (date -> score) |
| `task_by_day` | `dict[str, TaskCount]` | `{}` | `taskByDay` | Daily task counts |
| `task_by_week` | `dict[str, TaskCount]` | `{}` | `taskByWeek` | Weekly task counts |
| `task_by_month` | `dict[str, TaskCount]` | `{}` | `taskByMonth` | Monthly task counts |

##### Pomodoro Statistics

| Field | Type | Default | Alias | Description |
|-------|------|---------|-------|-------------|
| `today_pomo_count` | `int` | `0` | `todayPomoCount` | Pomodoros today |
| `yesterday_pomo_count` | `int` | `0` | `yesterdayPomoCount` | Pomodoros yesterday |
| `total_pomo_count` | `int` | `0` | `totalPomoCount` | Total pomodoros |
| `today_pomo_duration` | `int` | `0` | `todayPomoDuration` | Today's pomo duration (seconds) |
| `yesterday_pomo_duration` | `int` | `0` | `yesterdayPomoDuration` | Yesterday's pomo duration |
| `total_pomo_duration` | `int` | `0` | `totalPomoDuration` | Total pomo duration (seconds) |
| `pomo_goal` | `int` | `0` | `pomoGoal` | Daily pomodoro goal |
| `pomo_duration_goal` | `int` | `0` | `pomoDurationGoal` | Daily duration goal |

##### Pomodoro History

| Field | Type | Default | Alias | Description |
|-------|------|---------|-------|-------------|
| `pomo_by_day` | `dict[str, Any]` | `{}` | `pomoByDay` | Daily pomo stats |
| `pomo_by_week` | `dict[str, Any]` | `{}` | `pomoByWeek` | Weekly pomo stats |
| `pomo_by_month` | `dict[str, Any]` | `{}` | `pomoByMonth` | Monthly pomo stats |

#### UserStatistics Properties

| Property | Type | Description |
|----------|------|-------------|
| `total_pomo_duration_hours` | `float` | Total pomo duration in hours |
| `today_pomo_duration_minutes` | `float` | Today's pomo duration in minutes |

---

### TaskCount Class

Task completion count data (used within UserStatistics).

```python
class TaskCount(TickTickModel):
    """Task completion counts."""
```

#### TaskCount Fields

| Field | Type | Default | Alias | Description |
|-------|------|---------|-------|-------------|
| `complete_count` | `int` | `0` | `completeCount` | Completed tasks |
| `not_complete_count` | `int` | `0` | `notCompleteCount` | Incomplete tasks |

#### TaskCount Properties

| Property | Type | Description |
|----------|------|-------------|
| `total` | `int` | Sum of complete + incomplete |

---

## Section 8: Model Relationships Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           TickTick Model Relationships                           │
└─────────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │ ProjectGroup │ (Folder)
                              │   (V2 only)  │
                              └──────┬───────┘
                                     │ contains (via group_id)
                                     │ 1:N
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
            ┌───────────────┐                 ┌───────────────┐
            │    Project    │                 │    Project    │
            │ (id, name,    │                 │               │
            │  kind, color) │                 │               │
            └───────┬───────┘                 └───────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
┌───────────┐ ┌───────────┐ ┌───────────┐
│  Column   │ │   Task    │ │   Task    │
│ (Kanban)  │ │           │ │           │
└───────────┘ └─────┬─────┘ └───────────┘
                    │
    ┌───────────────┼───────────────┐
    │               │               │
    ▼               ▼               ▼
┌─────────┐   ┌──────────┐    ┌──────────┐
│  Task   │   │Checklist │    │   Tag    │
│(Subtask)│   │  Item    │    │ (V2 only)│
│via      │   │ (items)  │    │ via tags │
│parent_id│   └──────────┘    │   list   │
└─────────┘                   └──────────┘


                    ┌──────────────┐
                    │ HabitSection │
                    │ (time of day)│
                    └──────┬───────┘
                           │ contains (via section_id)
                           │ 1:N
               ┌───────────┴───────────┐
               ▼                       ▼
        ┌───────────┐           ┌───────────┐
        │   Habit   │           │   Habit   │
        │           │           │           │
        └─────┬─────┘           └───────────┘
              │
              │ has many (via habit_id)
              │ 1:N
              ▼
        ┌─────────────┐
        │HabitCheckin │
        │ (daily log) │
        └─────────────┘


                    ┌──────────────┐
                    │     User     │
                    │  (profile)   │
                    └──────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌────────────┐  ┌────────────┐  ┌──────────────────┐
    │ UserStatus │  │   Habit    │  │  UserStatistics  │
    │(subscription)│ │ Preferences│  │ (productivity)   │
    └────────────┘  └────────────┘  └──────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Relationship Summary                                │
├────────────────────┬────────────────────┬───────────────────────────────────────┤
│ Parent             │ Child              │ Relationship                           │
├────────────────────┼────────────────────┼───────────────────────────────────────┤
│ ProjectGroup       │ Project            │ 1:N via Project.group_id              │
│ Project            │ Task               │ 1:N via Task.project_id               │
│ Project            │ Column             │ 1:N via Column.project_id             │
│ Task               │ Task (subtask)     │ 1:N via Task.parent_id/child_ids      │
│ Task               │ ChecklistItem      │ 1:N via Task.items list               │
│ Task               │ Tag                │ N:N via Task.tags list (tag names)    │
│ Task               │ TaskReminder       │ 1:N via Task.reminders list           │
│ HabitSection       │ Habit              │ 1:N via Habit.section_id              │
│ Habit              │ HabitCheckin       │ 1:N via HabitCheckin.habit_id         │
│ UserStatistics     │ TaskCount          │ 1:N via task_by_* dicts               │
└────────────────────┴────────────────────┴───────────────────────────────────────┘
```

### Key Relationship Notes

1. **Task Subtasks**: Tasks can have subtasks in TWO ways:
   - **Via `parent_id`/`child_ids`**: V2-only, true subtask hierarchy where subtasks are full Task objects
   - **Via `items` (ChecklistItem)**: Both V1/V2, checklist items within a task

2. **Task-Tag Relationship**: Many-to-many via tag names (strings), not IDs. Tasks store tag names in the `tags` list.

3. **Project-Folder Relationship**: One-way reference. Projects reference folders via `group_id`, but folders don't store project lists.

4. **Habit Sections**: Pre-defined sections (morning/afternoon/night). Habits reference via `section_id`, but this is optional.

---

## Section 9: Field Type Reference

### Custom Type Aliases (from constants.py)

| Type Alias | Underlying Type | Description | Example |
|------------|-----------------|-------------|---------|
| `ObjectId` | `str` | 24-character hex MongoDB-style ID | `"507f1f77bcf86cd799439011"` |
| `ETag` | `str` | 8-character alphanumeric version | `"abc12345"` |
| `InboxId` | `str` | Inbox project ID | `"inbox123456"` |
| `TagLabel` | `str` | Display tag name | `"Work Projects"` |
| `TagName` | `str` | Lowercase tag identifier | `"workprojects"` |
| `TimeZoneName` | `str` | IANA timezone | `"America/New_York"` |
| `ICalTrigger` | `str` | RFC 5545 reminder trigger | `"TRIGGER:-PT30M"` |
| `RRule` | `str` | iCalendar recurrence rule | `"RRULE:FREQ=DAILY"` |
| `HexColor` | `str` | Hex color code | `"#f18181"` |

### Enumerations Reference

#### TaskStatus (IntEnum)

```python
class TaskStatus(IntEnum):
    ABANDONED = -1    # Won't do
    ACTIVE = 0        # Open/in progress
    COMPLETED_ALT = 1 # Completed (V2 alternative)
    COMPLETED = 2     # Completed (standard)
```

**Helper Methods**:
- `TaskStatus.is_completed(status: int) -> bool` - Returns True if status is 1 or 2
- `TaskStatus.is_closed(status: int) -> bool` - Returns True if status is -1, 1, or 2

#### TaskPriority (IntEnum)

```python
class TaskPriority(IntEnum):
    NONE = 0
    LOW = 1
    MEDIUM = 3
    HIGH = 5
```

**Helper Methods**:
- `TaskPriority.from_string(priority: str) -> TaskPriority` - Converts "low", "medium", "high" to enum
- `TaskPriority.to_string() -> str` - Returns lowercase name

#### TaskKind (StrEnum)

```python
class TaskKind(StrEnum):
    TEXT = "TEXT"
    NOTE = "NOTE"
    CHECKLIST = "CHECKLIST"
```

#### SubtaskStatus (IntEnum)

```python
class SubtaskStatus(IntEnum):
    NORMAL = 0
    COMPLETED = 1  # Note: Different from TaskStatus.COMPLETED (2)
```

#### ProjectKind (StrEnum)

```python
class ProjectKind(StrEnum):
    TASK = "TASK"
    NOTE = "NOTE"
```

#### ViewMode (StrEnum)

```python
class ViewMode(StrEnum):
    LIST = "list"
    KANBAN = "kanban"
    TIMELINE = "timeline"
```

#### Permission (StrEnum)

```python
class Permission(StrEnum):
    READ = "read"
    WRITE = "write"
    COMMENT = "comment"
```

#### SortOption (StrEnum)

```python
class SortOption(StrEnum):
    SORT_ORDER = "sortOrder"
    DUE_DATE = "dueDate"
    TAG = "tag"
    PRIORITY = "priority"
    PROJECT = "project"
    TITLE = "title"
    NONE = "none"
```

#### RepeatFrom (IntEnum)

```python
class RepeatFrom(IntEnum):
    DUE_DATE = 0        # Next occurrence from due date
    COMPLETED_DATE = 1  # Next occurrence from completion date
    UNKNOWN = 2
```

#### APIVersion (StrEnum)

```python
class APIVersion(StrEnum):
    V1 = "v1"
    V2 = "v2"
```

### Datetime Format Constants

| Constant | Format | Description | Example |
|----------|--------|-------------|---------|
| `DATETIME_FORMAT_V1` | `%Y-%m-%dT%H:%M:%S%z` | V1 API datetime | `2024-12-15T14:30:00+00:00` |
| `DATETIME_FORMAT_V2` | `%Y-%m-%dT%H:%M:%S.000+0000` | V2 API datetime | `2024-12-15T14:30:00.000+0000` |
| `DATETIME_FORMAT_V2_QUERY` | `%Y-%m-%d %H:%M:%S` | V2 query params | `2024-12-15 14:30:00` |
| `DATE_FORMAT_STATS` | `%Y%m%d` | Statistics dates | `20241215` |

---

## Section 10: V1 vs V2 Model Differences

### Feature Availability by API Version

| Feature | V1 API | V2 API | Notes |
|---------|--------|--------|-------|
| Task CRUD | Yes | Yes | V2 has more fields |
| Task Tags | No | Yes | V2 only |
| Task Subtasks (parent_id) | No | Yes | V2 only |
| Task Progress | No | Yes | V2 only |
| Task Attachments | No | Yes | V2 only |
| Task Focus/Pomo | No | Yes | V2 only |
| Projects | Yes | Yes | V2 has more metadata |
| Project Folders | No | Yes | V2 only |
| Kanban Columns | Yes | Yes | Via get_project_with_data |
| Tags | No | Yes | V2 only |
| Habits | No | Yes | V2 only |
| User Profile | No | Yes | V2 only |
| User Statistics | No | Yes | V2 only |

### Task Field Comparison

| Field | V1 Response | V2 Response | SDK Field |
|-------|-------------|-------------|-----------|
| ID | `id` | `id` | `id` |
| Project ID | `projectId` | `projectId` | `project_id` |
| ETag | - | `etag` | `etag` |
| Title | `title` | `title` | `title` |
| Content | `content` | `content` | `content` |
| Description | `desc` | `desc` | `desc` |
| Kind | `kind` | `kind` | `kind` |
| Status | `status` | `status` | `status` |
| Priority | `priority` | `priority` | `priority` |
| Progress | - | `progress` | `progress` |
| Deleted | - | `deleted` | `deleted` |
| Start Date | `startDate` | `startDate` | `start_date` |
| Due Date | `dueDate` | `dueDate` | `due_date` |
| Created Time | `createdTime` | `createdTime` | `created_time` |
| Modified Time | `modifiedTime` | `modifiedTime` | `modified_time` |
| Completed Time | `completedTime` | `completedTime` | `completed_time` |
| Timezone | `timeZone` | `timeZone` | `time_zone` |
| All Day | `isAllDay` | `isAllDay` | `is_all_day` |
| Floating | - | `isFloating` | `is_floating` |
| Repeat Flag | `repeatFlag` | `repeatFlag` | `repeat_flag` |
| Repeat From | - | `repeatFrom` | `repeat_from` |
| Reminders | `reminders` (strings) | `reminders` (objects) | `reminders` |
| Parent ID | - | `parentId` | `parent_id` |
| Child IDs | - | `childIds` | `child_ids` |
| Items | `items` | `items` | `items` |
| Tags | - | `tags` | `tags` |
| Column ID | - | `columnId` | `column_id` |
| Sort Order | `sortOrder` | `sortOrder` | `sort_order` |
| Assignee | - | `assignee` | `assignee` |
| Creator | - | `creator` | `creator` |
| Attachments | - | `attachments` | `attachments` |
| Focus Summaries | - | `focusSummaries` | `focus_summaries` |
| Pomo Summaries | - | `pomodoroSummaries` | `pomodoro_summaries` |

### How the SDK Handles Missing Fields

1. **V2-only fields from V1 response**: Set to default values (None or empty list)
2. **Field aliases**: Allow either `projectId` (camelCase) or `project_id` (snake_case)
3. **Extra fields ignored**: Pydantic config `extra="ignore"` drops unknown fields
4. **Datetime parsing**: Automatic detection of V1 vs V2 format

### Datetime Format Differences

| API | Format | Example |
|-----|--------|---------|
| V1 Input | `%Y-%m-%dT%H:%M:%S%z` | `2024-12-15T14:30:00+00:00` |
| V2 Input | `%Y-%m-%dT%H:%M:%S.000+0000` | `2024-12-15T14:30:00.000+0000` |
| V1 Output | Same as input | - |
| V2 Output | Same as input | - |

The SDK's `parse_datetime()` method handles both formats automatically. The `format_datetime()` method accepts a `for_api` parameter to output the correct format.

---

## Section 11: Model Validation Rules

### Required vs Optional Fields

#### Task Model
- **Required**: `id`, `project_id`
- **All other fields**: Optional with defaults

#### Project Model
- **Required**: `id`, `name`
- **All other fields**: Optional with defaults

#### ProjectGroup Model
- **Required**: `id`, `name`
- **All other fields**: Optional with defaults

#### Tag Model
- **Required**: `name`, `label`
- **All other fields**: Optional with defaults

#### Habit Model
- **Required**: `id`, `name`
- **All other fields**: Optional with defaults

#### User Models
- **User**: `username` required
- **UserStatus**: `user_id`, `username`, `inbox_id` required
- **UserStatistics**: All optional with defaults

### Default Values Reference

| Model | Field | Default |
|-------|-------|---------|
| Task | `kind` | `"TEXT"` |
| Task | `status` | `0` (ACTIVE) |
| Task | `priority` | `0` (NONE) |
| Task | `deleted` | `0` |
| Task | `is_floating` | `False` |
| Task | `tags` | `[]` |
| Task | `items` | `[]` |
| Task | `reminders` | `[]` |
| Task | `attachments` | `[]` |
| Project | `kind` | `"TASK"` |
| Project | `view_mode` | `"list"` |
| ProjectGroup | `deleted` | `0` |
| ProjectGroup | `show_all` | `False` |
| Habit | `icon` | `"habit_daily_check_in"` |
| Habit | `color` | `"#97E38B"` |
| Habit | `status` | `0` |
| Habit | `habit_type` | `"Boolean"` |
| Habit | `goal` | `1.0` |
| Habit | `step` | `0.0` |
| Habit | `unit` | `"Count"` |
| HabitCheckin | `value` | `1.0` |
| HabitCheckin | `goal` | `1.0` |
| HabitCheckin | `status` | `2` |

### Field Validators

#### Automatic Datetime Parsing

All datetime fields in `Task`, `ChecklistItem`, `Project`, and other models use the `parse_datetime_field` validator:

```python
@field_validator("start_date", "due_date", ..., mode="before")
@classmethod
def parse_datetime_field(cls, v: Any) -> datetime | None:
    return cls.parse_datetime(v)
```

This validator:
- Accepts `None`, `datetime`, or `str`
- Parses strings using multiple format attempts
- Returns `None` if parsing fails (no exception)

#### List Field Validators

**Task.reminders**:
- Accepts `None` -> `[]`
- Accepts list of strings -> `[TaskReminder(trigger=s) for s in list]`
- Accepts list of dicts -> `[TaskReminder.model_validate(d) for d in list]`

**Task.items**:
- Accepts `None` -> `[]`
- Accepts list of dicts -> `[ChecklistItem.model_validate(d) for d in list]`

**ProjectData.columns**:
- Accepts `None` -> `[]`
- Accepts list of dicts -> `[Column.model_validate(c) for c in list]`

### Pydantic Model Configuration Effects

| Setting | Effect |
|---------|--------|
| `populate_by_name=True` | Can use `Task(project_id="...")` or `Task(projectId="...")` |
| `validate_assignment=True` | `task.priority = "invalid"` raises ValidationError |
| `extra="ignore"` | Unknown API fields are silently dropped |
| `use_enum_values=True` | `TaskPriority.HIGH` serializes as `5`, not `"HIGH"` |

---

## Section 12: Working with Models (Practical Guide)

### Creating Model Instances

#### Creating a Task

```python
from ticktick_sdk.models import Task
from datetime import datetime, timezone

# Minimal task
task = Task(
    id="task123",
    project_id="project456",
    title="Buy groceries"
)

# Task with all common fields
task = Task(
    id="task123",
    project_id="project456",
    title="Buy groceries",
    content="Milk, eggs, bread",
    priority=5,  # HIGH
    start_date=datetime.now(timezone.utc),
    due_date=datetime(2024, 12, 20, 18, 0, 0, tzinfo=timezone.utc),
    is_all_day=False,
    tags=["shopping", "personal"],
    reminders=[TaskReminder(trigger="TRIGGER:-PT30M")]
)
```

#### Creating a Project

```python
from ticktick_sdk.models import Project

project = Project(
    id="project123",
    name="Work Tasks",
    color="#F18181",
    kind="TASK",
    view_mode="kanban"
)
```

#### Creating a Tag

```python
from ticktick_sdk.models import Tag

# Using factory method (recommended)
tag = Tag.create(
    label="Work Projects",
    color="#4A90D9",
    parent="work"  # Nested under 'work' tag
)

# Manual creation
tag = Tag(
    name="workprojects",
    label="Work Projects",
    color="#4A90D9"
)
```

#### Creating a Habit

```python
from ticktick_sdk.models import Habit

# Boolean habit
habit = Habit(
    id="habit123",
    name="Exercise",
    habit_type="Boolean",
    goal=1.0,
    color="#97E38B",
    reminders=["07:00"]
)

# Numeric habit
habit = Habit(
    id="habit456",
    name="Drink Water",
    habit_type="Real",
    goal=8.0,
    step=1.0,
    unit="glasses",
    record_enable=True
)
```

### Modifying Models

```python
# Direct attribute assignment (validates on assignment)
task.title = "Updated title"
task.priority = 3  # MEDIUM
task.tags.append("urgent")

# Modifying nested objects
task.items.append(ChecklistItem(
    id="item1",
    title="Buy milk",
    status=0
))
```

### Serializing for API Calls

#### For V1 API

```python
# Get dict for V1 API request
v1_data = task.to_v1_dict()

# Result:
{
    "id": "task123",
    "projectId": "project456",
    "title": "Buy groceries",
    "priority": 5,
    "startDate": "2024-12-15T14:30:00+00:00",
    "dueDate": "2024-12-20T18:00:00+00:00",
    "isAllDay": False,
    "reminders": ["TRIGGER:-PT30M"],
    "items": [{"id": "item1", "title": "Buy milk", "status": 0}]
}
# Note: No 'tags' field (V1 doesn't support tags)
```

#### For V2 API

```python
# For creation
v2_data = task.to_v2_dict()

# For update (includes empty strings to clear dates)
v2_data = task.to_v2_dict(for_update=True)

# Result:
{
    "id": "task123",
    "projectId": "project456",
    "title": "Buy groceries",
    "priority": 5,
    "startDate": "2024-12-15T14:30:00.000+0000",
    "dueDate": "2024-12-20T18:00:00.000+0000",
    "isAllDay": False,
    "tags": ["shopping", "personal"],
    "reminders": [{"trigger": "TRIGGER:-PT30M"}],
    "items": [...]
}
```

### Deserializing from API Responses

#### From V1 Response

```python
# V1 API response data
v1_response = {
    "id": "task123",
    "projectId": "project456",
    "title": "Buy groceries",
    "priority": 5,
    "startDate": "2024-12-15T14:30:00+00:00",
    "status": 0
}

task = Task.from_v1(v1_response)
# task.start_date is now a datetime object
# task.tags is [] (V1 doesn't provide tags)
```

#### From V2 Response

```python
# V2 API response data
v2_response = {
    "id": "task123",
    "projectId": "project456",
    "title": "Buy groceries",
    "priority": 5,
    "startDate": "2024-12-15T14:30:00.000+0000",
    "status": 0,
    "tags": ["shopping"],
    "etag": "abc12345"
}

task = Task.from_v2(v2_response)
# task.start_date is now a datetime object
# task.tags is ["shopping"]
# task.etag is "abc12345"
```

### Working with Task Properties

```python
# Check task state
if task.is_completed:
    print(f"Completed at: {task.completed_time}")

if task.is_subtask:
    print(f"Parent task: {task.parent_id}")

if task.has_subtasks:
    print(f"Subtask IDs: {task.child_ids}")

# Get human-readable priority
print(f"Priority: {task.priority_label}")  # "high", "medium", etc.
```

### Working with Habits and Check-ins

```python
from ticktick_sdk.models import Habit, HabitCheckin

# Check habit type
if habit.is_boolean:
    print("This is a yes/no habit")
elif habit.is_numeric:
    print(f"Target: {habit.goal} {habit.unit}")

# Create a check-in
checkin = HabitCheckin(
    habit_id=habit.id,
    checkin_stamp=20241215,  # December 15, 2024
    value=1.0,
    status=2  # Completed
)
```

### Common Patterns

#### Finding Tasks by Tag

```python
tasks = await client.get_all_tasks()
work_tasks = [t for t in tasks if "work" in t.tags]
```

#### Filtering Overdue Tasks

```python
from datetime import datetime, timezone

now = datetime.now(timezone.utc)
overdue = [
    t for t in tasks
    if t.due_date and t.due_date < now and t.is_active
]
```

#### Creating a Subtask (V2 Only)

```python
# 1. Create the task
child_task = await client.create_task(
    title="Subtask",
    project_id=project_id
)

# 2. Make it a subtask (parent_id is ignored on create)
await client.make_subtask(
    task_id=child_task.id,
    parent_id=parent_task.id,
    project_id=project_id
)
```

#### Handling Recurrence

```python
# IMPORTANT: Recurrence requires start_date
task = await client.create_task(
    title="Daily standup",
    start_date=datetime.now(timezone.utc),  # Required!
    recurrence="RRULE:FREQ=DAILY;BYDAY=MO,TU,WE,TH,FR"
)
```

---

## Appendix: Source File Reference

| File | Location | Models Defined |
|------|----------|----------------|
| `models/__init__.py` | `/src/ticktick_sdk/models/__init__.py` | Exports all public models |
| `models/base.py` | `/src/ticktick_sdk/models/base.py` | `TickTickModel` |
| `models/task.py` | `/src/ticktick_sdk/models/task.py` | `Task`, `ChecklistItem`, `TaskReminder` |
| `models/project.py` | `/src/ticktick_sdk/models/project.py` | `Project`, `ProjectGroup`, `Column`, `ProjectData`, `SortOption` |
| `models/tag.py` | `/src/ticktick_sdk/models/tag.py` | `Tag` |
| `models/habit.py` | `/src/ticktick_sdk/models/habit.py` | `Habit`, `HabitSection`, `HabitCheckin`, `HabitPreferences` |
| `models/user.py` | `/src/ticktick_sdk/models/user.py` | `User`, `UserStatus`, `UserStatistics`, `TaskCount` |
| `constants.py` | `/src/ticktick_sdk/constants.py` | All enums and constants |
| `api/v1/types.py` | `/src/ticktick_sdk/api/v1/types.py` | V1 TypedDicts (reference) |
| `api/v2/types.py` | `/src/ticktick_sdk/api/v2/types.py` | V2 TypedDicts (reference) |

---

## Document Metadata

- **Total Models Documented**: 14 unified models + 6 enums
- **Source Files Referenced**: 10
- **Generated For**: TickTick SDK v0.2.1
- **Pydantic Version**: v2.0+
- **Python Version**: 3.11+
