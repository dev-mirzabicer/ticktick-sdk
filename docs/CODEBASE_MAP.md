# TickTick SDK - Comprehensive Codebase Map

> **Purpose**: This document provides a complete structural map of the TickTick SDK codebase. It serves as the foundational reference for all other documentation and enables developers/agents to understand the entire system without reading 14,000+ lines of source code.

---

## Executive Summary

This is a production-grade Python SDK for TickTick (3K+ downloads) that reverse-engineers both V1 (official OAuth2 API) and V2 (unofficial session API) to provide comprehensive task management capabilities. The codebase is 14,585 lines of Python across 32 source modules, with 11 test suites covering 300+ test cases.

| Metric | Value |
|--------|-------|
| **Current Version** | 0.2.1 |
| **License** | MIT |
| **Python Support** | 3.11, 3.12, 3.13 |
| **Key Dependencies** | httpx (async HTTP), pydantic v2 (validation), MCP v1.0+ (AI integration) |

---

## Part 1: Project Structure & Organization

### 1.1 Directory Tree

```
ticktick-mcp/
├── src/ticktick_sdk/              # Main SDK source code
│   ├── __init__.py                # Public API surface (28 exports)
│   ├── api/                       # HTTP API clients
│   │   ├── base.py               # BaseTickTickClient (abstract HTTP layer)
│   │   ├── v1/                   # Official OAuth2 API
│   │   │   ├── client.py         # TickTickV1Client (531 lines)
│   │   │   ├── auth.py           # OAuth2Handler & OAuth2Token
│   │   │   └── types.py          # V1 request/response types
│   │   └── v2/                   # Unofficial session API
│   │       ├── client.py         # TickTickV2Client (1521 lines)
│   │       ├── auth.py           # SessionHandler & SessionToken
│   │       └── types.py          # V2 request/response types (798 lines)
│   ├── client/                    # High-level user-facing client
│   │   ├── __init__.py
│   │   └── client.py             # TickTickClient main class (1070 lines)
│   ├── unified/                   # API routing & unification
│   │   ├── api.py                # UnifiedTickTickAPI (1968 lines)
│   │   └── router.py             # APIRouter with operation routing table (321 lines)
│   ├── models/                    # Unified Pydantic data models
│   │   ├── base.py               # TickTickModel base class
│   │   ├── task.py               # Task & ChecklistItem (339 lines)
│   │   ├── project.py            # Project, ProjectGroup, Column (273 lines)
│   │   ├── habit.py              # Habit, HabitSection, HabitCheckin (285 lines)
│   │   ├── tag.py                # Tag model
│   │   ├── user.py               # User, UserStatus, UserStatistics
│   │   └── __init__.py           # Models public API
│   ├── server.py                  # MCP server with 45 tools (2445 lines)
│   ├── cli.py                     # Command-line interface (261 lines)
│   ├── auth_cli.py               # OAuth2 flow CLI (575 lines)
│   ├── settings.py               # Configuration via environment (268 lines)
│   ├── constants.py              # Enums, URLs, timeouts
│   ├── exceptions.py             # Exception hierarchy (271 lines)
│   └── tools/                     # MCP tool utilities
│       ├── formatting.py         # Markdown/JSON formatters (431 lines)
│       └── inputs.py             # MCP tool input models (917 lines)
├── tests/                         # Test suite (11 test modules)
│   ├── conftest.py               # Pytest fixtures & mock clients
│   ├── test_client_tasks.py      # Task CRUD tests
│   ├── test_client_projects.py   # Project tests
│   ├── test_client_tags.py       # Tag tests
│   ├── test_client_habits.py     # Habit tests
│   ├── test_client_folders.py    # Folder tests
│   ├── test_client_user.py       # User tests
│   ├── test_client_focus_sync.py # Focus/sync tests
│   ├── test_client_lifecycle.py  # Connection lifecycle
│   └── test_client_errors.py     # Error handling
├── docs/                          # Documentation
├── pyproject.toml                 # Package configuration
├── README.md                      # User documentation (1000+ lines)
├── LICENSE.md                     # MIT License
└── .env.example                   # Environment variable template
```

### 1.2 Key Configuration Files

**pyproject.toml:**
- Build system: hatchling
- Package name: `ticktick-sdk`
- Entry point: `ticktick-sdk = "ticktick_sdk.cli:cli_main"`
- Dependencies: httpx>=0.27.0, pydantic>=2.0.0, pydantic-settings>=2.0.0, mcp>=1.0.0
- Dev dependencies: pytest, pytest-asyncio, respx (httpx mocking), freezegun (time mocking)
- Python version: 3.11+ minimum

---

## Part 2: Core Architecture

### 2.1 Three-Layer Architecture Pattern

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1: User Application or MCP Server                │
│  (Your code or AI assistants via Model Context Protocol)│
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│  Layer 2: TickTickClient (High-level facade)            │
│  File: client/client.py (1070 lines)                    │
│  - Convenience methods (get_today_tasks, etc.)          │
│  - Async context manager lifecycle                      │
│  - Single entry point for all operations                │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│  Layer 3: UnifiedTickTickAPI (Routing & conversion)     │
│  File: unified/api.py (1968 lines)                      │
│  - Routes operations to V1 or V2                        │
│  - Converts unified models ↔ API-specific formats       │
│  - Error handling and batch operations                  │
└─────────────────────────┬───────────────────────────────┘
                  ┌───────┴───────┐
         ┌────────▼──────┐  ┌─────▼──────────┐
         │  V1 API       │  │  V2 API        │
         │  (OAuth2)     │  │  (Session)     │
         │  Official     │  │  Unofficial    │
         │  Limited      │  │  Full-featured │
         └───────────────┘  └────────────────┘
```

### 2.2 API Version Comparison

| Aspect | V1 API (Official OAuth2) | V2 API (Reverse-Engineered Session) |
|--------|--------------------------|-------------------------------------|
| **Base URL** | `https://api.ticktick.com/open/v1` | `https://api.ticktick.com/api/v2` |
| **Authentication** | OAuth2 Bearer token | Session token + Bearer token + cookies |
| **Use cases** | Project with tasks (basic operations) | Tags, habits, focus, subtasks, folders |
| **Limitations** | No tags, habits, focus tracking | None (comprehensive feature set) |

### 2.3 Routing Strategy (router.py)

| Resource | Primary API | Fallback | Reason |
|----------|-------------|----------|--------|
| Tasks | V2 | V1 | V2 has more fields (tags, subtasks) |
| complete_task | V1 | - | Simpler dedicated endpoint |
| Projects | V1 | V2 | V1 has dedicated endpoint |
| Tags | V2 only | - | Not available in V1 |
| Habits | V2 only | - | Not available in V1 |
| User | V2 only | - | Not available in V1 |
| Focus | V2 only | - | Not available in V1 |

---

## Part 3: Authentication Mechanisms

### 3.1 V1 OAuth2 Flow

**Location:** `api/v1/auth.py` (341 lines)

**OAuth2Handler Class:**
- Manages authorization code flow with PKCE
- Generates authorization URL
- Exchanges authorization code for access token
- Token refresh support
- Scopes: `["tasks:read", "tasks:write"]`

**OAuth2Token Dataclass:**
```python
access_token: str
token_type: str = "Bearer"
expires_in: int | None
refresh_token: str | None
scope: str | None
created_at: datetime  # For expiry checking (60-second buffer)
```

### 3.2 V2 Session Flow

**Location:** `api/v2/auth.py` (366 lines)

**SessionHandler Class:**
- POST `/user/signon` with username/password
- Receives token and session cookies
- 2FA detection (manual input required)
- Device ID generation (MongoDB ObjectId format)

**SessionToken Dataclass:**
```python
token: str
user_id: str
username: str
inbox_id: str
user_code: str | None
is_pro: bool
pro_start_date: str | None
pro_end_date: str | None
cookies: dict[str, str]  # Session cookies for subsequent requests
```

### 3.3 Settings Integration

**Location:** `settings.py` (268 lines)

**TickTickSettings Class (Pydantic BaseSettings):**

| Category | Variable | Type | Default |
|----------|----------|------|---------|
| V1 OAuth2 | `TICKTICK_CLIENT_ID` | str | required |
| V1 OAuth2 | `TICKTICK_CLIENT_SECRET` | SecretStr | required |
| V1 OAuth2 | `TICKTICK_REDIRECT_URI` | str | `http://localhost:8080/callback` |
| V1 OAuth2 | `TICKTICK_ACCESS_TOKEN` | SecretStr | optional |
| V1 OAuth2 | `TICKTICK_REFRESH_TOKEN` | SecretStr | optional |
| V2 Session | `TICKTICK_USERNAME` | str | required |
| V2 Session | `TICKTICK_PASSWORD` | SecretStr | required |
| General | `TICKTICK_TIMEOUT` | float | 30 seconds |
| General | `TICKTICK_TIMEZONE` | str | UTC |
| General | `TICKTICK_DEVICE_ID` | str | auto-generated |

---

## Part 4: Data Models (Unified Layer)

### 4.1 Base Model

**Location:** `models/base.py`

**TickTickModel Class (Pydantic BaseModel):**

Configuration:
- `populate_by_name=True` (accept field name or alias)
- `extra="ignore"` (allow unknown fields from API)
- `validate_assignment=True`
- `use_enum_values=True`

Key Methods:
- `parse_datetime()` - Handles V1/V2 datetime formats
- `format_datetime()` - Format for API submission
- `to_v1_dict()` / `to_v2_dict()` - Conversion for submission
- `from_v1()` / `from_v2()` - Create from API response

### 4.2 Task Model

**Location:** `models/task.py` (339 lines)

**Task Class Fields:**

| Category | Field | Type | Notes |
|----------|-------|------|-------|
| **Identifiers** | id | str | required |
| | project_id | str | alias: projectId |
| | etag | str \| None | for concurrency |
| **Content** | title | str \| None | |
| | content | str \| None | notes |
| | desc | str \| None | checklist description |
| | kind | str | TEXT, NOTE, CHECKLIST |
| **Status** | status | int | ACTIVE=0, COMPLETED=2, ABANDONED=-1 |
| | priority | int | NONE=0, LOW=1, MEDIUM=3, HIGH=5 |
| | progress | int \| None | 0-100 for checklists |
| | deleted | int | 0 or 1 (soft delete) |
| **Dates** | start_date | datetime \| None | |
| | due_date | datetime \| None | |
| | created_time | datetime \| None | |
| | modified_time | datetime \| None | |
| | completed_time | datetime \| None | |
| | time_zone | str \| None | |
| | is_all_day | bool \| None | |
| **Organization** | tags | list[str] | V2 only |
| | parent_id | str \| None | V2 only (subtask parent) |
| | child_ids | list[str] | V2 only |
| **Metadata** | repeat_flag | str \| None | RRULE format |
| | reminders | list[TaskReminder] | |
| | checklists | list[ChecklistItem] | subtasks |
| | creator | str \| None | V2 only |
| | assignee | str \| None | V2 only |

**ChecklistItem (Subtask) Fields:**
- `id: str`
- `title: str | None`
- `status: int` (NORMAL=0, COMPLETED=2)
- `completed_time: datetime | None`
- `start_date: datetime | None`
- `is_all_day: bool | None`

**TaskReminder Class:**
- `trigger: str` (ICalTrigger format, e.g., "TRIGGER:-PT30M")

### 4.3 Project Models

**Location:** `models/project.py` (273 lines)

**Project Class:**

| Category | Field | Type | Notes |
|----------|-------|------|-------|
| **Identifiers** | id | str | |
| | etag | str \| None | |
| **Content** | name | str | |
| | description | str \| None | |
| | kind | str | TASK or NOTE |
| **Organization** | folder_id | str \| None | parent folder |
| | color | str \| None | hex color |
| | view_mode | str | list, kanban, timeline |
| **Status** | is_archived | bool \| None | |
| | closed | int \| None | |
| **Metadata** | owner | str \| None | |
| | permission | str \| None | read, write, comment |

**ProjectGroup (Folder) Class:**
- `id: str`
- `name: str`
- `view_mode: str | None`
- `sort_option: SortOption | None`
- `sort_order: int | None`
- `deleted: int` (0 or 1)
- `show_all: bool`

**Column (Kanban) Class:**
- `id: str`
- `project_id: str`
- `name: str`
- `sort_order: int | None`

**ProjectData Class:**
- Container for `get_project_with_data` response
- Fields: `project: Project`, `tasks: list[Task]`, `columns: list[Column]`

### 4.4 Tag Model

**Location:** `models/tag.py`

**Tag Class:**

| Field | Type | Notes |
|-------|------|-------|
| name | str | lowercase identifier, used in API calls |
| label | str | display name |
| raw_name | str \| None | |
| color | str \| None | hex color |
| parent | str \| None | parent tag name |
| is_nested | bool | computed property |
| sort_option | SortOption \| None | |
| sort_order | int \| None | |

### 4.5 Habit Models

**Location:** `models/habit.py` (285 lines)

**Habit Class:**

| Category | Field | Type | Notes |
|----------|-------|------|-------|
| **Identifiers** | id | str | |
| | etag | str \| None | |
| **Content** | name | str | |
| | icon | str | resource name |
| | color | str | hex |
| | encouragement | str | motivation text |
| **Type & Goals** | habit_type | str | Boolean or Real |
| | goal | float | 1.0 for boolean, numeric for real |
| | step | float | increment for numeric |
| | unit | str | e.g., "Pages", "km" |
| **Schedule** | repeat_rule | str | RRULE format |
| | reminders | list[str] | HH:MM format |
| | target_days | int | goal in days, 0=none |
| | section_id | str \| None | morning/afternoon/night |
| **Status** | archived | bool | |
| | deleted | int | |
| **Streak** | current_streak | int | |
| | total_checkins | int | |
| | best_streak | int | |
| **Dates** | created_time | datetime \| None | |
| | modified_time | datetime \| None | |

**HabitSection Class:**
- `id: str`
- `name: str` (_morning, _afternoon, _night)
- `sort_order: int`
- `display_name: str` (property, readable name)

**HabitCheckin Class:**
- `id: str`
- `habit_id: str`
- `checkin_date: date`
- `value: float`
- `status: int` (NORMAL, COMPLETED)
- `created_time: datetime | None`

### 4.6 User Models

**Location:** `models/user.py`

**User Class:**
- `username: str`
- `display_name: str | None`
- `email: str | None`
- `picture: str | None` (avatar URL)
- `locale: str | None`
- `verified_email: bool`
- `filled_password: bool`

**UserStatus Class:**
- `user_id: str`
- `username: str`
- `inbox_id: str` (special project for inbox)
- `is_pro: bool`
- `pro_start_date: str | None`
- `pro_end_date: str | None`
- `subscribe_type: str | None`
- `team_user: bool`
- `team_pro: bool`

**UserStatistics Class:**
- `level: int` (productivity level)
- `score: int` (total score)
- `today_completed: int` (tasks today)
- `all_completed: int` (tasks ever)
- `all_total: int` (total tasks ever)
- `all_week_completed: int`
- `all_week_total: int`

---

## Part 5: API Layer Implementation

### 5.1 Base HTTP Client

**Location:** `api/base.py` (469 lines)

**BaseTickTickClient Abstract Class:**
- Lifecycle: `__aenter__`, `__aexit__` (context manager)
- HTTP client setup with `httpx.AsyncClient`
- Error handling and response parsing

Key Methods:
- `_get_base_headers()` - Common headers (User-Agent)
- `_request()` - Core HTTP method with error handling
- `_handle_response()` - Parse response and raise exceptions
- `initialize()` - Setup and authentication
- `close()` - Cleanup

Error Mapping:
| Status Code | Exception |
|-------------|-----------|
| 401/403 | TickTickAuthenticationError |
| 404 | TickTickNotFoundError |
| 429 | TickTickRateLimitError |
| 403 | TickTickForbiddenError |
| 5xx | TickTickServerError |

### 5.2 V1 Client

**Location:** `api/v1/client.py` (531 lines)

**TickTickV1Client Class:**
- API version: `APIVersion.V1`
- Base URL: `https://api.ticktick.com/open/v1`

Authentication Methods:
- `get_authorization_url()` - Generate OAuth URL
- `authenticate_with_code()` - Exchange code for token
- `set_access_token()` - Set existing token

Endpoints:
```
Tasks:
  - get_project_task(project_id, task_id) → dict
  - create_task(data) → dict
  - update_task(project_id, task_id, data) → dict
  - complete_task(project_id, task_id) → dict
  - delete_task(project_id, task_id) → dict

Projects:
  - get_projects() → list[dict]
  - get_project(project_id) → dict
  - get_project_data(project_id) → dict (with tasks & columns)
  - create_project(data) → dict
  - update_project(project_id, data) → dict
  - delete_project(project_id) → dict
```

### 5.3 V2 Client

**Location:** `api/v2/client.py` (1521 lines)

**TickTickV2Client Class:**
- API version: `APIVersion.V2`
- Base URL: `https://api.ticktick.com/api/v2`

Authentication Methods:
- `authenticate()` - Login with username/password
- `set_session_token()` - Set existing session
- `refresh_session()` - Refresh session

Endpoints:
```
Authentication:
  - post_signon(username, password) → SessionToken

User:
  - get_status() → UserStatusV2
  - get_profile() → UserProfileV2
  - get_preferences() → UserPreferencesV2

Sync:
  - batch_check(sync_time) → SyncStateV2

Tasks (Batch):
  - batch_task(request) → BatchResponseV2
  - get_task(task_id) → TaskV2
  - batch_task_project(requests) → BatchResponseV2 (move)
  - batch_task_parent(requests) → BatchTaskParentResponseV2 (subtasks)

Task Queries:
  - get_completed_tasks(start_date, end_date) → list[TaskV2]
  - get_abandoned_tasks() → list[TaskV2]
  - get_deleted_tasks() → TrashResponseV2

Projects (Batch):
  - batch_project(request) → BatchResponseV2

Project Groups (Batch):
  - batch_project_group(request) → BatchResponseV2

Tags (Batch):
  - batch_tag(request) → BatchResponseV2
  - rename_tag(old_name, new_name)
  - merge_tags(source, target)
  - delete_tag(name)

Focus/Pomodoro:
  - get_focus_heatmap(from_date, to_date) → FocusHeatmapV2
  - get_focus_by_tag(from_date, to_date) → FocusDistributionV2

Habits:
  - batch_habit(request) → BatchResponseV2
  - get_habit_sections() → list[HabitSectionV2]
  - batch_habit_checkin(requests) → BatchResponseV2
  - query_habit_checkins(habit_ids, after_stamp) → dict
  - get_habit_preferences() → HabitPreferencesV2

Statistics:
  - get_statistics() → UserStatisticsV2
```

---

## Part 6: Unified API Layer

**Location:** `unified/api.py` (1968 lines)

### 6.1 UnifiedTickTickAPI Class

Purpose:
- Single source of truth for version-agnostic operations
- Routes calls to V1 or V2 based on APIRouter
- Converts unified models ↔ API-specific formats
- Manages both V1 and V2 client lifecycle

Constructor Parameters:
- `client_id: str`
- `client_secret: str`
- `redirect_uri: str`
- `v1_access_token: str | None`
- `username: str | None`
- `password: str | None`
- `timeout: float`
- `device_id: str | None`

### 6.2 Operation Categories

**Tasks (15 methods):**
- `create_task()`, `get_task()`, `update_task()`, `delete_task()`
- `complete_task()`, `list_all_tasks()`, `move_task()`
- `set_task_parent()`, `unset_task_parent()` (subtasks)
- `list_completed_tasks()`, `list_deleted_tasks()`, `list_abandoned_tasks()`
- Query helpers: `get_today_tasks()`, `search_tasks()`

**Projects (8 methods):**
- `create_project()`, `get_project()`, `update_project()`, `delete_project()`
- `list_projects()`, `get_project_with_data()`
- `sync_all()` (complete state)

**Folders (4 methods):**
- `create_project_group()`, `update_project_group()`, `delete_project_group()`
- `list_project_groups()`

**Tags (7 methods):**
- `create_tag()`, `update_tag()`, `delete_tag()`
- `rename_tag()`, `merge_tags()`
- `list_tags()`

**Habits (11 methods):**
- `create_habit()`, `update_habit()`, `delete_habit()`
- `list_habits()`, `get_habit()`
- `checkin_habit()` (with backdating support)
- `archive_habit()`, `unarchive_habit()`
- `list_habit_sections()`, `get_habit_preferences()`
- `get_habit_checkins()`

**User & Analytics (6 methods):**
- `get_user_profile()`, `get_user_status()`, `get_user_statistics()`
- `get_user_preferences()`
- `get_focus_heatmap()`, `get_focus_by_tag()`

### 6.3 Batch Error Handling

Helper Function: `_check_batch_response_errors()`
- Checks V2 batch response `id2error` field
- Maps error codes to semantic exceptions:
  - `TASK_NOT_FOUND` → TickTickNotFoundError
  - `EXCEED_QUOTA` → TickTickQuotaExceededError
  - Others → TickTickAPIError

---

## Part 7: API Routing Logic

**Location:** `unified/router.py` (321 lines)

### 7.1 APIRouter Class

**APIPreference Enum:**
```python
V1_ONLY = auto()    # Only in V1
V2_ONLY = auto()    # Only in V2
V2_PRIMARY = auto() # Try V2 first, fallback to V1
V1_PRIMARY = auto() # Try V1 first, fallback to V2
```

### 7.2 Operation Routing Table

| Operation | Routing | Reason |
|-----------|---------|--------|
| **Tasks** | | |
| create_task | V2_PRIMARY | tags support |
| get_task | V2_PRIMARY | no project_id needed |
| update_task | V2_PRIMARY | richer options |
| complete_task | V1_PRIMARY | simpler endpoint |
| list_all_tasks | V2_ONLY | V1 can only list per-project |
| move_task | V2_ONLY | |
| set_task_parent | V2_ONLY | subtasks |
| **Projects** | | |
| create_project | V2_PRIMARY | more options |
| get_project | V1_PRIMARY | dedicated endpoint |
| get_project_with_data | V1_ONLY | includes tasks + columns |
| list_projects | V1_PRIMARY | |
| update_project | V2_PRIMARY | batch operations |
| delete_project | V2_PRIMARY | batch operations |
| **Tags** | | |
| create_tag | V2_ONLY | |
| update_tag | V2_ONLY | |
| delete_tag | V2_ONLY | |
| rename_tag | V2_ONLY | |
| merge_tags | V2_ONLY | |
| list_tags | V2_ONLY | |
| **Habits** | | |
| create_habit | V2_ONLY | |
| update_habit | V2_ONLY | |
| delete_habit | V2_ONLY | |
| list_habits | V2_ONLY | |
| checkin_habit | V2_ONLY | |
| **User** | | |
| get_user_profile | V2_ONLY | |
| get_user_status | V2_ONLY | |
| get_user_statistics | V2_ONLY | |
| get_focus_heatmap | V2_ONLY | |
| get_focus_by_tag | V2_ONLY | |

---

## Part 8: High-Level Client

**Location:** `client/client.py` (1070 lines)

### 8.1 TickTickClient Class

Purpose: Friendly, documented user-facing API

Entry Points:
```python
# From environment variables
async with TickTickClient.from_settings() as client:
    tasks = await client.get_all_tasks()

# With explicit credentials
async with TickTickClient(
    client_id="...",
    client_secret="...",
    v1_access_token="...",
    username="...",
    password="...",
) as client:
    tasks = await client.get_all_tasks()
```

### 8.2 Public API Methods (80+ total)

**Lifecycle (4 methods):**
- `connect()` - Authenticate both V1/V2
- `disconnect()` - Cleanup
- `is_connected` - Connection status property
- `inbox_id` - Get inbox project ID

**Tasks (16 methods):**
- `get_all_tasks()`, `get_task()`, `create_task()`, `update_task()`
- `complete_task()`, `delete_task()`, `move_task()`
- `make_subtask()`, `unparent_subtask()`
- `get_completed_tasks()`, `get_abandoned_tasks()`, `get_deleted_tasks()`
- `quick_add()`, `search_tasks()`
- `get_today_tasks()`, `get_overdue_tasks()`, `get_tasks_by_tag()`, `get_tasks_by_priority()`

**Projects (8 methods):**
- `get_all_projects()`, `get_project()`, `get_project_tasks()`
- `create_project()`, `update_project()`, `delete_project()`

**Folders (4 methods):**
- `get_all_folders()`, `create_folder()`, `rename_folder()`, `delete_folder()`

**Tags (7 methods):**
- `get_all_tags()`, `create_tag()`, `update_tag()`, `delete_tag()`
- `rename_tag()`, `merge_tags()`

**Habits (11 methods):**
- `get_all_habits()`, `get_habit()`, `get_habit_sections()`, `get_habit_preferences()`
- `create_habit()`, `update_habit()`, `delete_habit()`
- `checkin_habit()`, `archive_habit()`, `unarchive_habit()`, `get_habit_checkins()`

**User & Analytics (4 methods):**
- `get_profile()`, `get_status()`, `get_statistics()`, `get_preferences()`

**Focus/Pomodoro (2 methods):**
- `get_focus_heatmap()`, `get_focus_by_tag()`

---

## Part 9: Exception Hierarchy

**Location:** `exceptions.py` (271 lines)

### 9.1 Exception Tree

```
TickTickError (base)
├── TickTickAuthenticationError
│   ├── TickTickOAuthError (V1-specific)
│   └── TickTickSessionError (V2-specific, 2FA detection)
├── TickTickAPIError
│   ├── TickTickRateLimitError
│   ├── TickTickNotFoundError
│   ├── TickTickForbiddenError
│   ├── TickTickServerError
│   └── TickTickQuotaExceededError
├── TickTickValidationError
├── TickTickConfigurationError
└── TickTickAPIUnavailableError
```

### 9.2 Exception Details

| Exception | Key Fields |
|-----------|------------|
| TickTickError | `message`, `details: dict` |
| TickTickOAuthError | `oauth_error`, `oauth_error_description` |
| TickTickSessionError | `requires_2fa`, `auth_id` |
| TickTickAPIError | `status_code`, `response_body`, `api_version`, `endpoint` |
| TickTickRateLimitError | `retry_after` (seconds) |
| TickTickNotFoundError | `resource_type`, `resource_id` |
| TickTickValidationError | `field`, `value`, `expected` |
| TickTickConfigurationError | `missing_config` (list of missing env vars) |

---

## Part 10: MCP Server

**Location:** `server.py` (2445 lines)

### 10.1 Overview

- Framework: FastMCP (async MCP server framework)
- Entry Point: `main()` function
- Total Tools: 45

### 10.2 Tool Organization

| Category | Count | Tools |
|----------|-------|-------|
| Task | 13 | create, get, list, update, complete, delete, move, make_subtask, unparent_subtask, completed, abandoned, deleted, search |
| Project | 6 | list, get, create, update, delete |
| Folder | 4 | list, create, rename, delete |
| Tag | 7 | list, create, update, delete, rename, merge |
| Habit | 10 | list, get, sections, create, update, delete, checkin, archive, unarchive, checkins |
| User & Analytics | 6 | profile, status, statistics, preferences, focus_heatmap, focus_by_tag |

### 10.3 Tool Input Models

**Location:** `tools/inputs.py` (917 lines)

- `ResponseFormat` enum: MARKDOWN, JSON
- Input dataclasses with Pydantic validation
- Each tool has a corresponding input model

### 10.4 Tool Output Formatting

**Location:** `tools/formatting.py` (431 lines)

| Format | Purpose | Character Limit |
|--------|---------|-----------------|
| Markdown | Human-readable (headers, lists, tables, emoji) | 25,000 chars |
| JSON | Machine-readable (full model serialization) | None |

---

## Part 11: CLI

### 11.1 Main CLI

**Location:** `cli.py` (261 lines)

**Entry Point:** `ticktick-sdk` command

Commands:
```
ticktick-sdk              # Default: run server
ticktick-sdk server       # Explicit: run server
ticktick-sdk auth         # OAuth2 flow (browser)
ticktick-sdk auth --manual # OAuth2 flow (SSH-friendly)
ticktick-sdk --version    # Show version
ticktick-sdk --help       # Show help
```

### 11.2 Auth CLI

**Location:** `auth_cli.py` (575 lines)

**Auto Mode (Browser):**
1. Generates authorization URL
2. Opens browser → User approves
3. Browser redirects to callback
4. CLI captures code
5. Exchanges for token
6. Displays token for `.env`

**Manual Mode (SSH):**
1. Displays authorization URL
2. User copies URL to browser elsewhere
3. User copies authorization code back
4. Exchange for token

---

## Part 12: Constants

**Location:** `constants.py`

### 12.1 API Configuration

| Constant | Value |
|----------|-------|
| `TICKTICK_API_BASE_V1` | `https://api.ticktick.com/open/v1` |
| `TICKTICK_API_BASE_V2` | `https://api.ticktick.com/api/v2` |
| `TICKTICK_OAUTH_BASE` | `https://ticktick.com/oauth` |
| `DEFAULT_TIMEOUT` | 30.0 |
| `OAUTH_SCOPES` | `["tasks:read", "tasks:write"]` |

### 12.2 Enumerations

| Enum | Values |
|------|--------|
| TaskStatus | ABANDONED=-1, ACTIVE=0, COMPLETED_ALT=1, COMPLETED=2 |
| TaskPriority | NONE=0, LOW=1, MEDIUM=3, HIGH=5 |
| TaskKind | TEXT, NOTE, CHECKLIST |
| ProjectKind | TASK, NOTE |
| ViewMode | LIST, KANBAN, TIMELINE |
| RepeatFrom | DUE_DATE=0, COMPLETED_DATE=1, UNKNOWN=2 |
| Permission | READ, WRITE, COMMENT |

---

## Part 13: Testing Structure

**Location:** `tests/` (11 modules, 300+ test cases)

### 13.1 Test Modules

| Module | Purpose |
|--------|---------|
| `conftest.py` | Pytest fixtures & mock clients |
| `test_client_tasks.py` | Task CRUD operations |
| `test_client_projects.py` | Project management |
| `test_client_tags.py` | Tag operations |
| `test_client_habits.py` | Habit management |
| `test_client_folders.py` | Folder operations |
| `test_client_user.py` | User/stats |
| `test_client_focus_sync.py` | Focus/sync |
| `test_client_lifecycle.py` | Connection lifecycle |
| `test_client_errors.py` | Error handling |

### 13.2 Testing Patterns

- Framework: pytest with asyncio plugin
- Mocking: `respx` for httpx, `freezegun` for time
- Markers: `@pytest.mark.asyncio`, `@pytest.mark.mock_only`, `@pytest.mark.live_only`
- Live tests require `--live` flag

---

## Part 14: Build & Distribution

### 14.1 Package Configuration

**Build System:** hatchling

**Dependencies:**
| Type | Package | Minimum Version |
|------|---------|-----------------|
| Core | httpx | 0.27.0 |
| Core | pydantic | 2.0.0 |
| Core | pydantic-settings | 2.0.0 |
| Core | mcp | 1.0.0 |
| Dev | pytest | - |
| Dev | pytest-asyncio | - |
| Dev | respx | - |
| Dev | freezegun | - |
| Dev | mypy | - |
| Dev | ruff | - |

**Entry Points:**
```toml
[project.scripts]
ticktick-sdk = "ticktick_sdk.cli:cli_main"
```

### 14.2 Code Quality

| Tool | Configuration |
|------|--------------|
| mypy | strict=true, warn_return_any=true, disallow_untyped_defs=true |
| ruff | target-version="py311", line-length=100 |

---

## Part 15: Key Architectural Decisions

### 15.1 V2-First Design

1. **Feature Completeness:** V1 API lacks tags, habits, focus tracking
2. **Unified Interface:** Single model set regardless of API version
3. **Fallback Safety:** Can fall back to V1 for certain operations
4. **Future-Proof:** V2 features automatically available

### 15.2 Unified Model Strategy

- Single source of truth for data structures
- API agnostic - users don't care which version provides data
- Alias support for field name differences (camelCase ↔ snake_case)
- Conversion methods `to_v1_dict()`, `to_v2_dict()`

### 15.3 Async-First Architecture

- All I/O methods are `async`
- Context manager lifecycle management
- Natural for MCP server (concurrent tool requests)
- No sync wrapper (users use `asyncio.run()`)

### 15.4 Error Handling Philosophy

- Semantic exceptions (NotFoundError, AuthenticationError, etc.)
- Detailed error info in `details` dict
- Batch operation error checking via `_check_batch_response_errors()`

---

## Part 16: TickTick API Quirks

The SDK handles these known API quirks:

| Quirk | Description | SDK Handling |
|-------|-------------|--------------|
| Recurrence requires start_date | Cannot set repeat_flag without start_date | Documented in docstrings |
| Subtasks require separate call | `parent_id` in create_task is ignored | `make_subtask()` method |
| Soft delete | Tasks moved to trash (deleted=1) | Still accessible via get_task |
| Date clearing | Must clear both start_date and due_date together | Documented |
| Tag order not preserved | API doesn't guarantee tag order | SDK doesn't rely on ordering |
| Inbox is special | Cannot be deleted | ID cached from get_status() |

---

## Part 17: Key Files by Importance

| File | Lines | Purpose |
|------|-------|---------|
| server.py | 2,445 | MCP server with 45 tools |
| unified/api.py | 1,968 | Core routing and unification |
| api/v2/client.py | 1,521 | V2 API client implementation |
| client/client.py | 1,070 | High-level user API |
| tools/inputs.py | 917 | MCP tool input models |
| api/v2/types.py | 798 | V2 request/response types |
| auth_cli.py | 575 | OAuth2 CLI flow |
| api/v1/client.py | 531 | V1 API client |
| api/base.py | 469 | Base HTTP client |
| tools/formatting.py | 431 | Response formatting |
| models/task.py | 339 | Task model |
| unified/router.py | 321 | Operation routing table |
| models/habit.py | 285 | Habit models |
| models/project.py | 273 | Project models |
| exceptions.py | 271 | Exception hierarchy |
| settings.py | 268 | Configuration management |
| cli.py | 261 | Command-line interface |

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Python files | 32 source + 11 tests |
| Total lines of code | 14,585 |
| Largest file | server.py (2,445 lines) |
| Number of models | 13 unified models |
| Number of exceptions | 12 exception types |
| Number of MCP tools | 45 tools |
| Test count | 300+ test cases |
| Documentation lines | 1,000+ in README |
| Public API methods | 80+ in TickTickClient |
| API versions supported | 2 (V1 OAuth2, V2 session) |
| Python versions | 3.11, 3.12, 3.13 |
| Dependencies | 4 core, 6+ dev |
| PyPI downloads | 3,000+ |
