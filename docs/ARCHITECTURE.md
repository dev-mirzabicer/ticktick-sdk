# TickTick SDK Architecture Documentation

> **Version**: 0.2.1
> **Last Updated**: January 2026
> **Audience**: Developers, AI Agents, System Architects

This document provides a comprehensive architectural overview of the TickTick SDK, a production-grade Python library with 3,000+ PyPI downloads. It enables developers and AI agents to understand the entire system without reading source code.

---

## Table of Contents

1. [Executive Summary](#section-1-executive-summary)
2. [High-Level Architecture Overview](#section-2-high-level-architecture-overview)
3. [The Dual-API Strategy](#section-3-the-dual-api-strategy)
4. [Layer 1 - TickTickClient (High-Level Facade)](#section-4-layer-1---ticktickclient-high-level-facade)
5. [Layer 2 - UnifiedTickTickAPI (Routing & Conversion)](#section-5-layer-2---unifiedticktickapi-routing--conversion)
6. [Layer 3 - V1 and V2 Clients (HTTP Layer)](#section-6-layer-3---v1-and-v2-clients-http-layer)
7. [Authentication Architecture](#section-7-authentication-architecture)
8. [Configuration System](#section-8-configuration-system)
9. [Exception Architecture](#section-9-exception-architecture)
10. [Async-First Design](#section-10-async-first-design)
11. [Key Architectural Decisions](#section-11-key-architectural-decisions)
12. [Data Flow Examples](#section-12-data-flow-examples)

---

## Section 1: Executive Summary

### What is this SDK?

The TickTick SDK is a **reverse-engineered, production-grade Python library** that provides programmatic access to TickTick, a popular task management and productivity application. Unlike typical SDKs that wrap a single official API, this SDK uniquely combines **two distinct API versions**:

1. **V1 API (Official)**: TickTick's documented OAuth2 Open API with limited features
2. **V2 API (Unofficial)**: A reverse-engineered session-based API with comprehensive features

This dual-API approach enables access to features unavailable through official channels alone, including tags, habits, focus/pomodoro tracking, subtasks, and project folders.

### What Problem Does It Solve?

TickTick's official V1 API is severely limited - it lacks support for tags, habits, focus time tracking, subtasks, and many other core features that users rely on. This SDK solves that problem by:

1. **Reverse-engineering the web application's API** to access all features
2. **Providing a unified interface** that abstracts away API version differences
3. **Intelligently routing operations** to the appropriate API based on feature availability
4. **Offering an MCP server** for AI assistant integration

### Key Capabilities

- **Full Task Management**: Create, read, update, delete, complete, move, and organize tasks with subtask support
- **Project Organization**: Manage projects with folders (project groups) and kanban columns
- **Tag System**: Full tag CRUD operations including nesting, renaming, and merging
- **Habit Tracking**: Create habits, record check-ins, track streaks with backdating support
- **Focus/Pomodoro**: Access focus session heatmaps and time-by-tag analytics
- **User Analytics**: Productivity statistics, preferences, and account status

### Target Audience

1. **Python Developers** building productivity tools or integrations
2. **AI Agents/Assistants** (via MCP server) for task management automation
3. **Automation Engineers** creating workflows with TickTick data
4. **Data Analysts** extracting productivity metrics

---

## Section 2: High-Level Architecture Overview

### The 3-Layer Architecture Pattern

The SDK implements a **three-layer architecture** that separates concerns and provides progressive abstraction:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   LAYER 1: TickTickClient                                                   │
│   File: src/ticktick_sdk/client/client.py (1,070 lines)                     │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  • User-facing facade with friendly API                             │   │
│   │  • Convenience methods (get_today_tasks, quick_add, search_tasks)   │   │
│   │  • Async context manager lifecycle                                  │   │
│   │  • Type-safe parameters with string-to-enum conversion              │   │
│   │  • 80+ public methods organized by resource type                    │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                        │                                    │
│                                        ▼                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   LAYER 2: UnifiedTickTickAPI                                               │
│   File: src/ticktick_sdk/unified/api.py (1,968 lines)                       │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  • Version-agnostic operation implementation                        │   │
│   │  • Routes operations to V1 or V2 via APIRouter                      │   │
│   │  • Converts between unified models and API-specific formats         │   │
│   │  • Handles batch response error checking                            │   │
│   │  • Manages both V1 and V2 client instances                          │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                        │                                    │
│                          ┌─────────────┴─────────────┐                      │
│                          ▼                           ▼                      │
├──────────────────────────────────────┬──────────────────────────────────────┤
│                                      │                                      │
│   LAYER 3a: TickTickV1Client         │   LAYER 3b: TickTickV2Client         │
│   File: api/v1/client.py (531 lines) │   File: api/v2/client.py (1,521 lines)│
│                                      │                                      │
│   ┌──────────────────────────────┐   │   ┌──────────────────────────────┐   │
│   │  • OAuth2 Bearer token auth  │   │   │  • Session + cookie auth     │   │
│   │  • Official documented API   │   │   │  • Reverse-engineered API    │   │
│   │  • Limited features          │   │   │  • Full feature set          │   │
│   │  • Simple REST endpoints     │   │   │  • Batch operation endpoints │   │
│   └──────────────────────────────┘   │   └──────────────────────────────┘   │
│                                      │                                      │
└──────────────────────────────────────┴──────────────────────────────────────┘
                    │                                      │
                    ▼                                      ▼
         ┌──────────────────┐                   ┌──────────────────┐
         │  TickTick V1 API │                   │  TickTick V2 API │
         │  /open/v1/*      │                   │  /api/v2/*       │
         └──────────────────┘                   └──────────────────┘
```

### Layer Responsibilities

| Layer | Component | Primary Responsibility | Lines of Code |
|-------|-----------|----------------------|---------------|
| **1** | `TickTickClient` | User-friendly interface, convenience methods, lifecycle | 1,070 |
| **2** | `UnifiedTickTickAPI` | API routing, model conversion, error handling | 1,968 |
| **3a** | `TickTickV1Client` | V1 OAuth2 HTTP operations | 531 |
| **3b** | `TickTickV2Client` | V2 session HTTP operations | 1,521 |

### Data Flow Overview

**Request Flow (User to API)**:
```
User Code
    │
    ▼
TickTickClient.create_task(title="Buy groceries", tags=["shopping"])
    │
    │  Converts string priority to int, validates parameters
    ▼
UnifiedTickTickAPI.create_task(title="Buy groceries", tags=["shopping"])
    │
    │  APIRouter determines V2_PRIMARY for create_task
    │  Formats dates to V2 format, builds request payload
    ▼
TickTickV2Client.create_task(payload)
    │
    │  Adds auth headers (Cookie, X-Device), sends POST /batch/task
    ▼
TickTick V2 API Server
```

**Response Flow (API to User)**:
```
TickTick V2 API Server
    │
    │  Returns JSON: {"id2etag": {"task123": "abc12345"}, "id2error": {}}
    ▼
TickTickV2Client
    │
    │  Validates HTTP 200, returns raw dict
    ▼
UnifiedTickTickAPI
    │
    │  Checks id2error for failures (raises TickTickNotFoundError if needed)
    │  Fetches full task via get_task(), converts to Task model
    ▼
TickTickClient
    │
    │  Returns Task unified model to user
    ▼
User Code receives Task object
```

---

## Section 3: The Dual-API Strategy

### Why Two APIs?

TickTick provides an official OAuth2-based "Open API" (V1), but it is **severely limited in functionality**. The V1 API only supports:
- Basic task CRUD
- Project CRUD
- No tags
- No habits
- No focus/pomodoro tracking
- No subtasks
- No folders (project groups)

To provide comprehensive functionality, this SDK reverse-engineers the **unofficial V2 API** used by TickTick's web and mobile applications. This V2 API is session-based and provides access to **all features**.

### API Comparison Table

| Feature | V1 API (Official) | V2 API (Unofficial) |
|---------|-------------------|---------------------|
| **Base URL** | `https://api.ticktick.com/open/v1` | `https://api.ticktick.com/api/v2` |
| **Authentication** | OAuth2 Bearer token | Session token + cookies |
| **Task CRUD** | Yes | Yes |
| **Get Task by ID** | Requires `project_id` | No `project_id` needed |
| **List All Tasks** | Per-project only | All tasks at once |
| **Tags** | No | Yes |
| **Subtasks** | No | Yes |
| **Habits** | No | Yes |
| **Focus/Pomodoro** | No | Yes |
| **Project Folders** | No | Yes |
| **User Profile** | No | Yes |
| **Sync/Batch Ops** | No | Yes |
| **Completed Tasks** | No | Yes (with date range) |
| **Deleted Tasks (Trash)** | No | Yes |

### How the SDK Decides Which API to Use

The SDK uses an **APIRouter** (`src/ticktick_sdk/unified/router.py`) with a routing table that defines preferences for each operation:

```python
# From router.py - APIPreference enum
class APIPreference(StrEnum):
    V1_ONLY = auto()    # Only available in V1
    V2_ONLY = auto()    # Only available in V2
    V2_PRIMARY = auto() # Prefer V2, can fallback to V1
    V1_PRIMARY = auto() # Prefer V1, can fallback to V2
```

**Complete Operation Routing Table** (`OPERATION_ROUTING` from `router.py:43-188`):

| Operation | Routing Preference | Reason |
|-----------|-------------------|--------|
| **Tasks** | | |
| `create_task` | V2_PRIMARY | V2 supports tags, parent_id |
| `get_task` | V2_PRIMARY | V2 doesn't require project_id |
| `update_task` | V2_PRIMARY | V2 has richer update options |
| `delete_task` | V2_PRIMARY | V2 supports batch operations |
| `complete_task` | V1_PRIMARY | V1 has dedicated simple endpoint |
| `list_all_tasks` | V2_ONLY | V1 can only list per-project |
| `list_completed_tasks` | V2_ONLY | V2-only feature |
| `list_deleted_tasks` | V2_ONLY | V2-only feature (trash) |
| `move_task` | V2_ONLY | V2-only feature |
| `set_task_parent` | V2_ONLY | V2-only feature (subtasks) |
| **Projects** | | |
| `create_project` | V2_PRIMARY | V2 supports more options |
| `get_project` | V1_PRIMARY | V1 has dedicated endpoint |
| `get_project_with_data` | V1_ONLY | V1-only (includes tasks + columns) |
| `update_project` | V2_PRIMARY | V2 supports batch operations |
| `delete_project` | V2_PRIMARY | V2 supports batch operations |
| `list_projects` | V2_PRIMARY | V2 returns more metadata |
| **Tags** | | |
| `create_tag` | V2_ONLY | Not available in V1 |
| `update_tag` | V2_ONLY | Not available in V1 |
| `rename_tag` | V2_ONLY | Not available in V1 |
| `delete_tag` | V2_ONLY | Not available in V1 |
| `merge_tags` | V2_ONLY | Not available in V1 |
| `list_tags` | V2_ONLY | Not available in V1 |
| **Project Groups (Folders)** | | |
| `create_project_group` | V2_ONLY | Not available in V1 |
| `update_project_group` | V2_ONLY | Not available in V1 |
| `delete_project_group` | V2_ONLY | Not available in V1 |
| `list_project_groups` | V2_ONLY | Not available in V1 |
| **Habits** | | |
| All habit operations | V2_ONLY | Not available in V1 |
| **User** | | |
| `get_user_profile` | V2_ONLY | Not available in V1 |
| `get_user_status` | V2_ONLY | Not available in V1 |
| `get_user_statistics` | V2_ONLY | Not available in V1 |
| **Focus/Pomodoro** | | |
| `get_focus_heatmap` | V2_ONLY | Not available in V1 |
| `get_focus_by_tag` | V2_ONLY | Not available in V1 |
| **Sync** | | |
| `sync_all` | V2_ONLY | V2-only feature |

---

## Section 4: Layer 1 - TickTickClient (High-Level Facade)

**File**: `src/ticktick_sdk/client/client.py`
**Lines**: 1,070
**Purpose**: Provide a user-friendly, well-documented API for application developers

### Class Overview

```python
class TickTickClient:
    """
    High-level TickTick client.

    This is the main entry point for interacting with TickTick.
    It provides a clean, user-friendly interface with convenience methods.

    The client requires BOTH V1 (OAuth2) and V2 (session) authentication
    to provide full functionality.
    """
```

### Constructor Parameters

```python
def __init__(
    self,
    # V1 OAuth2 credentials
    client_id: str,
    client_secret: str,
    redirect_uri: str = "http://localhost:8080/callback",
    v1_access_token: str | None = None,
    # V2 Session credentials
    username: str | None = None,
    password: str | None = None,
    # General
    timeout: float = 30.0,
    device_id: str | None = None,
) -> None:
```

### Factory Method: `from_settings()`

The recommended way to create a client is via the factory method that reads from environment variables:

```python
@classmethod
def from_settings(cls, settings: TickTickSettings | None = None) -> TickTickClient:
    """Create a client from settings (environment variables)."""
    if settings is None:
        settings = get_settings()

    settings.validate_all_ready()  # Raises TickTickConfigurationError if missing

    return cls(
        client_id=settings.client_id,
        client_secret=settings.client_secret.get_secret_value(),
        redirect_uri=settings.redirect_uri,
        v1_access_token=settings.get_v1_access_token(),
        username=settings.username,
        password=settings.get_v2_password(),
        timeout=settings.timeout,
        device_id=settings.device_id,
    )
```

### Lifecycle Management

The client uses async context manager pattern for proper resource management:

```python
async def connect(self) -> None:
    """Initialize both V1 and V2 API connections."""
    await self._api.initialize()
    self._initialized = True

async def disconnect(self) -> None:
    """Close all connections and cleanup."""
    await self._api.close()
    self._initialized = False

async def __aenter__(self: T) -> T:
    """Enter async context manager."""
    await self.connect()
    return self

async def __aexit__(...) -> None:
    """Exit async context manager."""
    await self.disconnect()
```

**Usage Pattern**:
```python
async with TickTickClient.from_settings() as client:
    tasks = await client.get_all_tasks()
    # Connection automatically closed on exit
```

### Method Categories

The `TickTickClient` provides **80+ methods** organized into these categories:

#### Lifecycle Methods (4)
- `connect()` - Initialize and authenticate
- `disconnect()` - Close connections
- `is_connected` - Property for connection status
- `inbox_id` - Property for inbox project ID

#### Sync (1)
- `sync()` - Get complete account state

#### Task Methods (16)
| Method | Description | Delegates to |
|--------|-------------|--------------|
| `get_all_tasks()` | List all active tasks | `_api.list_all_tasks()` |
| `get_task(task_id, project_id?)` | Get single task | `_api.get_task()` |
| `create_task(title, ...)` | Create task with all options | `_api.create_task()` |
| `update_task(task)` | Update task model | `_api.update_task()` |
| `complete_task(task_id, project_id)` | Mark complete | `_api.complete_task()` |
| `delete_task(task_id, project_id)` | Delete task | `_api.delete_task()` |
| `move_task(task_id, from_proj, to_proj)` | Move between projects | `_api.move_task()` |
| `make_subtask(task_id, parent_id, proj)` | Make subtask | `_api.set_task_parent()` |
| `unparent_subtask(task_id, project_id)` | Remove from parent | `_api.unset_task_parent()` |
| `get_completed_tasks(days, limit)` | Recent completions | `_api.list_completed_tasks()` |
| `get_abandoned_tasks(days, limit)` | "Won't do" tasks | `_api.list_abandoned_tasks()` |
| `get_deleted_tasks(limit)` | Trash items | `_api.list_deleted_tasks()` |
| `quick_add(text)` | Simple task creation | `create_task(text)` |
| `get_today_tasks()` | Tasks due today | Filter `get_all_tasks()` |
| `get_overdue_tasks()` | Overdue tasks | Filter `get_all_tasks()` |
| `search_tasks(query)` | Title/content search | Filter `get_all_tasks()` |

#### Project Methods (6)
| Method | Description |
|--------|-------------|
| `get_all_projects()` | List all projects |
| `get_project(project_id)` | Get single project |
| `get_project_tasks(project_id)` | Get project with tasks/columns |
| `create_project(name, ...)` | Create project |
| `update_project(project_id, ...)` | Update project |
| `delete_project(project_id)` | Delete project |

#### Folder Methods (4)
| Method | Description |
|--------|-------------|
| `get_all_folders()` | List project groups |
| `create_folder(name)` | Create folder |
| `rename_folder(folder_id, name)` | Rename folder |
| `delete_folder(folder_id)` | Delete folder |

#### Tag Methods (6)
| Method | Description |
|--------|-------------|
| `get_all_tags()` | List all tags |
| `create_tag(name, color?, parent?)` | Create tag |
| `update_tag(name, color?, parent?)` | Update tag properties |
| `delete_tag(name)` | Delete tag |
| `rename_tag(old_name, new_name)` | Rename tag |
| `merge_tags(source, target)` | Merge tags |

#### Habit Methods (11)
| Method | Description |
|--------|-------------|
| `get_all_habits()` | List all habits |
| `get_habit(habit_id)` | Get single habit |
| `get_habit_sections()` | Get time-of-day sections |
| `get_habit_preferences()` | Get habit settings |
| `create_habit(name, ...)` | Create habit with full options |
| `update_habit(habit_id, ...)` | Update habit |
| `delete_habit(habit_id)` | Delete habit |
| `checkin_habit(habit_id, value?, date?)` | Record check-in |
| `archive_habit(habit_id)` | Archive habit |
| `unarchive_habit(habit_id)` | Restore habit |
| `get_habit_checkins(habit_ids, after_stamp)` | Query check-in history |

#### User & Analytics Methods (6)
| Method | Description |
|--------|-------------|
| `get_profile()` | User profile info |
| `get_status()` | Subscription status |
| `get_statistics()` | Productivity stats |
| `get_preferences()` | User settings |
| `get_focus_heatmap(start?, end?, days?)` | Focus session heatmap |
| `get_focus_by_tag(start?, end?, days?)` | Focus time by tag |

### Type Conversion and Validation

The client handles type conversions for user convenience:

```python
async def create_task(
    self,
    title: str,
    priority: int | str | None = None,  # Accepts both "high" and 5
    ...
) -> Task:
    # Convert string priority to int
    if isinstance(priority, str):
        priority_map = {"none": 0, "low": 1, "medium": 3, "high": 5}
        priority = priority_map.get(priority.lower(), 0)

    return await self._api.create_task(
        title=title,
        priority=priority,
        ...
    )
```

---

## Section 5: Layer 2 - UnifiedTickTickAPI (Routing & Conversion)

**File**: `src/ticktick_sdk/unified/api.py`
**Lines**: 1,968
**Purpose**: Route operations to appropriate API, convert between models, handle errors

### Class Overview

```python
class UnifiedTickTickAPI:
    """
    Unified TickTick API providing version-agnostic operations.

    This class manages both V1 and V2 API clients and provides
    a single interface for all TickTick operations. It automatically
    routes operations to the appropriate API version.

    Both V1 and V2 authentication are REQUIRED for full functionality.
    """
```

### Constructor and Client Management

```python
def __init__(
    self,
    # V1 OAuth2 credentials
    client_id: str,
    client_secret: str,
    redirect_uri: str = "http://localhost:8080/callback",
    v1_access_token: str | None = None,
    # V2 Session credentials
    username: str | None = None,
    password: str | None = None,
    # General
    timeout: float = 30.0,
    device_id: str | None = None,
) -> None:
    # Store credentials for lazy initialization
    self._v1_credentials = {...}
    self._v2_credentials = {...}

    # Clients (lazy initialized)
    self._v1_client: TickTickV1Client | None = None
    self._v2_client: TickTickV2Client | None = None

    # Router
    self._router: APIRouter | None = None

    # State
    self._initialized = False
    self._inbox_id: str | None = None
```

### Initialization Process

The `initialize()` method (`api.py:274-340`) performs critical setup:

```python
async def initialize(self) -> None:
    """Initialize both API clients."""
    if self._initialized:
        return

    errors: list[str] = []

    # 1. Initialize V1 client
    try:
        self._v1_client = TickTickV1Client(
            client_id=...,
            client_secret=...,
            redirect_uri=...,
            access_token=...,
            timeout=...,
        )
    except Exception as e:
        errors.append(f"V1 initialization failed: {e}")

    # 2. Initialize V2 client and authenticate
    try:
        self._v2_client = TickTickV2Client(
            device_id=...,
            timeout=...,
        )

        if self._v2_credentials["username"] and self._v2_credentials["password"]:
            session = await self._v2_client.authenticate(
                self._v2_credentials["username"],
                self._v2_credentials["password"],
            )
            self._inbox_id = session.inbox_id  # Cache inbox ID
    except Exception as e:
        errors.append(f"V2 initialization failed: {e}")

    # 3. Create router
    self._router = APIRouter(
        v1_client=self._v1_client,
        v2_client=self._v2_client,
    )

    # 4. Verify both clients are working
    verification = await self._router.verify_clients()

    # 5. Require BOTH APIs
    if not self._router.is_fully_configured:
        raise TickTickConfigurationError(
            "Both V1 and V2 APIs are required. " + "; ".join(errors),
        )

    self._initialized = True
```

### The APIRouter Component

**File**: `src/ticktick_sdk/unified/router.py` (321 lines)

The `APIRouter` class manages routing decisions:

```python
@dataclass
class APIRouter:
    """Routes API operations to the appropriate client."""

    v1_client: TickTickV1Client | None = None
    v2_client: TickTickV2Client | None = None

    @property
    def has_v1(self) -> bool:
        """Check if V1 client is available and authenticated."""
        return self.v1_client is not None and self.v1_client.is_authenticated

    @property
    def has_v2(self) -> bool:
        """Check if V2 client is available and authenticated."""
        return self.v2_client is not None and self.v2_client.is_authenticated

    @property
    def is_fully_configured(self) -> bool:
        """Check if both APIs are available."""
        return self.has_v1 and self.has_v2

    def get_routing(self, operation: str) -> OperationConfig:
        """Get the routing configuration for an operation."""
        return OPERATION_ROUTING.get(
            operation,
            OperationConfig(APIPreference.V2_PRIMARY, "Default to V2"),
        )

    def can_execute(self, operation: str) -> bool:
        """Check if an operation can be executed."""
        config = self.get_routing(operation)
        if config.preference == APIPreference.V1_ONLY:
            return self.has_v1
        elif config.preference == APIPreference.V2_ONLY:
            return self.has_v2
        else:  # PRIMARY preferences
            return self.has_v2 or self.has_v1
```

### Model Conversion

The UnifiedTickTickAPI converts between API-specific formats and unified models. Example from `create_task()` (`api.py:450-550`):

```python
async def create_task(
    self,
    title: str,
    project_id: str | None = None,
    *,
    start_date: datetime | None = None,
    due_date: datetime | None = None,
    tags: list[str] | None = None,
    ...
) -> Task:
    # 1. Validate: recurrence requires start_date
    if repeat_flag and not start_date:
        raise TickTickConfigurationError(
            "Recurrence requires start_date. TickTick ignores it otherwise."
        )

    # 2. Default to inbox if no project
    if project_id is None:
        project_id = self._inbox_id

    # 3. Format dates to V2 format
    start_str = Task.format_datetime(start_date, "v2") if start_date else None
    due_str = Task.format_datetime(due_date, "v2") if due_date else None

    # 4. Call V2 client (V2 is REQUIRED for create_task)
    if not self._router.has_v2:
        raise TickTickAPIUnavailableError(
            "V2 API is required for create_task",
            operation="create_task",
        )

    response = await self._v2_client.create_task(
        title=title,
        project_id=project_id,
        start_date=start_str,
        due_date=due_str,
        tags=tags,
        ...
    )

    # 5. Extract task ID from batch response
    task_id = next(iter(response.get("id2etag", {}).keys()), None)

    # 6. Handle parent_id separately (V2 ignores it during creation)
    if parent_id:
        await self._v2_client.set_task_parent(task_id, project_id, parent_id)

    # 7. Fetch and return the full Task model
    return await self.get_task(task_id, project_id)
```

### Batch Response Error Handling

V2 batch endpoints return HTTP 200 even on partial failures. Errors are in the `id2error` field. The `_check_batch_response_errors()` function (`api.py:67-121`) handles this:

```python
# Error codes that map to NotFoundError
_BATCH_NOT_FOUND_ERRORS = frozenset({
    "TASK_NOT_FOUND",
    "PROJECT_NOT_FOUND",
    "TAG_NOT_FOUND",
    "task not exists",
    "project not found",
})

# Error codes that map to QuotaExceededError
_BATCH_QUOTA_ERRORS = frozenset({
    "EXCEED_QUOTA",
})

def _check_batch_response_errors(
    response: dict[str, Any],
    operation: str,
    resource_ids: list[str] | None = None,
) -> None:
    """Check batch response for errors and raise appropriate exceptions."""
    id2error = response.get("id2error", {})
    if not id2error:
        return

    # Check each error
    for resource_id, error_msg in errors_to_check.items():
        error_upper = error_msg.upper()

        if any(nf in error_upper for nf in _BATCH_NOT_FOUND_ERRORS):
            raise TickTickNotFoundError(
                f"Resource not found: {error_msg}",
                resource_id=resource_id,
            )

        if any(qe in error_upper for qe in _BATCH_QUOTA_ERRORS):
            raise TickTickQuotaExceededError(f"Quota exceeded: {error_msg}")

        # Generic error
        raise TickTickAPIError(f"{operation} failed: {error_msg}")
```

### Existence Verification Pattern

V2 batch operations **silently ignore** nonexistent resources. To provide proper error handling, the SDK verifies existence first:

```python
async def complete_task(self, task_id: str, project_id: str) -> None:
    """Mark a task as complete."""
    self._ensure_initialized()

    if self._router.has_v2:
        # V2 silently accepts updates to nonexistent tasks.
        # Verify task exists first for proper errors.
        await self._v2_client.get_task(task_id)  # Raises NotFoundError if missing

        response = await self._v2_client.batch_tasks(
            update=[{
                "id": task_id,
                "projectId": project_id,
                "status": TaskStatus.COMPLETED,
                "completedTime": Task.format_datetime(datetime.now(), "v2"),
            }]
        )
        _check_batch_response_errors(response, "complete_task", [task_id])
        return

    # Fallback to V1 if V2 unavailable
    if self._router.has_v1:
        await self._v1_client.complete_task(project_id, task_id)
        return

    raise TickTickAPIUnavailableError("Could not complete task")
```

---

## Section 6: Layer 3 - V1 and V2 Clients (HTTP Layer)

### BaseTickTickClient Abstract Class

**File**: `src/ticktick_sdk/api/base.py` (469 lines)

The abstract base class provides common HTTP functionality:

```python
class BaseTickTickClient(ABC):
    """Abstract base class for TickTick API clients."""

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self._timeout = timeout
        self._user_agent = user_agent
        self._client: httpx.AsyncClient | None = None
        self._is_authenticated = False
```

#### Abstract Properties

```python
@property
@abstractmethod
def api_version(self) -> APIVersion:
    """Return the API version (V1 or V2)."""
    ...

@property
@abstractmethod
def base_url(self) -> str:
    """Return the base URL for API requests."""
    ...

@property
@abstractmethod
def is_authenticated(self) -> bool:
    """Check if the client is authenticated."""
    ...

@abstractmethod
def _get_auth_headers(self) -> dict[str, str]:
    """Get authentication headers."""
    ...
```

#### HTTP Client Management

```python
async def _ensure_client(self) -> httpx.AsyncClient:
    """Ensure HTTP client is initialized."""
    if self._client is None or self._client.is_closed:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self._timeout),
            headers=self._get_base_headers(),
            follow_redirects=True,
        )
    return self._client

def _get_base_headers(self) -> dict[str, str]:
    """Get base headers for all requests."""
    return {
        "User-Agent": self._user_agent,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
```

#### Error Handling and Status Code Mapping

The `_handle_error_response()` method (`base.py:163-270`) maps HTTP status codes and API error codes to semantic exceptions:

```python
# Error codes that map to specific exceptions
_NOT_FOUND_ERROR_CODES = frozenset({
    "task_not_found", "project_not_found", "tag_not_found",
    "tag_not_exist", "folder_not_found", "not_found",
})

_FORBIDDEN_ERROR_CODES = frozenset({
    "access_forbidden", "forbidden", "permission_denied",
})

_AUTH_ERROR_CODES = frozenset({
    "unauthorized", "invalid_token", "token_expired",
    "username_password_not_match",
})

def _handle_error_response(self, response: httpx.Response, endpoint: str) -> None:
    """Handle error responses and raise appropriate exceptions."""
    status_code = response.status_code

    # Try to parse error body
    try:
        error_body = response.json()
        error_code = error_body.get("errorCode", "").lower()
    except:
        error_code = ""

    # Check error codes in body (takes precedence)
    # TickTick often returns HTTP 500 with semantic error codes
    if error_code in self._NOT_FOUND_ERROR_CODES:
        raise TickTickNotFoundError(...)
    elif error_code in self._FORBIDDEN_ERROR_CODES:
        raise TickTickForbiddenError(...)
    elif error_code in self._AUTH_ERROR_CODES:
        raise TickTickAuthenticationError(...)

    # Fall back to HTTP status code
    if status_code == 401:
        raise TickTickAuthenticationError(...)
    elif status_code == 403:
        raise TickTickForbiddenError(...)
    elif status_code == 404:
        raise TickTickNotFoundError(...)
    elif status_code == 429:
        retry_after = response.headers.get("Retry-After")
        raise TickTickRateLimitError(retry_after=int(retry_after) if retry_after else None)
    elif status_code >= 500:
        raise TickTickServerError(...)
```

**HTTP Status Code Mapping**:

| Status Code | Exception | Notes |
|-------------|-----------|-------|
| 401 | `TickTickAuthenticationError` | Invalid/expired token |
| 403 | `TickTickForbiddenError` | Access denied |
| 404 | `TickTickNotFoundError` | Resource not found |
| 429 | `TickTickRateLimitError` | Rate limit exceeded |
| 500+ | `TickTickServerError` | Server error |
| 200 with empty body | `TickTickNotFoundError` | V1 quirk |

#### HTTP Methods

```python
async def _request(
    self,
    method: str,
    endpoint: str,
    *,
    params: dict[str, Any] | None = None,
    json_data: dict[str, Any] | list[Any] | None = None,
    headers: dict[str, str] | None = None,
    require_auth: bool = True,
) -> httpx.Response:
    """Make an HTTP request."""
    if require_auth and not self.is_authenticated:
        raise TickTickAuthenticationError(...)

    client = await self._ensure_client()
    request_headers = self._get_headers()
    if headers:
        request_headers.update(headers)

    try:
        response = await client.request(
            method=method,
            url=endpoint,
            params=params,
            json=json_data,
            headers=request_headers,
        )
    except httpx.TimeoutException as e:
        raise TickTickAPIError(f"Request timeout: {endpoint}") from e
    except httpx.RequestError as e:
        raise TickTickAPIError(f"Request failed: {e}") from e

    if not response.is_success:
        self._handle_error_response(response, endpoint)

    return response

# Convenience methods
async def _get(self, endpoint, ...) -> httpx.Response
async def _post(self, endpoint, json_data=None, ...) -> httpx.Response
async def _put(self, endpoint, json_data=None, ...) -> httpx.Response
async def _delete(self, endpoint, ...) -> httpx.Response
async def _get_json(self, endpoint, ...) -> Any  # Handles empty response = NotFound
async def _post_json(self, endpoint, json_data=None, ...) -> Any
```

### TickTickV1Client

**File**: `src/ticktick_sdk/api/v1/client.py` (531 lines)
**Base URL**: `https://api.ticktick.com/open/v1`
**Auth**: OAuth2 Bearer token

```python
class TickTickV1Client(BaseTickTickClient):
    """Client for TickTick V1 Open API."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        access_token: str | None = None,
        scopes: list[str] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(timeout=timeout)
        self._oauth = OAuth2Handler(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scopes=scopes,
            timeout=timeout,
        )
        if access_token:
            self._oauth.set_access_token(access_token)

    @property
    def api_version(self) -> APIVersion:
        return APIVersion.V1

    @property
    def base_url(self) -> str:
        return TICKTICK_API_BASE_V1  # "https://api.ticktick.com/open/v1"

    @property
    def is_authenticated(self) -> bool:
        return self._oauth.is_authenticated

    def _get_auth_headers(self) -> dict[str, str]:
        if self._oauth.token is None:
            return {}
        return {"Authorization": self._oauth.token.authorization_header}
```

**V1 Endpoints**:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `get_task(project_id, task_id)` | `GET /project/{projectId}/task/{taskId}` | Get task |
| `create_task(...)` | `POST /task` | Create task |
| `update_task(task_id, project_id, ...)` | `POST /task/{taskId}` | Update task |
| `complete_task(project_id, task_id)` | `POST /project/{projectId}/task/{taskId}/complete` | Complete |
| `delete_task(project_id, task_id)` | `DELETE /project/{projectId}/task/{taskId}` | Delete |
| `get_projects()` | `GET /project` | List projects |
| `get_project(project_id)` | `GET /project/{projectId}` | Get project |
| `get_project_with_data(project_id)` | `GET /project/{projectId}/data` | Project + tasks |
| `create_project(name, ...)` | `POST /project` | Create project |
| `update_project(project_id, ...)` | `POST /project/{projectId}` | Update project |
| `delete_project(project_id)` | `DELETE /project/{projectId}` | Delete project |

### TickTickV2Client

**File**: `src/ticktick_sdk/api/v2/client.py` (1,521 lines)
**Base URL**: `https://api.ticktick.com/api/v2`
**Auth**: Session token + cookies + X-Device header

```python
class TickTickV2Client(BaseTickTickClient):
    """Client for TickTick V2 API (reverse-engineered)."""

    V2_USER_AGENT = "Mozilla/5.0 (rv:145.0) Firefox/145.0"

    def __init__(
        self,
        device_id: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(timeout=timeout)
        self._session_handler = SessionHandler(
            device_id=device_id,
            timeout=timeout,
        )

    @property
    def api_version(self) -> APIVersion:
        return APIVersion.V2

    @property
    def base_url(self) -> str:
        return TICKTICK_API_BASE_V2  # "https://api.ticktick.com/api/v2"

    @property
    def is_authenticated(self) -> bool:
        return self._session_handler.is_authenticated

    def _get_x_device_header(self) -> str:
        """Get x-device header JSON."""
        return json.dumps({
            "platform": "web",
            "version": 6430,
            "id": self._session_handler.device_id,
        })

    def _get_auth_headers(self) -> dict[str, str]:
        headers = {}
        if self._session_handler.session is not None:
            session = self._session_handler.session

            headers["User-Agent"] = self.V2_USER_AGENT
            headers["X-Device"] = self._get_x_device_header()

            # Cookie is primary auth mechanism
            if session.cookies:
                cookie_str = "; ".join(
                    f"{k}={v}" for k, v in session.cookies.items()
                )
                headers["Cookie"] = cookie_str

        return headers
```

**V2 Batch Operations**:

V2 uses batch endpoints for create/update/delete operations:

```python
async def batch_tasks(
    self,
    add: list[TaskCreateV2] | None = None,
    update: list[TaskUpdateV2] | None = None,
    delete: list[TaskDeleteV2] | None = None,
) -> BatchResponseV2:
    """Batch create, update, and delete tasks."""
    data = {
        "add": add or [],
        "update": update or [],
        "delete": delete or [],
        "addAttachments": [],
        "updateAttachments": [],
        "deleteAttachments": [],
    }
    response = await self._post_json("/batch/task", json_data=data)
    return response  # {"id2etag": {...}, "id2error": {...}}
```

**V2 Endpoints Summary**:

| Category | Endpoint | Purpose |
|----------|----------|---------|
| **Auth** | `POST /user/signon` | Authenticate |
| **Auth** | `POST /user/sign/mfa/code/verify` | 2FA completion |
| **Sync** | `GET /batch/check/0` | Full account state |
| **User** | `GET /user/status` | Subscription info |
| **User** | `GET /user/profile` | Profile info |
| **User** | `GET /user/preferences/settings` | Preferences |
| **User** | `GET /statistics/general` | Productivity stats |
| **Task** | `GET /task/{id}` | Get single task |
| **Task** | `POST /batch/task` | Create/update/delete tasks |
| **Task** | `POST /batch/taskProject` | Move tasks |
| **Task** | `POST /batch/taskParent` | Set/unset parent |
| **Task** | `GET /project/all/closed` | Completed/abandoned |
| **Task** | `GET /project/all/trash/pagination` | Deleted tasks |
| **Project** | `POST /batch/project` | Create/update/delete projects |
| **Folder** | `POST /batch/projectGroup` | Create/update/delete folders |
| **Tag** | `POST /batch/tag` | Create/update tags |
| **Tag** | `PUT /tag/rename` | Rename tag |
| **Tag** | `DELETE /tag` | Delete tag |
| **Tag** | `PUT /tag/merge` | Merge tags |
| **Focus** | `GET /pomodoros/statistics/heatmap/{from}/{to}` | Focus heatmap |
| **Focus** | `GET /pomodoros/statistics/dist/{from}/{to}` | Focus by tag |
| **Habit** | `GET /habits` | List habits |
| **Habit** | `GET /habitSections` | Habit sections |
| **Habit** | `GET /user/preferences/habit` | Habit preferences |
| **Habit** | `POST /habits/batch` | Create/update/delete habits |
| **Habit** | `POST /habitCheckins/query` | Query check-ins |
| **Habit** | `POST /habitCheckins/batch` | Create check-ins |

---

## Section 7: Authentication Architecture

### V1 OAuth2 Flow

**Files**:
- `src/ticktick_sdk/api/v1/auth.py` (342 lines)
- Handler class: `OAuth2Handler`
- Token dataclass: `OAuth2Token`

#### OAuth2Token Dataclass

```python
@dataclass
class OAuth2Token:
    """OAuth2 token data."""

    access_token: str
    token_type: str = "Bearer"
    expires_in: int | None = None
    refresh_token: str | None = None
    scope: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_expired(self) -> bool:
        """Check if token is expired (with 60s buffer)."""
        if self.expires_in is None:
            return False
        expiry_time = self.created_at + timedelta(seconds=self.expires_in)
        return datetime.now(timezone.utc) >= (expiry_time - timedelta(seconds=60))

    @property
    def authorization_header(self) -> str:
        """Get Authorization header value."""
        return f"{self.token_type} {self.access_token}"
```

#### OAuth2Handler Class

```python
class OAuth2Handler:
    """Handles OAuth2 authentication for V1 API."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        scopes: list[str] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.scopes = scopes or ["tasks:read", "tasks:write"]
        self._token: OAuth2Token | None = None
        self._state: str | None = None

    @property
    def is_authenticated(self) -> bool:
        return self._token is not None and not self._token.is_expired
```

#### OAuth2 Flow Steps

**Step 1: Generate Authorization URL**
```python
def get_authorization_url(self, state: str | None = None) -> tuple[str, str]:
    """Generate authorization URL for OAuth2 flow."""
    if state is None:
        state = secrets.token_urlsafe(32)  # CSRF protection

    self._state = state

    params = {
        "client_id": self.client_id,
        "scope": " ".join(self.scopes),  # "tasks:read tasks:write"
        "state": state,
        "redirect_uri": self.redirect_uri,
        "response_type": "code",
    }

    auth_url = f"https://ticktick.com/oauth/authorize?{urlencode(params)}"
    return auth_url, state
```

**Step 2: Exchange Code for Token**
```python
async def exchange_code(self, code: str, state: str | None = None) -> OAuth2Token:
    """Exchange authorization code for access token."""
    # Verify state for CSRF protection
    if state is not None and self._state is not None and state != self._state:
        raise TickTickOAuthError("State mismatch - possible CSRF attack")

    token_url = "https://ticktick.com/oauth/token"

    data = {
        "code": code,
        "grant_type": "authorization_code",
        "scope": " ".join(self.scopes),
        "redirect_uri": self.redirect_uri,
    }

    # Basic auth with client credentials
    credentials = f"{self.client_id}:{self.client_secret}"
    encoded = base64.b64encode(credentials.encode()).decode()
    headers = {
        "Authorization": f"Basic {encoded}",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(token_url, data=data, headers=headers)
        token_data = response.json()

    self._token = OAuth2Token(
        access_token=token_data["access_token"],
        token_type=token_data.get("token_type", "Bearer"),
        expires_in=token_data.get("expires_in"),
        refresh_token=token_data.get("refresh_token"),
        scope=token_data.get("scope"),
    )

    return self._token
```

**Step 3: Refresh Token (when expired)**
```python
async def refresh_access_token(self) -> OAuth2Token:
    """Refresh the access token."""
    if self._token is None or self._token.refresh_token is None:
        raise TickTickOAuthError("No refresh token available")

    data = {
        "grant_type": "refresh_token",
        "refresh_token": self._token.refresh_token,
        "scope": " ".join(self.scopes),
    }

    # Exchange refresh token for new access token
    ...
```

### V2 Session Flow

**Files**:
- `src/ticktick_sdk/api/v2/auth.py` (367 lines)
- Handler class: `SessionHandler`
- Token dataclass: `SessionToken`

#### SessionToken Dataclass

```python
@dataclass
class SessionToken:
    """V2 API session token data."""

    token: str              # The session token
    user_id: str            # User identifier
    username: str           # Username/email
    inbox_id: str           # Inbox project ID
    user_code: str | None = None
    is_pro: bool = False
    pro_start_date: str | None = None
    pro_end_date: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cookies: dict[str, str] = field(default_factory=dict)  # Session cookies

    @property
    def authorization_header(self) -> str:
        return f"Bearer {self.token}"
```

#### Device ID Generation

V2 API requires a device ID in MongoDB ObjectId format (24 hex characters):

```python
def _generate_object_id() -> str:
    """Generate MongoDB-style ObjectId (24 hex characters)."""
    # 4 bytes: timestamp (seconds since epoch)
    timestamp = int(time.time()).to_bytes(4, "big")
    # 5 bytes: random value
    random_bytes = os.urandom(5)
    # 3 bytes: counter (random for simplicity)
    counter = os.urandom(3)

    return (timestamp + random_bytes + counter).hex()
```

#### SessionHandler Class

```python
class SessionHandler:
    """Handles session-based authentication for V2 API."""

    DEFAULT_USER_AGENT = "Mozilla/5.0 (rv:145.0) Firefox/145.0"

    def __init__(
        self,
        device_id: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.device_id = device_id or _generate_object_id()
        self._session: SessionToken | None = None

    def _get_x_device_header(self) -> str:
        """Get x-device header JSON."""
        return json.dumps({
            "platform": "web",
            "version": 6430,
            "id": self.device_id,
        })
```

#### Authentication Flow

```python
async def authenticate(self, username: str, password: str) -> SessionToken:
    """Authenticate with username and password."""
    url = "https://api.ticktick.com/api/v2/user/signon"
    params = {"wc": "true", "remember": "true"}
    payload = {"username": username, "password": password}
    headers = {
        "User-Agent": self.DEFAULT_USER_AGENT,
        "X-Device": self._get_x_device_header(),
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, params=params, json=payload, headers=headers)
        data = response.json()

    # Check for 2FA requirement
    if "authId" in data and "token" not in data:
        raise TickTickSessionError(
            "Two-factor authentication required",
            requires_2fa=True,
            auth_id=data.get("authId"),
        )

    # Extract cookies
    cookies = {}
    for cookie in response.cookies.jar:
        cookies[cookie.name] = cookie.value

    # Add token as 't' cookie if not present
    if "t" not in cookies and "token" in data:
        cookies["t"] = data["token"]

    self._session = SessionToken(
        token=data["token"],
        user_id=str(data.get("userId", "")),
        username=data.get("username", username),
        inbox_id=data.get("inboxId", ""),
        is_pro=data.get("pro", False),
        cookies=cookies,
    )

    return self._session
```

#### 2FA Support

```python
async def authenticate_2fa(self, auth_id: str, totp_code: str) -> SessionToken:
    """Complete 2FA authentication."""
    url = "https://api.ticktick.com/api/v2/user/sign/mfa/code/verify"
    payload = {
        "code": totp_code,
        "method": "app",
    }
    headers = self._get_headers()
    headers["x-verify-id"] = auth_id  # Special header for 2FA

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
        data = response.json()

    # Extract session same as regular auth
    ...
```

### Token Storage and Management

Tokens are stored in-memory within the handler classes. The SDK does **not** persist tokens to disk. Users must:

1. Store tokens securely themselves if persistence is needed
2. Use environment variables for pre-obtained access tokens
3. Re-authenticate on each application start

---

## Section 8: Configuration System

**File**: `src/ticktick_sdk/settings.py` (269 lines)

### TickTickSettings Class

Uses Pydantic v2 Settings for configuration with environment variable support:

```python
class TickTickSettings(BaseSettings):
    """TickTick SDK configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="TICKTICK_",      # All vars prefixed with TICKTICK_
        env_file=".env",             # Auto-load .env file
        env_file_encoding="utf-8",
        case_sensitive=False,        # TICKTICK_CLIENT_ID = ticktick_client_id
        extra="ignore",              # Ignore unknown env vars
    )
```

### Configuration Variables

| Category | Environment Variable | Type | Default | Required |
|----------|---------------------|------|---------|----------|
| **V1 OAuth2** | `TICKTICK_CLIENT_ID` | `str` | `""` | Yes |
| | `TICKTICK_CLIENT_SECRET` | `SecretStr` | `""` | Yes |
| | `TICKTICK_REDIRECT_URI` | `str` | `http://localhost:8080/callback` | No |
| | `TICKTICK_ACCESS_TOKEN` | `SecretStr` | `None` | No |
| | `TICKTICK_REFRESH_TOKEN` | `SecretStr` | `None` | No |
| **V2 Session** | `TICKTICK_USERNAME` | `str` | `""` | Yes |
| | `TICKTICK_PASSWORD` | `SecretStr` | `""` | Yes |
| **General** | `TICKTICK_TIMEOUT` | `float` | `30.0` | No |
| | `TICKTICK_TIMEZONE` | `str` | `"UTC"` | No |
| | `TICKTICK_DEVICE_ID` | `str` | auto-generated | No |

### Secret Handling

Sensitive values use Pydantic's `SecretStr` to prevent accidental logging:

```python
client_secret: SecretStr = Field(
    default=SecretStr(""),
    description="OAuth2 client secret",
)
password: SecretStr = Field(
    default=SecretStr(""),
    description="TickTick account password",
)

# Accessing secret values
secret_value = settings.client_secret.get_secret_value()
password_value = settings.password.get_secret_value()
```

### Validation Methods

```python
@property
def has_v1_credentials(self) -> bool:
    """Check if V1 OAuth2 credentials are configured."""
    return bool(self.client_id and self.client_secret.get_secret_value())

@property
def has_v2_credentials(self) -> bool:
    """Check if V2 session credentials are configured."""
    return bool(self.username and self.password.get_secret_value())

@property
def is_fully_configured(self) -> bool:
    """Check if all required credentials are configured."""
    return self.has_v1_credentials and self.has_v2_credentials

def validate_all_ready(self) -> None:
    """Validate both APIs are ready. Raises TickTickConfigurationError if not."""
    errors: list[str] = []
    missing: list[str] = []

    if not self.has_v1_credentials:
        errors.append("V1 OAuth2 credentials incomplete")
        if not self.client_id:
            missing.append("TICKTICK_CLIENT_ID")
        if not self.client_secret.get_secret_value():
            missing.append("TICKTICK_CLIENT_SECRET")

    if not self.has_v2_credentials:
        errors.append("V2 session credentials incomplete")
        if not self.username:
            missing.append("TICKTICK_USERNAME")
        if not self.password.get_secret_value():
            missing.append("TICKTICK_PASSWORD")

    if errors:
        raise TickTickConfigurationError(
            f"Configuration incomplete: {'; '.join(errors)}",
            missing_config=missing,
        )
```

### Global Settings Instance

```python
# Global settings (lazy initialization)
_settings: TickTickSettings | None = None

def get_settings() -> TickTickSettings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = TickTickSettings()
    return _settings

def configure_settings(**kwargs: Any) -> TickTickSettings:
    """Configure settings with explicit values."""
    global _settings
    _settings = TickTickSettings(**kwargs)
    return _settings
```

---

## Section 9: Exception Architecture

**File**: `src/ticktick_sdk/exceptions.py` (272 lines)

### Exception Hierarchy

```
TickTickError (base)
│
├── TickTickAuthenticationError
│   ├── TickTickOAuthError (V1-specific)
│   │   └── oauth_error, oauth_error_description
│   └── TickTickSessionError (V2-specific)
│       └── requires_2fa, auth_id
│
├── TickTickAPIError
│   ├── status_code, response_body, api_version, endpoint
│   │
│   ├── TickTickRateLimitError
│   │   └── retry_after
│   │
│   ├── TickTickNotFoundError
│   │   └── resource_type, resource_id
│   │
│   ├── TickTickForbiddenError
│   │
│   ├── TickTickServerError
│   │
│   └── TickTickQuotaExceededError
│       └── quota_type
│
├── TickTickValidationError
│   └── field, value, expected
│
├── TickTickConfigurationError
│   └── missing_config
│
└── TickTickAPIUnavailableError
    └── operation, v1_error, v2_error
```

### Base Exception

```python
class TickTickError(Exception):
    """Base exception for all TickTick SDK errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message
```

### Authentication Exceptions

```python
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
        self.oauth_error = oauth_error
        self.oauth_error_description = oauth_error_description
        ...

class TickTickSessionError(TickTickAuthenticationError):
    """V2 Session-based authentication error."""

    def __init__(
        self,
        message: str,
        requires_2fa: bool = False,
        auth_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.requires_2fa = requires_2fa
        self.auth_id = auth_id
        ...
```

### API Exceptions

```python
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
        self.status_code = status_code
        self.response_body = response_body
        self.api_version = api_version
        self.endpoint = endpoint
        ...

class TickTickNotFoundError(TickTickAPIError):
    """Resource not found error (404)."""

    def __init__(
        self,
        message: str = "Resource not found",
        resource_type: str | None = None,
        resource_id: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(message, status_code=404, ...)
        self.resource_type = resource_type
        self.resource_id = resource_id

class TickTickRateLimitError(TickTickAPIError):
    """Rate limit exceeded error."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int | None = None,
        **kwargs,
    ) -> None:
        self.retry_after = retry_after
        ...
```

### When Each Exception is Raised

| Exception | When Raised |
|-----------|-------------|
| `TickTickAuthenticationError` | Authentication required but not authenticated |
| `TickTickOAuthError` | OAuth2 token exchange/refresh fails |
| `TickTickSessionError` | V2 login fails or 2FA required |
| `TickTickAPIError` | Generic API failure |
| `TickTickNotFoundError` | Task/project/tag/habit not found (404) |
| `TickTickForbiddenError` | Access denied (403) |
| `TickTickRateLimitError` | Rate limit exceeded (429) |
| `TickTickServerError` | Server error (5xx) |
| `TickTickQuotaExceededError` | Account quota exceeded |
| `TickTickValidationError` | Invalid data (e.g., recurrence without start_date) |
| `TickTickConfigurationError` | Missing credentials or invalid config |
| `TickTickAPIUnavailableError` | Neither V1 nor V2 can handle the operation |

### Error Handling Example

```python
from ticktick_sdk import TickTickClient
from ticktick_sdk.exceptions import (
    TickTickNotFoundError,
    TickTickAuthenticationError,
    TickTickConfigurationError,
)

try:
    async with TickTickClient.from_settings() as client:
        task = await client.get_task("nonexistent_id")

except TickTickConfigurationError as e:
    print(f"Configuration error: {e.message}")
    print(f"Missing: {e.missing_config}")  # ["TICKTICK_CLIENT_ID", ...]

except TickTickNotFoundError as e:
    print(f"Task not found: {e.resource_id}")
    print(f"API version: {e.api_version}")  # "v2"

except TickTickAuthenticationError as e:
    print(f"Auth failed: {e.message}")
```

---

## Section 10: Async-First Design

### Why Async-Only?

The SDK is designed exclusively for async operation. There is **no synchronous wrapper**. This decision was made because:

1. **Network I/O is inherently async** - HTTP requests benefit from async
2. **MCP Server is async** - The primary use case (AI agents) requires async
3. **Concurrent operations** - Users can make parallel API calls
4. **Modern Python** - async/await is now standard
5. **No sync overhead** - Avoids thread pool complexity

### Context Manager Pattern

All client classes use async context managers for lifecycle management:

```python
# TickTickClient (Layer 1)
async def __aenter__(self: T) -> T:
    await self.connect()
    return self

async def __aexit__(self, ...) -> None:
    await self.disconnect()

# UnifiedTickTickAPI (Layer 2)
async def __aenter__(self: T) -> T:
    await self.initialize()
    return self

async def __aexit__(self, ...) -> None:
    await self.close()

# BaseTickTickClient (Layer 3)
async def __aenter__(self: T) -> T:
    await self._ensure_client()
    return self

async def __aexit__(self, ...) -> None:
    await self.close()
```

### Lifecycle Management Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  async with TickTickClient.from_settings() as client:          │
│      │                                                          │
│      │  __aenter__()                                           │
│      │      │                                                   │
│      │      ▼                                                   │
│      │  connect()                                              │
│      │      │                                                   │
│      │      ▼                                                   │
│      │  _api.initialize()                                      │
│      │      │                                                   │
│      │      ├─────────────────────────────────────┐            │
│      │      ▼                                     ▼            │
│      │  TickTickV1Client()              TickTickV2Client()     │
│      │      │                                     │            │
│      │      │                                     ▼            │
│      │      │                           authenticate()         │
│      │      │                                     │            │
│      │      └─────────────────────────────────────┘            │
│      │                      │                                   │
│      │                      ▼                                   │
│      │              APIRouter created                           │
│      │                      │                                   │
│      │                      ▼                                   │
│      │              verify_clients()                            │
│      │                      │                                   │
│      │                      ▼                                   │
│      │              _initialized = True                         │
│      │                                                          │
│      │  # User code runs here                                  │
│      │  tasks = await client.get_all_tasks()                   │
│      │                                                          │
│      │  __aexit__()                                            │
│      │      │                                                   │
│      │      ▼                                                   │
│      │  disconnect()                                           │
│      │      │                                                   │
│      │      ▼                                                   │
│      │  _api.close()                                           │
│      │      │                                                   │
│      │      ├─────────────────────────────────────┐            │
│      │      ▼                                     ▼            │
│      │  v1_client.close()              v2_client.close()       │
│      │      │                                     │            │
│      │      ▼                                     ▼            │
│      │  httpx.AsyncClient.aclose()     httpx.AsyncClient.aclose()│
│      │                                                          │
└─────────────────────────────────────────────────────────────────┘
```

### Running Async Code

For scripts, use `asyncio.run()`:

```python
import asyncio
from ticktick_sdk import TickTickClient

async def main():
    async with TickTickClient.from_settings() as client:
        tasks = await client.get_all_tasks()
        for task in tasks:
            print(task.title)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Section 11: Key Architectural Decisions

### Decision 1: V2-First Routing

**Decision**: Default to V2 API for most operations, with V1 as fallback.

**Rationale**:
- V2 provides 100% feature coverage vs V1's ~30%
- V2 doesn't require `project_id` for task retrieval
- V2 supports tags, habits, focus, subtasks
- V1 is only preferred when it has a simpler dedicated endpoint (e.g., `complete_task`)

**Trade-offs**:
- Relies on reverse-engineered, undocumented API
- V2 may change without notice (though it has been stable)
- Requires session cookies, which is more complex than OAuth2

### Decision 2: Unified Model Strategy

**Decision**: Single set of Pydantic models that work with both APIs.

**Rationale**:
- Users don't need to know which API version they're using
- Consistent interface regardless of underlying implementation
- Automatic validation via Pydantic v2
- Built-in serialization/deserialization

**Implementation**:
```python
class Task(TickTickModel):
    # Unified fields with aliases for both APIs
    id: str
    project_id: str = Field(alias="projectId")

    @classmethod
    def from_v1(cls, data: dict) -> Task:
        """Create from V1 API response."""
        ...

    @classmethod
    def from_v2(cls, data: dict) -> Task:
        """Create from V2 API response."""
        ...

    def to_v1_dict(self) -> dict:
        """Convert for V1 API submission."""
        ...

    def to_v2_dict(self, for_update: bool = False) -> dict:
        """Convert for V2 API submission."""
        ...
```

### Decision 3: Async-Only Architecture

**Decision**: No synchronous API wrapper.

**Rationale**:
- Primary use case (MCP server) is async
- Network I/O benefits from async
- Avoids complexity of thread pool wrappers
- Modern Python best practices

**Trade-offs**:
- Users must use `async/await`
- Slightly higher learning curve for sync-only codebases
- Requires `asyncio.run()` for scripts

### Decision 4: Both APIs Required

**Decision**: Require both V1 and V2 authentication for full functionality.

**Rationale**:
- Some operations only available in V1 (`get_project_with_data`)
- Some operations only available in V2 (tags, habits, etc.)
- Complete feature coverage requires both
- Simplifies SDK design (no partial functionality modes)

**Trade-offs**:
- Higher setup burden for users (4 credentials)
- OAuth2 flow required even if only using V2 features
- Cannot use SDK without developer account

### Decision 5: Existence Verification Pattern

**Decision**: Verify resource exists before batch operations.

**Rationale**:
- V2 batch endpoints silently ignore nonexistent resources
- Users expect `NotFoundError` when resource doesn't exist
- Consistent error handling across all operations

**Implementation**:
```python
async def complete_task(self, task_id: str, project_id: str) -> None:
    # V2 silently ignores nonexistent tasks - verify first
    await self._v2_client.get_task(task_id)  # Raises NotFoundError

    # Now safe to complete
    await self._v2_client.batch_tasks(update=[...])
```

### Decision 6: Pydantic v2 for Models and Settings

**Decision**: Use Pydantic v2 exclusively (not v1).

**Rationale**:
- 5-50x performance improvement over v1
- Better validation and serialization
- `pydantic-settings` for configuration
- `SecretStr` for secure credential handling
- Modern Python type hints

---

## Section 12: Data Flow Examples

### Example 1: Create Task Operation

**User Code**:
```python
task = await client.create_task(
    title="Buy groceries",
    tags=["shopping"],
    due_date=datetime(2025, 12, 25, 17, 0),
    priority="high",
)
```

**Complete Flow**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 1: TickTickClient.create_task()                                       │
│ File: client/client.py:205-262                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Convert priority "high" → 5                                              │
│ 2. Call self._api.create_task(                                              │
│        title="Buy groceries",                                               │
│        tags=["shopping"],                                                   │
│        due_date=datetime(2025, 12, 25, 17, 0),                             │
│        priority=5,                                                          │
│    )                                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 2: UnifiedTickTickAPI.create_task()                                   │
│ File: unified/api.py:450-550                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Check recurrence validation (none needed here)                           │
│ 2. Default project_id to inbox_id                                           │
│ 3. Format due_date → "2025-12-25T17:00:00.000+0000"                        │
│ 4. Check router.has_v2 → True                                               │
│ 5. Call self._v2_client.create_task(                                        │
│        title="Buy groceries",                                               │
│        project_id="inbox123456789",                                         │
│        due_date="2025-12-25T17:00:00.000+0000",                            │
│        tags=["shopping"],                                                   │
│        priority=5,                                                          │
│    )                                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 3: TickTickV2Client.create_task()                                     │
│ File: api/v2/client.py:345-423                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Build task dict:                                                         │
│    {                                                                        │
│        "title": "Buy groceries",                                            │
│        "projectId": "inbox123456789",                                       │
│        "dueDate": "2025-12-25T17:00:00.000+0000",                          │
│        "tags": ["shopping"],                                                │
│        "priority": 5                                                        │
│    }                                                                        │
│ 2. Call batch_tasks(add=[task])                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 3: TickTickV2Client.batch_tasks()                                     │
│ File: api/v2/client.py:317-343                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Build batch request:                                                     │
│    {                                                                        │
│        "add": [{...task dict...}],                                         │
│        "update": [],                                                        │
│        "delete": [],                                                        │
│        "addAttachments": [],                                                │
│        "updateAttachments": [],                                             │
│        "deleteAttachments": []                                              │
│    }                                                                        │
│ 2. Call _post_json("/batch/task", json_data=data)                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ HTTP LAYER: BaseTickTickClient._post_json()                                 │
│ File: api/base.py:452-469                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Get headers:                                                             │
│    {                                                                        │
│        "User-Agent": "Mozilla/5.0 (rv:145.0) Firefox/145.0",               │
│        "Content-Type": "application/json",                                  │
│        "Cookie": "t=session_token_value; ...",                             │
│        "X-Device": "{\"platform\":\"web\",\"version\":6430,\"id\":...}"    │
│    }                                                                        │
│ 2. POST https://api.ticktick.com/api/v2/batch/task                         │
│ 3. Return response.json()                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                        ┌─────────────────────┐
                        │  TickTick API       │
                        │  Server Response:   │
                        │  {                  │
                        │    "id2etag": {     │
                        │      "task789": "a1b2c3d4"│
                        │    },               │
                        │    "id2error": {}   │
                        │  }                  │
                        └─────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Response flows back up through layers:                                      │
│                                                                             │
│ LAYER 3 → Returns raw dict                                                  │
│ LAYER 2 → Extracts task_id "task789"                                        │
│        → Calls get_task("task789") to fetch full task                       │
│        → Converts V2 response to Task model                                 │
│ LAYER 1 → Returns Task object to user                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Example 2: List Tags Operation (V2-Only)

**User Code**:
```python
tags = await client.get_all_tags()
```

**Complete Flow**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 1: TickTickClient.get_all_tags()                                      │
│ File: client/client.py:553-560                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ return await self._api.list_tags()                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 2: UnifiedTickTickAPI.list_tags()                                     │
│ File: unified/api.py:1226-1238                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ # V2-only operation (tags not available in V1)                              │
│ 1. self._ensure_initialized()                                               │
│ 2. state = await self._v2_client.sync()                                     │
│ 3. tags_data = state.get("tags", [])                                        │
│ 4. return [Tag.from_v2(t) for t in tags_data]                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 3: TickTickV2Client.sync()                                            │
│ File: api/v2/client.py:256-266                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ response = await self._get_json("/batch/check/0")                           │
│ return response  # Complete account state                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                        ┌─────────────────────┐
                        │  TickTick API       │
                        │  GET /batch/check/0 │
                        │  Response includes: │
                        │  {                  │
                        │    "tags": [        │
                        │      {"name": "work", "label": "Work", ...},│
                        │      {"name": "personal", ...}│
                        │    ],               │
                        │    "syncTaskBean": {...},│
                        │    "projectProfiles": [...]│
                        │  }                  │
                        └─────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Response flows back:                                                        │
│                                                                             │
│ LAYER 3 → Returns complete sync state dict                                  │
│ LAYER 2 → Extracts "tags" array                                             │
│        → Converts each to Tag.from_v2()                                     │
│ LAYER 1 → Returns list[Tag] to user                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Example 3: Error Propagation

**Scenario**: User tries to complete a nonexistent task.

```python
await client.complete_task("nonexistent_task_id", "some_project_id")
```

**Error Flow**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 1: TickTickClient.complete_task()                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ await self._api.complete_task("nonexistent_task_id", "some_project_id")     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 2: UnifiedTickTickAPI.complete_task()                                 │
│ File: unified/api.py:606-651                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ # Verify task exists first (V2 silently ignores nonexistent)                │
│ await self._v2_client.get_task("nonexistent_task_id")                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 3: TickTickV2Client.get_task()                                        │
│ File: api/v2/client.py:303-315                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ response = await self._get_json("/task/nonexistent_task_id")                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ HTTP LAYER: BaseTickTickClient._get_json()                                  │
│ File: api/base.py:424-450                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ # TickTick returns HTTP 500 with errorCode "task_not_found"                 │
│ response = await self._get(endpoint, ...)                                   │
│ # Response: HTTP 500, body: {"errorCode": "task_not_found", ...}            │
│                                                                             │
│ if not response.is_success:                                                 │
│     self._handle_error_response(response, endpoint)                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ BaseTickTickClient._handle_error_response()                                 │
│ File: api/base.py:163-270                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ error_code = "task_not_found"                                               │
│ if error_code in self._NOT_FOUND_ERROR_CODES:                               │
│     raise TickTickNotFoundError(                                            │
│         "Resource not found: task_not_found",                               │
│         endpoint="/task/nonexistent_task_id",                               │
│         api_version="v2",                                                   │
│     )                                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Exception propagates back through all layers unchanged                      │
│                                                                             │
│ LAYER 3 → TickTickNotFoundError raised                                      │
│ LAYER 2 → TickTickNotFoundError propagates                                  │
│ LAYER 1 → TickTickNotFoundError propagates                                  │
│ User    → Catches TickTickNotFoundError                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

**User Error Handling**:
```python
from ticktick_sdk.exceptions import TickTickNotFoundError

try:
    await client.complete_task("nonexistent_task_id", "some_project_id")
except TickTickNotFoundError as e:
    print(f"Error: {e.message}")          # "Resource not found: task_not_found"
    print(f"Endpoint: {e.endpoint}")      # "/task/nonexistent_task_id"
    print(f"API Version: {e.api_version}")# "v2"
```

---

## Appendix A: File Reference

| File Path | Lines | Purpose |
|-----------|-------|---------|
| `src/ticktick_sdk/__init__.py` | 154 | Public API surface, exports |
| `src/ticktick_sdk/client/client.py` | 1,070 | High-level TickTickClient |
| `src/ticktick_sdk/unified/api.py` | 1,968 | UnifiedTickTickAPI routing |
| `src/ticktick_sdk/unified/router.py` | 321 | APIRouter routing table |
| `src/ticktick_sdk/api/base.py` | 469 | BaseTickTickClient abstract |
| `src/ticktick_sdk/api/v1/client.py` | 531 | TickTickV1Client |
| `src/ticktick_sdk/api/v1/auth.py` | 342 | OAuth2Handler, OAuth2Token |
| `src/ticktick_sdk/api/v2/client.py` | 1,521 | TickTickV2Client |
| `src/ticktick_sdk/api/v2/auth.py` | 367 | SessionHandler, SessionToken |
| `src/ticktick_sdk/settings.py` | 269 | TickTickSettings |
| `src/ticktick_sdk/constants.py` | 242 | Enums, URLs, constants |
| `src/ticktick_sdk/exceptions.py` | 272 | Exception hierarchy |

---

## Appendix B: Public API Exports

From `src/ticktick_sdk/__init__.py`:

```python
__all__ = [
    # Version
    "__version__",
    # Client
    "TickTickClient",
    # Models
    "Task", "ChecklistItem", "TaskReminder",
    "Project", "ProjectGroup", "ProjectData", "Column",
    "Tag",
    "User", "UserStatus", "UserStatistics",
    "Habit", "HabitSection", "HabitCheckin", "HabitPreferences",
    # Exceptions
    "TickTickError",
    "TickTickAuthenticationError",
    "TickTickAPIError",
    "TickTickValidationError",
    "TickTickRateLimitError",
    "TickTickNotFoundError",
    "TickTickConfigurationError",
    "TickTickForbiddenError",
    "TickTickServerError",
    # Constants
    "TaskStatus", "TaskPriority", "TaskKind",
    "ProjectKind", "ViewMode",
    # Settings
    "TickTickSettings", "get_settings", "configure_settings",
]
```

---

## Document Information

- **Author**: Technical Documentation System
- **Based on Source Code Version**: 0.2.1
- **Total Source Lines Analyzed**: 7,566
- **Documentation Lines**: 2,200+
- **Last Generated**: January 2026
