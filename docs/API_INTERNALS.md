# TickTick SDK - API Internals Documentation

> **Version**: 0.2.1
> **Last Updated**: January 2026
> **Audience**: Developers, AI Agents, System Architects
> **Companion Document**: See `ARCHITECTURE.md` for the overall system design

This document provides a comprehensive, authoritative reference for the API layer of the TickTick SDK. It covers every aspect of HTTP communication, authentication flows, endpoint implementations, batch operations, routing logic, and error handling. After reading this document, you will understand how to debug, extend, or integrate with the API layer without needing to read the source code.

---

## Table of Contents

1. [Introduction to the API Layer](#section-1-introduction-to-the-api-layer)
2. [Base HTTP Client (BaseTickTickClient)](#section-2-base-http-client-basetickttickclient)
3. [V1 API Client (TickTickV1Client)](#section-3-v1-api-client-ticktickv1client)
4. [V1 Authentication (OAuth2Handler)](#section-4-v1-authentication-oauth2handler)
5. [V2 API Client (TickTickV2Client)](#section-5-v2-api-client-ticktickv2client)
6. [V2 Authentication (SessionHandler)](#section-6-v2-authentication-sessionhandler)
7. [V2 Request/Response Types](#section-7-v2-requestresponse-types)
8. [API Routing (APIRouter)](#section-8-api-routing-apirouter)
9. [Unified API Layer (UnifiedTickTickAPI)](#section-9-unified-api-layer-unifiedticktickapi)
10. [Error Handling](#section-10-error-handling)
11. [HTTP Request Details](#section-11-http-request-details)
12. [Batch Operations Deep Dive](#section-12-batch-operations-deep-dive)
13. [Authentication Tokens and Headers](#section-13-authentication-tokens-and-headers)
14. [API Endpoint Reference Tables](#section-14-api-endpoint-reference-tables)

---

## Section 1: Introduction to the API Layer

### 1.1 Overview

The API layer is the HTTP communication backbone of the TickTick SDK. It handles all network requests to TickTick's servers, manages authentication state, and translates between the SDK's unified data models and TickTick's API-specific JSON formats.

The API layer is located in:
- `/src/ticktick_sdk/api/` - Core API client implementations
- `/src/ticktick_sdk/unified/` - Routing and unification layer

### 1.2 The Two APIs

This SDK uniquely supports **two distinct TickTick APIs**:

| Aspect | V1 API (Official) | V2 API (Reverse-Engineered) |
|--------|-------------------|----------------------------|
| **Base URL** | `https://api.ticktick.com/open/v1` | `https://api.ticktick.com/api/v2` |
| **Authentication** | OAuth2 Bearer Token | Session Token + Cookies |
| **Documentation** | Official (limited) | None (reverse-engineered) |
| **Features** | Tasks, Projects only | Full: Tags, Habits, Focus, Subtasks, Folders |
| **Stability** | Stable, versioned | May change without notice |
| **Rate Limits** | Documented | Undocumented |

### 1.3 Why Two APIs?

The official V1 API is severely limited:
- No tag support
- No habit tracking
- No focus/pomodoro statistics
- No subtask hierarchy (only checklist items)
- No project folders/groups
- Cannot list all tasks (only per-project)

The unofficial V2 API provides full access to TickTick's features but requires reverse-engineering and session-based authentication.

### 1.4 Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `api/base.py` | 470 | Abstract base HTTP client |
| `api/v1/client.py` | 532 | V1 API implementation |
| `api/v1/auth.py` | 342 | OAuth2 authentication |
| `api/v1/types.py` | 171 | V1 TypedDict definitions |
| `api/v2/client.py` | 1521 | V2 API implementation |
| `api/v2/auth.py` | 367 | Session authentication |
| `api/v2/types.py` | 799 | V2 TypedDict definitions |
| `unified/api.py` | 1968 | Unified API layer |
| `unified/router.py` | 322 | Routing logic |

---

## Section 2: Base HTTP Client (BaseTickTickClient)

**File**: `/src/ticktick_sdk/api/base.py`

### 2.1 Class Design

`BaseTickTickClient` is an abstract base class that provides common HTTP functionality for both V1 and V2 clients. It uses the **Template Method** pattern where subclasses implement specific authentication and URL configurations.

```
BaseTickTickClient (abstract)
    │
    ├── TickTickV1Client
    │
    └── TickTickV2Client
```

### 2.2 Constructor and Instance Variables

```python
def __init__(
    self,
    timeout: float = 30.0,       # Request timeout in seconds
    user_agent: str = DEFAULT_USER_AGENT,
) -> None:
    self._timeout = timeout
    self._user_agent = user_agent
    self._client: httpx.AsyncClient | None = None  # Lazy-initialized
    self._is_authenticated = False
```

### 2.3 Abstract Properties (Must Be Implemented)

```python
@property
@abstractmethod
def api_version(self) -> APIVersion:
    """Return APIVersion.V1 or APIVersion.V2"""

@property
@abstractmethod
def base_url(self) -> str:
    """Return the API base URL"""

@property
@abstractmethod
def is_authenticated(self) -> bool:
    """Check if authenticated"""

@abstractmethod
def _get_auth_headers(self) -> dict[str, str]:
    """Get authentication-specific headers"""
```

### 2.4 HTTP Client Setup

The HTTP client is lazily initialized using `httpx.AsyncClient`:

```python
async def _ensure_client(self) -> httpx.AsyncClient:
    if self._client is None or self._client.is_closed:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self._timeout),
            headers=self._get_base_headers(),
            follow_redirects=True,
        )
    return self._client
```

Key configuration:
- **Base URL**: Set per-client (V1 or V2)
- **Timeout**: Configurable, default 30 seconds
- **Follow Redirects**: Enabled
- **Connection Pooling**: Managed by httpx

### 2.5 Common Headers

All requests include these base headers:

```python
def _get_base_headers(self) -> dict[str, str]:
    return {
        "User-Agent": self._user_agent,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
```

The complete headers for a request combine base + auth headers:

```python
def _get_headers(self) -> dict[str, str]:
    headers = self._get_base_headers()
    if self.is_authenticated:
        headers.update(self._get_auth_headers())
    return headers
```

### 2.6 The Core _request() Method

This is the central HTTP method that all requests flow through:

```python
async def _request(
    self,
    method: str,              # GET, POST, PUT, DELETE
    endpoint: str,            # Relative to base_url
    *,
    params: dict | None = None,       # Query parameters
    json_data: dict | list | None = None,  # Request body
    headers: dict | None = None,       # Additional headers
    require_auth: bool = True,         # Require authentication?
) -> httpx.Response:
```

**Request Flow:**

1. Check authentication if required
2. Ensure HTTP client is initialized
3. Merge headers (base + auth + custom)
4. Make the HTTP request via httpx
5. Handle timeouts and network errors
6. Check response status and handle errors
7. Return the response

**Error Handling in _request():**

```python
try:
    response = await client.request(...)
except httpx.TimeoutException:
    raise TickTickAPIError("Request timeout")
except httpx.RequestError:
    raise TickTickAPIError("Request failed")

if not response.is_success:
    self._handle_error_response(response, endpoint)
```

### 2.7 HTTP Method Shortcuts

The base class provides convenience methods:

```python
async def _get(self, endpoint, *, params=None, ...) -> Response
async def _post(self, endpoint, *, json_data=None, ...) -> Response
async def _put(self, endpoint, *, json_data=None, ...) -> Response
async def _delete(self, endpoint, *, params=None, ...) -> Response
```

And JSON-returning variants:

```python
async def _get_json(self, endpoint, ...) -> Any    # Returns parsed JSON
async def _post_json(self, endpoint, ...) -> Any   # Returns parsed JSON
```

**Important**: `_get_json()` handles a V1 API quirk where nonexistent resources return HTTP 200 with an empty body:

```python
if not response.content or response.content.strip() == b"":
    raise TickTickNotFoundError("Resource not found (empty response)")
```

### 2.8 Error Response Handling

The `_handle_error_response()` method maps HTTP responses to semantic exceptions:

```python
def _handle_error_response(self, response: httpx.Response, endpoint: str) -> None:
```

**Error Code Detection:**

TickTick's API often returns HTTP 500 with semantic error codes in the body. The base client checks the response body first:

```python
error_body = response.json()
error_code = error_body.get("errorCode", "").lower()
```

**Error Code Sets:**

```python
_NOT_FOUND_ERROR_CODES = frozenset({
    "task_not_found", "project_not_found", "tag_not_found",
    "tag_not_exist", "folder_not_found", "group_not_found",
    "resource_not_found", "not_found",
})

_FORBIDDEN_ERROR_CODES = frozenset({
    "access_forbidden", "forbidden", "permission_denied",
})

_AUTH_ERROR_CODES = frozenset({
    "unauthorized", "invalid_token", "token_expired",
    "username_password_not_match", "incorrect_password_too_many_times",
})
```

**Status Code Mapping:**

| HTTP Status | Exception | Notes |
|-------------|-----------|-------|
| 401 | `TickTickAuthenticationError` | Invalid/expired token |
| 403 | `TickTickForbiddenError` | Access denied |
| 404 | `TickTickNotFoundError` | Resource not found |
| 429 | `TickTickRateLimitError` | Rate limit exceeded |
| 5xx | `TickTickServerError` | Server-side error |

**Special: Quota Exceeded Detection:**

```python
if error_body and error_body.get("id2error"):
    for error in error_body["id2error"].values():
        if error == "EXCEED_QUOTA":
            raise TickTickQuotaExceededError()
```

### 2.9 Lifecycle Management

**Context Manager Support:**

```python
async def __aenter__(self) -> Self:
    await self._ensure_client()
    return self

async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
    await self.close()
```

**Usage:**

```python
async with TickTickV2Client() as client:
    await client.authenticate(username, password)
    data = await client.sync()
```

**Cleanup:**

```python
async def close(self) -> None:
    if self._client is not None and not self._client.is_closed:
        await self._client.aclose()
        self._client = None
```

---

## Section 3: V1 API Client (TickTickV1Client)

**File**: `/src/ticktick_sdk/api/v1/client.py`

### 3.1 Class Overview

`TickTickV1Client` implements TickTick's official OAuth2-based Open API. It provides methods for basic task and project operations.

```python
class TickTickV1Client(BaseTickTickClient):
    @property
    def api_version(self) -> APIVersion:
        return APIVersion.V1

    @property
    def base_url(self) -> str:
        return "https://api.ticktick.com/open/v1"
```

### 3.2 Constructor

```python
def __init__(
    self,
    client_id: str,              # OAuth2 client ID
    client_secret: str,          # OAuth2 client secret
    redirect_uri: str,           # OAuth2 callback URL
    access_token: str | None = None,  # Pre-obtained token
    scopes: list[str] | None = None,  # Default: ["tasks:read", "tasks:write"]
    timeout: float = 30.0,
) -> None:
```

### 3.3 Authentication Integration

The V1 client delegates authentication to `OAuth2Handler`:

```python
self._oauth = OAuth2Handler(
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri=redirect_uri,
    scopes=scopes,
    timeout=timeout,
)

if access_token:
    self._oauth.set_access_token(access_token)
```

**Authentication Properties:**

```python
@property
def is_authenticated(self) -> bool:
    return self._oauth.is_authenticated  # Checks token validity

def _get_auth_headers(self) -> dict[str, str]:
    return {"Authorization": self._oauth.token.authorization_header}
    # Returns: "Bearer {access_token}"
```

### 3.4 Authentication Methods

```python
def get_authorization_url(self, state: str | None = None) -> tuple[str, str]:
    """Get OAuth2 authorization URL and state parameter"""
    return self._oauth.get_authorization_url(state)

async def authenticate_with_code(self, code: str, state: str | None = None) -> OAuth2Token:
    """Exchange authorization code for access token"""
    return await self._oauth.exchange_code(code, state)

async def refresh_token(self) -> OAuth2Token:
    """Refresh the access token using refresh token"""
    return await self._oauth.refresh_access_token()

def set_access_token(self, access_token: str) -> None:
    """Set a pre-obtained access token"""
    self._oauth.set_access_token(access_token)
```

### 3.5 Task Endpoints

#### GET /project/{projectId}/task/{taskId}

```python
async def get_task(self, project_id: str, task_id: str) -> TaskV1:
    """Get a task by project ID and task ID."""
    endpoint = f"/project/{project_id}/task/{task_id}"
    return await self._get_json(endpoint)
```

**Important**: V1 requires `project_id` to get a task. V2 does not.

#### POST /task

```python
async def create_task(
    self,
    title: str,                      # Required
    project_id: str,                 # Required
    *,
    content: str | None = None,
    desc: str | None = None,
    is_all_day: bool | None = None,
    start_date: str | None = None,   # ISO format
    due_date: str | None = None,     # ISO format
    time_zone: str | None = None,    # IANA timezone
    reminders: list[str] | None = None,  # TRIGGER format
    repeat_flag: str | None = None,  # RRULE format
    priority: int | None = None,     # 0, 1, 3, 5
    sort_order: int | None = None,
    items: list[dict] | None = None, # Checklist items
) -> TaskV1:
```

**Request Body Structure:**

```json
{
  "title": "Task title",
  "projectId": "project_id_here",
  "content": "Task description/notes",
  "priority": 5,
  "startDate": "2026-01-20T09:00:00+0000",
  "dueDate": "2026-01-20T17:00:00+0000",
  "timeZone": "America/Los_Angeles",
  "isAllDay": false,
  "reminders": ["TRIGGER:-PT30M"],
  "repeatFlag": "RRULE:FREQ=DAILY",
  "items": [
    {"title": "Subtask 1", "status": 0}
  ]
}
```

#### POST /task/{taskId}

```python
async def update_task(
    self,
    task_id: str,      # Required
    project_id: str,   # Required
    **kwargs,          # Same optional params as create_task
) -> TaskV1:
```

**Note**: V1 uses POST for updates, not PUT.

#### POST /project/{projectId}/task/{taskId}/complete

```python
async def complete_task(self, project_id: str, task_id: str) -> None:
    endpoint = f"/project/{project_id}/task/{task_id}/complete"
    await self._post(endpoint)
```

This is a dedicated completion endpoint that V2 lacks (V2 uses batch updates).

#### DELETE /project/{projectId}/task/{taskId}

```python
async def delete_task(self, project_id: str, task_id: str) -> None:
    endpoint = f"/project/{project_id}/task/{task_id}"
    await self._delete(endpoint)
```

### 3.6 Project Endpoints

#### GET /project

```python
async def get_projects(self) -> list[ProjectV1]:
    """Get all user projects."""
    return await self._get_json("/project")
```

#### GET /project/{projectId}

```python
async def get_project(self, project_id: str) -> ProjectV1:
    endpoint = f"/project/{project_id}"
    return await self._get_json(endpoint)
```

#### GET /project/{projectId}/data

```python
async def get_project_with_data(self, project_id: str) -> ProjectDataV1:
    """Get project with all its tasks and kanban columns."""
    endpoint = f"/project/{project_id}/data"
    return await self._get_json(endpoint)
```

**This endpoint is unique to V1** and returns:

```json
{
  "project": { "id": "...", "name": "..." },
  "tasks": [ { "id": "...", "title": "..." }, ... ],
  "columns": [ { "id": "...", "name": "To Do" }, ... ]
}
```

#### POST /project

```python
async def create_project(
    self,
    name: str,  # Required
    *,
    color: str | None = None,      # Hex color
    sort_order: int | None = None,
    view_mode: str | None = None,  # list, kanban, timeline
    kind: str | None = None,       # TASK, NOTE
) -> ProjectV1:
```

#### POST /project/{projectId}

```python
async def update_project(self, project_id: str, **kwargs) -> ProjectV1:
```

#### DELETE /project/{projectId}

```python
async def delete_project(self, project_id: str) -> None:
```

### 3.7 V1 Limitations

- **No tags**: Cannot assign or manage tags
- **No habits**: Habit tracking unavailable
- **No focus/pomodoro**: Statistics unavailable
- **No subtask hierarchy**: Only checklist items, not true subtasks
- **No folders**: Cannot organize projects into groups
- **Per-project task listing**: Cannot list all tasks at once

---

## Section 4: V1 Authentication (OAuth2Handler)

**File**: `/src/ticktick_sdk/api/v1/auth.py`

### 4.1 OAuth2Token Dataclass

```python
@dataclass
class OAuth2Token:
    access_token: str
    token_type: str = "Bearer"
    expires_in: int | None = None      # Seconds until expiry
    refresh_token: str | None = None
    scope: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
```

**Token Expiry Check:**

```python
@property
def is_expired(self) -> bool:
    if self.expires_in is None:
        return False  # Assume non-expiring
    expiry_time = self.created_at + timedelta(seconds=self.expires_in)
    # 60-second buffer before actual expiry
    return datetime.now(timezone.utc) >= (expiry_time - timedelta(seconds=60))
```

**Authorization Header:**

```python
@property
def authorization_header(self) -> str:
    return f"{self.token_type} {self.access_token}"
    # Returns: "Bearer eyJhbGciOiJIUzI1NiIsInR5..."
```

### 4.2 OAuth2Handler Class

```python
class OAuth2Handler:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        scopes: list[str] | None = None,  # Default: ["tasks:read", "tasks:write"]
        timeout: float = 30.0,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.scopes = scopes or ["tasks:read", "tasks:write"]
        self._token: OAuth2Token | None = None
        self._state: str | None = None
```

### 4.3 Authorization URL Generation

```python
def get_authorization_url(self, state: str | None = None) -> tuple[str, str]:
    if state is None:
        state = secrets.token_urlsafe(32)  # Generate random state

    self._state = state  # Store for verification

    params = {
        "client_id": self.client_id,
        "scope": " ".join(self.scopes),  # "tasks:read tasks:write"
        "state": state,
        "redirect_uri": self.redirect_uri,
        "response_type": "code",
    }

    return f"https://ticktick.com/oauth/authorize?{urlencode(params)}", state
```

**Generated URL Example:**

```
https://ticktick.com/oauth/authorize?
  client_id=abc123&
  scope=tasks%3Aread%20tasks%3Awrite&
  state=Hs8dKj2mNpQr...&
  redirect_uri=http%3A%2F%2Flocalhost%3A8080%2Fcallback&
  response_type=code
```

### 4.4 Token Exchange

```python
async def exchange_code(self, code: str, state: str | None = None) -> OAuth2Token:
```

**State Verification:**

```python
if state is not None and self._state is not None and state != self._state:
    raise TickTickOAuthError("State mismatch - possible CSRF attack")
```

**Token Request:**

```
POST https://ticktick.com/oauth/token
Content-Type: application/x-www-form-urlencoded
Authorization: Basic {base64(client_id:client_secret)}

code={authorization_code}&
grant_type=authorization_code&
scope=tasks:read tasks:write&
redirect_uri=http://localhost:8080/callback
```

**Basic Auth Header Generation:**

```python
def _get_basic_auth_header(self) -> str:
    credentials = f"{self.client_id}:{self.client_secret}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return f"Basic {encoded}"
```

**Response Parsing:**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "Bearer",
  "expires_in": 15552000,
  "refresh_token": "dGlja3RpY2sgcmVmcmVzaA...",
  "scope": "tasks:read tasks:write"
}
```

### 4.5 Token Refresh

```python
async def refresh_access_token(self) -> OAuth2Token:
```

**Refresh Request:**

```
POST https://ticktick.com/oauth/token
Content-Type: application/x-www-form-urlencoded
Authorization: Basic {base64(client_id:client_secret)}

grant_type=refresh_token&
refresh_token={refresh_token}&
scope=tasks:read tasks:write
```

**Note**: If a new refresh token is not returned, the old one is preserved:

```python
refresh_token=token_data.get("refresh_token", self._token.refresh_token)
```

### 4.6 OAuth2 Scopes

The V1 API supports two scopes:

| Scope | Description |
|-------|-------------|
| `tasks:read` | Read tasks and projects |
| `tasks:write` | Create, update, delete tasks and projects |

Both are required for full functionality.

---

## Section 5: V2 API Client (TickTickV2Client)

**File**: `/src/ticktick_sdk/api/v2/client.py` (1521 lines)

### 5.1 Class Overview

`TickTickV2Client` implements TickTick's unofficial, reverse-engineered V2 API. This is the most feature-rich client, providing access to all TickTick functionality.

```python
class TickTickV2Client(BaseTickTickClient):
    @property
    def api_version(self) -> APIVersion:
        return APIVersion.V2

    @property
    def base_url(self) -> str:
        return "https://api.ticktick.com/api/v2"
```

### 5.2 Constructor

```python
def __init__(
    self,
    device_id: str | None = None,  # Auto-generated if not provided
    timeout: float = 30.0,
) -> None:
    super().__init__(timeout=timeout)
    self._session_handler = SessionHandler(
        device_id=device_id,
        timeout=timeout,
    )
```

### 5.3 V2-Specific Headers

V2 API requires special headers that mimic the web client:

```python
V2_USER_AGENT = "Mozilla/5.0 (rv:145.0) Firefox/145.0"

def _get_x_device_header(self) -> str:
    return json.dumps({
        "platform": "web",
        "version": 6430,
        "id": self._session_handler.device_id,
    })

def _get_auth_headers(self) -> dict[str, str]:
    headers = {}
    if self._session_handler.session:
        session = self._session_handler.session

        headers["User-Agent"] = self.V2_USER_AGENT
        headers["X-Device"] = self._get_x_device_header()

        # Cookie is the PRIMARY auth mechanism for V2
        if session.cookies:
            cookie_str = "; ".join(f"{k}={v}" for k, v in session.cookies.items())
            headers["Cookie"] = cookie_str

    return headers
```

**Critical**: The `t` cookie contains the session token and is required for all V2 requests.

### 5.4 Authentication Methods

```python
async def authenticate(self, username: str, password: str) -> SessionToken:
    """Authenticate with username/password"""
    return await self._session_handler.authenticate(username, password)

async def authenticate_2fa(self, auth_id: str, totp_code: str) -> SessionToken:
    """Complete 2FA authentication"""
    return await self._session_handler.authenticate_2fa(auth_id, totp_code)

def set_session(self, session: SessionToken) -> None:
    """Set an existing session"""
    self._session_handler.set_session(session)

@property
def session(self) -> SessionToken | None:
    return self._session_handler.session

@property
def inbox_id(self) -> str | None:
    return self._session_handler.inbox_id
```

### 5.5 Sync Endpoint

#### GET /batch/check/0

```python
async def sync(self) -> SyncStateV2:
    """Get complete account state."""
    return await self._get_json("/batch/check/0")
```

**Response Structure:**

```json
{
  "inboxId": "inbox123456",
  "projectProfiles": [...],
  "projectGroups": [...],
  "syncTaskBean": {
    "update": [...]  // All active tasks
  },
  "tags": [...],
  "checkPoint": 1705678901234
}
```

This is the primary way to get all data at once.

### 5.6 User Endpoints

#### GET /user/status

```python
async def get_user_status(self) -> UserStatusV2:
    return await self._get_json("/user/status")
```

**Response:**

```json
{
  "userId": "123456",
  "username": "user@example.com",
  "inboxId": "inbox123456",
  "pro": true,
  "proStartDate": "2024-01-01",
  "proEndDate": "2025-01-01",
  "teamUser": false
}
```

#### GET /user/profile

```python
async def get_user_profile(self) -> UserProfileV2:
    return await self._get_json("/user/profile")
```

#### GET /user/preferences/settings

```python
async def get_user_preferences(self, include_web: bool = True) -> UserPreferencesV2:
    params = {"includeWeb": str(include_web).lower()}
    return await self._get_json("/user/preferences/settings", params=params)
```

#### GET /statistics/general

```python
async def get_user_statistics(self) -> UserStatisticsV2:
    return await self._get_json("/statistics/general")
```

### 5.7 Task Endpoints

#### GET /task/{id}

```python
async def get_task(self, task_id: str) -> TaskV2:
    endpoint = f"/task/{task_id}"
    return await self._get_json(endpoint)
```

**Key difference from V1**: Does NOT require `project_id`.

#### POST /batch/task

This is the primary endpoint for all task mutations:

```python
async def batch_tasks(
    self,
    add: list[TaskCreateV2] | None = None,
    update: list[TaskUpdateV2] | None = None,
    delete: list[TaskDeleteV2] | None = None,
) -> BatchResponseV2:
    data: BatchTaskRequestV2 = {
        "add": add or [],
        "update": update or [],
        "delete": delete or [],
        "addAttachments": [],
        "updateAttachments": [],
        "deleteAttachments": [],
    }
    return await self._post_json("/batch/task", json_data=data)
```

**Request Structure:**

```json
{
  "add": [
    {
      "title": "New Task",
      "projectId": "proj123",
      "tags": ["work", "urgent"],
      "priority": 5
    }
  ],
  "update": [
    {
      "id": "task456",
      "projectId": "proj123",
      "status": 2,
      "completedTime": "2026-01-17T10:00:00.000+0000"
    }
  ],
  "delete": [
    {"projectId": "proj123", "taskId": "task789"}
  ],
  "addAttachments": [],
  "updateAttachments": [],
  "deleteAttachments": []
}
```

**Response Structure:**

```json
{
  "id2etag": {
    "task456": "a1b2c3d4",
    "newTaskId": "e5f6g7h8"
  },
  "id2error": {}
}
```

#### Convenience Methods

```python
async def create_task(self, title: str, project_id: str, **kwargs) -> BatchResponseV2:
    task = {"title": title, "projectId": project_id, ...}
    return await self.batch_tasks(add=[task])

async def update_task(self, task_id: str, project_id: str, **kwargs) -> BatchResponseV2:
    task = {"id": task_id, "projectId": project_id, ...}
    return await self.batch_tasks(update=[task])

async def delete_task(self, project_id: str, task_id: str) -> BatchResponseV2:
    return await self.batch_tasks(delete=[{"projectId": project_id, "taskId": task_id}])
```

#### POST /batch/taskProject

Move tasks between projects:

```python
async def move_tasks(self, moves: list[TaskMoveV2]) -> Any:
    return await self._post_json("/batch/taskProject", json_data=moves)

async def move_task(self, task_id: str, from_project_id: str, to_project_id: str) -> Any:
    move = {
        "taskId": task_id,
        "fromProjectId": from_project_id,
        "toProjectId": to_project_id,
    }
    return await self.move_tasks([move])
```

#### POST /batch/taskParent

Manage subtask relationships:

```python
async def set_task_parent(
    self, task_id: str, project_id: str, parent_id: str
) -> BatchTaskParentResponseV2:
    data = [{"taskId": task_id, "projectId": project_id, "parentId": parent_id}]
    return await self._post_json("/batch/taskParent", json_data=data)

async def unset_task_parent(
    self, task_id: str, project_id: str, old_parent_id: str
) -> BatchTaskParentResponseV2:
    data = [{"taskId": task_id, "projectId": project_id, "oldParentId": old_parent_id}]
    return await self._post_json("/batch/taskParent", json_data=data)
```

**Critical**: Setting `parentId` during task creation is IGNORED. You must use `batch/taskParent` after creating the task.

#### GET /project/all/closed

Get completed or abandoned tasks:

```python
async def get_completed_tasks(
    self, from_date: datetime, to_date: datetime, limit: int = 100
) -> list[TaskV2]:
    params = {
        "from": from_date.strftime("%Y-%m-%d %H:%M:%S"),
        "to": to_date.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "Completed",
        "limit": str(limit),
    }
    return await self._get_json("/project/all/closed", params=params)

async def get_abandoned_tasks(
    self, from_date: datetime, to_date: datetime, limit: int = 100
) -> list[TaskV2]:
    # Same as above but status="Abandoned"
```

#### GET /project/all/trash/pagination

Get deleted tasks (trash):

```python
async def get_deleted_tasks(self, start: int = 0, limit: int = 500) -> TrashResponseV2:
    params = {"start": str(start), "limit": str(limit)}
    return await self._get_json("/project/all/trash/pagination", params=params)
```

### 5.8 Project Endpoints

#### POST /batch/project

```python
async def batch_projects(
    self,
    add: list[ProjectCreateV2] | None = None,
    update: list[ProjectUpdateV2] | None = None,
    delete: list[str] | None = None,  # List of project IDs
) -> BatchResponseV2:
    data = {"add": add or [], "update": update or [], "delete": delete or []}
    return await self._post_json("/batch/project", json_data=data)
```

**Create Request:**

```json
{
  "add": [{
    "name": "New Project",
    "color": "#FF6B6B",
    "kind": "TASK",
    "viewMode": "list",
    "groupId": "folder123"
  }]
}
```

**Update Request (with folder change):**

```json
{
  "update": [{
    "id": "proj123",
    "name": "Renamed Project",
    "color": "#4ECDC4",
    "groupId": "NONE"  // Use "NONE" to remove from folder
  }]
}
```

### 5.9 Project Group (Folder) Endpoints

#### POST /batch/projectGroup

```python
async def batch_project_groups(
    self,
    add: list[ProjectGroupCreateV2] | None = None,
    update: list[ProjectGroupUpdateV2] | None = None,
    delete: list[str] | None = None,
) -> BatchResponseV2:
    data = {"add": add or [], "update": update or [], "delete": delete or []}
    return await self._post_json("/batch/projectGroup", json_data=data)
```

**Create Request:**

```json
{
  "add": [{
    "name": "Work",
    "listType": "group"  // Always "group"
  }]
}
```

### 5.10 Tag Endpoints

#### POST /batch/tag

```python
async def batch_tags(
    self,
    add: list[TagCreateV2] | None = None,
    update: list[TagUpdateV2] | None = None,
) -> BatchResponseV2:
    data = {"add": add or [], "update": update or []}
    return await self._post_json("/batch/tag", json_data=data)
```

**Create Request:**

```json
{
  "add": [{
    "label": "Work",
    "name": "work",     // Lowercase, auto-generated from label
    "color": "#FF6B6B",
    "parent": "projects" // For nested tags
  }]
}
```

#### PUT /tag/rename

```python
async def rename_tag(self, old_name: str, new_label: str) -> Any:
    data = {"name": old_name, "newName": new_label}
    response = await self._put("/tag/rename", json_data=data)
    return response.json() if response.content else None
```

#### DELETE /tag

```python
async def delete_tag(self, name: str) -> None:
    params = {"name": name}
    await self._delete("/tag", params=params)
```

#### PUT /tag/merge

```python
async def merge_tags(self, source_name: str, target_name: str) -> Any:
    data = {"name": source_name, "newName": target_name}
    response = await self._put("/tag/merge", json_data=data)
    return response.json() if response.content else None
```

### 5.11 Focus/Pomodoro Endpoints

#### GET /pomodoros/statistics/heatmap/{from}/{to}

```python
async def get_focus_heatmap(
    self, start_date: date, end_date: date
) -> list[FocusHeatmapV2]:
    start_str = start_date.strftime("%Y%m%d")  # Format: YYYYMMDD
    end_str = end_date.strftime("%Y%m%d")
    endpoint = f"/pomodoros/statistics/heatmap/{start_str}/{end_str}"
    return await self._get_json(endpoint)
```

#### GET /pomodoros/statistics/dist/{from}/{to}

```python
async def get_focus_by_tag(
    self, start_date: date, end_date: date
) -> FocusDistributionV2:
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    endpoint = f"/pomodoros/statistics/dist/{start_str}/{end_str}"
    return await self._get_json(endpoint)
```

**Response:**

```json
{
  "tagDurations": {
    "work": 7200,    // Seconds
    "personal": 3600
  }
}
```

### 5.12 Habit Endpoints

#### GET /habits

```python
async def get_habits(self) -> list[HabitV2]:
    return await self._get_json("/habits")
```

#### GET /habitSections

```python
async def get_habit_sections(self) -> list[HabitSectionV2]:
    return await self._get_json("/habitSections")
```

**Response:**

```json
[
  {"id": "sec1", "name": "_morning", "sortOrder": 0},
  {"id": "sec2", "name": "_afternoon", "sortOrder": 1},
  {"id": "sec3", "name": "_night", "sortOrder": 2}
]
```

#### GET /user/preferences/habit

```python
async def get_habit_preferences(self) -> HabitPreferencesV2:
    params = {"platform": "web"}
    return await self._get_json("/user/preferences/habit", params=params)
```

#### POST /habits/batch

```python
async def batch_habits(
    self,
    add: list[HabitCreateV2] | None = None,
    update: list[HabitUpdateV2] | None = None,
    delete: list[str] | None = None,
) -> BatchResponseV2:
    data = {"add": add or [], "update": update or [], "delete": delete or []}
    return await self._post_json("/habits/batch", json_data=data)
```

**Create Request:**

```json
{
  "add": [{
    "id": "habit123abc456def789012",  // 24-char hex, client-generated
    "name": "Exercise",
    "type": "Boolean",
    "goal": 1.0,
    "iconRes": "habit_daily_check_in",
    "color": "#97E38B",
    "status": 0,
    "totalCheckIns": 0,
    "currentStreak": 0,
    "repeatRule": "RRULE:FREQ=WEEKLY;BYDAY=SU,MO,TU,WE,TH,FR,SA",
    "reminders": ["08:00"],
    "createdTime": "2026-01-17T10:00:00.000+0000",
    "modifiedTime": "2026-01-17T10:00:00.000+0000"
  }]
}
```

#### POST /habitCheckins/query

```python
async def get_habit_checkins(
    self, habit_ids: list[str], after_stamp: int = 0
) -> Any:
    data = {"habitIds": habit_ids, "afterStamp": after_stamp}
    return await self._post_json("/habitCheckins/query", json_data=data)
```

**Request:**

```json
{
  "habitIds": ["habit123", "habit456"],
  "afterStamp": 20260101  // YYYYMMDD format, 0 for all
}
```

**Response:**

```json
{
  "checkins": {
    "habit123": [
      {
        "habitId": "habit123",
        "checkinStamp": 20260117,
        "value": 1.0,
        "status": 2
      }
    ]
  }
}
```

#### POST /habitCheckins/batch

```python
async def batch_habit_checkins(
    self,
    add: list[HabitCheckinCreateV2] | None = None,
    update: list[dict] | None = None,
    delete: list[str] | None = None,
) -> BatchResponseV2:
    data = {"add": add or [], "update": update or [], "delete": delete or []}
    return await self._post_json("/habitCheckins/batch", json_data=data)
```

**Check-in Request (supports backdating):**

```json
{
  "add": [{
    "id": "checkin123abc456def789",
    "habitId": "habit123",
    "checkinStamp": 20260115,  // Past date for backdating
    "checkinTime": "2026-01-17T10:00:00.000+0000",
    "opTime": "2026-01-17T10:00:00.000+0000",
    "value": 1.0,
    "goal": 1.0,
    "status": 2  // Completed
  }]
}
```

---

## Section 6: V2 Authentication (SessionHandler)

**File**: `/src/ticktick_sdk/api/v2/auth.py`

### 6.1 SessionToken Dataclass

```python
@dataclass
class SessionToken:
    token: str              # Session token (also stored in 't' cookie)
    user_id: str            # Numeric user ID
    username: str           # Account email
    inbox_id: str           # Inbox project ID (e.g., "inbox123456")
    user_code: str | None = None
    is_pro: bool = False
    pro_start_date: str | None = None
    pro_end_date: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cookies: dict[str, str] = field(default_factory=dict)  # All session cookies
```

**Key Property:**

```python
@property
def authorization_header(self) -> str:
    return f"Bearer {self.token}"
```

### 6.2 Device ID Generation

V2 API requires a device ID in MongoDB ObjectId format (24 hex characters):

```python
def _generate_object_id() -> str:
    """Generate MongoDB-style ObjectId."""
    # 4 bytes: timestamp (seconds since epoch)
    timestamp = int(time.time()).to_bytes(4, "big")
    # 5 bytes: random value
    random_bytes = os.urandom(5)
    # 3 bytes: counter (random for simplicity)
    counter = os.urandom(3)
    return (timestamp + random_bytes + counter).hex()
```

**Example**: `678a3b2c4d5e6f7a8b9c0d1e`

### 6.3 SessionHandler Class

```python
class SessionHandler:
    DEFAULT_USER_AGENT = "Mozilla/5.0 (rv:145.0) Firefox/145.0"

    def __init__(
        self,
        device_id: str | None = None,
        timeout: float = 30.0,
    ):
        self.device_id = device_id or _generate_object_id()
        self.timeout = timeout
        self._session: SessionToken | None = None
```

### 6.4 X-Device Header

```python
def _get_x_device_header(self) -> str:
    return json.dumps({
        "platform": "web",
        "version": 6430,
        "id": self.device_id,
    })
```

### 6.5 Authentication Request

```python
async def authenticate(self, username: str, password: str) -> SessionToken:
```

**Request:**

```
POST https://api.ticktick.com/api/v2/user/signon?wc=true&remember=true
Content-Type: application/json
User-Agent: Mozilla/5.0 (rv:145.0) Firefox/145.0
X-Device: {"platform":"web","version":6430,"id":"678a3b2c4d5e6f7a8b9c0d1e"}

{
  "username": "user@example.com",
  "password": "secretpassword"
}
```

**Success Response:**

```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "userId": 123456,
  "username": "user@example.com",
  "inboxId": "inbox123456",
  "userCode": "abc123",
  "pro": true,
  "proStartDate": "2024-01-01",
  "proEndDate": "2025-01-01"
}
```

**Cookie Extraction:**

```python
cookies = {}
for cookie in response.cookies.jar:
    cookies[cookie.name] = cookie.value

# Ensure 't' cookie exists (contains session token)
if "t" not in cookies and "token" in data:
    cookies["t"] = data["token"]
```

### 6.6 2FA Detection and Handling

If the account has 2FA enabled, the initial sign-on returns:

```json
{
  "authId": "auth_session_id_here",
  "expireTime": 300
}
```

**Detection:**

```python
if "authId" in data and "token" not in data:
    raise TickTickSessionError(
        "Two-factor authentication required",
        requires_2fa=True,
        auth_id=data.get("authId"),
    )
```

**2FA Completion:**

```python
async def authenticate_2fa(self, auth_id: str, totp_code: str) -> SessionToken:
```

**Request:**

```
POST https://api.ticktick.com/api/v2/user/sign/mfa/code/verify
Content-Type: application/json
User-Agent: Mozilla/5.0 (rv:145.0) Firefox/145.0
X-Device: {...}
x-verify-id: {auth_id}

{
  "code": "123456",
  "method": "app"
}
```

---

## Section 7: V2 Request/Response Types

**File**: `/src/ticktick_sdk/api/v2/types.py` (799 lines)

### 7.1 Batch Request Types

#### BatchTaskRequestV2

```python
class BatchTaskRequestV2(TypedDict, total=False):
    add: list[TaskCreateV2]
    update: list[TaskUpdateV2]
    delete: list[TaskDeleteV2]
    addAttachments: list[Any]
    updateAttachments: list[Any]
    deleteAttachments: list[Any]
```

#### BatchProjectRequestV2

```python
class BatchProjectRequestV2(TypedDict, total=False):
    add: list[ProjectCreateV2]
    update: list[ProjectUpdateV2]
    delete: list[str]  # Project IDs
```

#### BatchTagRequestV2

```python
class BatchTagRequestV2(TypedDict, total=False):
    add: list[TagCreateV2]
    update: list[TagUpdateV2]
    # Note: No delete - use DELETE /tag endpoint
```

#### BatchHabitRequestV2

```python
class BatchHabitRequestV2(TypedDict, total=False):
    add: list[HabitCreateV2]
    update: list[HabitUpdateV2]
    delete: list[str]  # Habit IDs
```

### 7.2 Batch Response Types

#### BatchResponseV2

```python
class BatchResponseV2(TypedDict):
    id2etag: dict[str, str]    # {resource_id: new_etag}
    id2error: dict[str, str]   # {resource_id: error_message}
```

**Success Example:**

```json
{
  "id2etag": {
    "task123": "a1b2c3d4",
    "task456": "e5f6g7h8"
  },
  "id2error": {}
}
```

**Error Example:**

```json
{
  "id2etag": {},
  "id2error": {
    "task999": "TASK_NOT_FOUND"
  }
}
```

#### BatchTaskParentResponseV2

```python
class BatchTaskParentResponseV2(TypedDict):
    id2etag: dict[str, dict[str, Any]]  # Complex nested structure
    id2error: dict[str, str]
```

### 7.3 Key Error Codes in id2error

| Error Code | Meaning |
|------------|---------|
| `TASK_NOT_FOUND` | Task doesn't exist |
| `PROJECT_NOT_FOUND` | Project doesn't exist |
| `TAG_NOT_FOUND` | Tag doesn't exist |
| `EXCEED_QUOTA` | Free tier limit exceeded |
| `task not exists` | Alternative not found message |

### 7.4 Task Types

#### TaskV2 (Response)

```python
class TaskV2(TypedDict):
    id: str
    projectId: str
    etag: NotRequired[str]
    title: NotRequired[str]
    content: NotRequired[str]
    desc: NotRequired[str]
    kind: NotRequired[str]  # TEXT, NOTE, CHECKLIST
    status: NotRequired[int]  # -1, 0, 1, 2
    priority: NotRequired[int]  # 0, 1, 3, 5
    progress: NotRequired[int]  # 0-100
    deleted: NotRequired[int]  # 0 or 1
    startDate: NotRequired[str]
    dueDate: NotRequired[str]
    createdTime: NotRequired[str]
    modifiedTime: NotRequired[str]
    completedTime: NotRequired[str]
    timeZone: NotRequired[str]
    isAllDay: NotRequired[bool]
    repeatFlag: NotRequired[str]  # RRULE
    repeatFrom: NotRequired[int]  # 0, 1, 2
    reminders: NotRequired[list[TaskReminderV2]]
    parentId: NotRequired[str]
    childIds: NotRequired[list[str]]
    items: NotRequired[list[ItemV2]]  # Checklist items
    tags: NotRequired[list[str]]
    columnId: NotRequired[str]  # For Kanban
    sortOrder: NotRequired[int]
    assignee: NotRequired[Any]
    creator: NotRequired[int]
    focusSummaries: NotRequired[list[FocusSummaryV2]]
    # ... more fields
```

#### TaskCreateV2 (Request)

```python
class TaskCreateV2(TypedDict, total=False):
    title: str
    projectId: str
    content: str
    desc: str
    kind: str
    priority: int
    startDate: str
    dueDate: str
    timeZone: str
    isAllDay: bool
    reminders: list[TaskReminderV2]
    repeatFlag: str
    tags: list[str]
    items: list[ItemV2]
    sortOrder: int
    parentId: str  # IGNORED during creation - use batch/taskParent
```

#### TaskDeleteV2 (Request)

```python
class TaskDeleteV2(TypedDict):
    projectId: str
    taskId: str
```

#### TaskMoveV2 (Request)

```python
class TaskMoveV2(TypedDict):
    fromProjectId: str
    toProjectId: str
    taskId: str
```

### 7.5 Sync State Type

```python
class SyncStateV2(TypedDict):
    inboxId: str
    projectProfiles: list[ProjectV2]
    projectGroups: list[ProjectGroupV2]
    syncTaskBean: SyncTaskBeanV2
    tags: list[TagV2]
    filters: NotRequired[list[Any]]
    checkPoint: int
    # ... more fields

class SyncTaskBeanV2(TypedDict):
    update: list[TaskV2]  # All active tasks
    add: NotRequired[list[TaskV2]]
    delete: NotRequired[list[Any]]
    empty: NotRequired[bool]
```

---

## Section 8: API Routing (APIRouter)

**File**: `/src/ticktick_sdk/unified/router.py`

### 8.1 APIPreference Enum

```python
class APIPreference(StrEnum):
    V1_ONLY = auto()      # Only available in V1
    V2_ONLY = auto()      # Only available in V2
    V2_PRIMARY = auto()   # Prefer V2, fallback to V1
    V1_PRIMARY = auto()   # Prefer V1, fallback to V2
```

### 8.2 Operation Routing Table

The complete routing table defines which API to use for each operation:

```python
OPERATION_ROUTING: dict[str, OperationConfig] = {
    # =========== TASKS ===========
    "create_task": OperationConfig(
        APIPreference.V2_PRIMARY,
        "V2 supports tags, parent_id, and more fields"
    ),
    "get_task": OperationConfig(
        APIPreference.V2_PRIMARY,
        "V2 doesn't require project_id"
    ),
    "update_task": OperationConfig(
        APIPreference.V2_PRIMARY,
        "V2 has richer update options"
    ),
    "delete_task": OperationConfig(
        APIPreference.V2_PRIMARY,
        "V2 supports batch operations"
    ),
    "complete_task": OperationConfig(
        APIPreference.V1_PRIMARY,
        "V1 has dedicated endpoint, simpler"
    ),
    "list_all_tasks": OperationConfig(
        APIPreference.V2_ONLY,
        "V1 can only list per-project"
    ),
    "list_completed_tasks": OperationConfig(
        APIPreference.V2_ONLY,
        "V2-only feature"
    ),
    "list_deleted_tasks": OperationConfig(
        APIPreference.V2_ONLY,
        "V2-only feature (trash)"
    ),
    "move_task": OperationConfig(
        APIPreference.V2_ONLY,
        "V2-only feature"
    ),
    "set_task_parent": OperationConfig(
        APIPreference.V2_ONLY,
        "V2-only feature (subtasks)"
    ),

    # =========== PROJECTS ===========
    "create_project": OperationConfig(
        APIPreference.V2_PRIMARY,
        "V2 supports more options"
    ),
    "get_project": OperationConfig(
        APIPreference.V1_PRIMARY,
        "V1 has dedicated endpoint"
    ),
    "get_project_with_data": OperationConfig(
        APIPreference.V1_ONLY,
        "V1-only feature (includes tasks + columns)"
    ),
    "update_project": OperationConfig(
        APIPreference.V2_PRIMARY,
        "V2 supports batch operations"
    ),
    "delete_project": OperationConfig(
        APIPreference.V2_PRIMARY,
        "V2 supports batch operations"
    ),
    "list_projects": OperationConfig(
        APIPreference.V2_PRIMARY,
        "V2 returns more metadata"
    ),

    # =========== PROJECT GROUPS ===========
    "create_project_group": OperationConfig(APIPreference.V2_ONLY, "V2-only"),
    "update_project_group": OperationConfig(APIPreference.V2_ONLY, "V2-only"),
    "delete_project_group": OperationConfig(APIPreference.V2_ONLY, "V2-only"),
    "list_project_groups": OperationConfig(APIPreference.V2_ONLY, "V2-only"),

    # =========== TAGS ===========
    "create_tag": OperationConfig(APIPreference.V2_ONLY, "V2-only"),
    "update_tag": OperationConfig(APIPreference.V2_ONLY, "V2-only"),
    "rename_tag": OperationConfig(APIPreference.V2_ONLY, "V2-only"),
    "delete_tag": OperationConfig(APIPreference.V2_ONLY, "V2-only"),
    "merge_tags": OperationConfig(APIPreference.V2_ONLY, "V2-only"),
    "list_tags": OperationConfig(APIPreference.V2_ONLY, "V2-only"),

    # =========== USER ===========
    "get_user_profile": OperationConfig(APIPreference.V2_ONLY, "V2-only"),
    "get_user_status": OperationConfig(APIPreference.V2_ONLY, "V2-only"),
    "get_user_statistics": OperationConfig(APIPreference.V2_ONLY, "V2-only"),
    "get_user_settings": OperationConfig(APIPreference.V2_ONLY, "V2-only"),

    # =========== FOCUS ===========
    "get_focus_heatmap": OperationConfig(APIPreference.V2_ONLY, "V2-only"),
    "get_focus_by_tag": OperationConfig(APIPreference.V2_ONLY, "V2-only"),

    # =========== HABITS ===========
    "get_habit_checkins": OperationConfig(APIPreference.V2_ONLY, "V2-only"),

    # =========== SYNC ===========
    "sync_all": OperationConfig(APIPreference.V2_ONLY, "V2-only"),
}
```

### 8.3 Summary Table

| Operation | Routing | Reason |
|-----------|---------|--------|
| **Tasks** | | |
| create_task | V2_PRIMARY | Tags, more fields |
| get_task | V2_PRIMARY | No project_id needed |
| update_task | V2_PRIMARY | Richer options |
| delete_task | V2_PRIMARY | Batch support |
| complete_task | V1_PRIMARY | Dedicated endpoint |
| list_all_tasks | V2_ONLY | V1 only per-project |
| move_task | V2_ONLY | V2 exclusive |
| set_task_parent | V2_ONLY | V2 exclusive |
| **Projects** | | |
| create_project | V2_PRIMARY | More options |
| get_project | V1_PRIMARY | Dedicated endpoint |
| get_project_with_data | V1_ONLY | Includes tasks/columns |
| update_project | V2_PRIMARY | Batch support |
| delete_project | V2_PRIMARY | Batch support |
| list_projects | V2_PRIMARY | More metadata |
| **Tags/Habits/Focus/User** | V2_ONLY | Not in V1 |

### 8.4 APIRouter Class

```python
@dataclass
class APIRouter:
    v1_client: TickTickV1Client | None = None
    v2_client: TickTickV2Client | None = None
    _v1_verified: bool = False
    _v2_verified: bool = False

    @property
    def has_v1(self) -> bool:
        return self.v1_client is not None and self.v1_client.is_authenticated

    @property
    def has_v2(self) -> bool:
        return self.v2_client is not None and self.v2_client.is_authenticated

    @property
    def is_fully_configured(self) -> bool:
        return self.has_v1 and self.has_v2
```

**Routing Methods:**

```python
def get_primary_client(self, operation: str) -> tuple[str, object | None]:
    """Returns ('v1' or 'v2', client_instance)"""
    config = self.get_routing(operation)

    if config.preference == APIPreference.V1_ONLY:
        return ("v1", self.v1_client)
    elif config.preference == APIPreference.V2_ONLY:
        return ("v2", self.v2_client)
    elif config.preference == APIPreference.V1_PRIMARY:
        if self.has_v1:
            return ("v1", self.v1_client)
        return ("v2", self.v2_client)
    else:  # V2_PRIMARY
        if self.has_v2:
            return ("v2", self.v2_client)
        return ("v1", self.v1_client)

def get_fallback_client(self, operation: str) -> tuple[str, object | None]:
    """Get fallback client. Returns (None, None) for V1_ONLY/V2_ONLY."""
```

---

## Section 9: Unified API Layer (UnifiedTickTickAPI)

**File**: `/src/ticktick_sdk/unified/api.py` (1968 lines)

### 9.1 Purpose

The `UnifiedTickTickAPI` class is the main integration point that:

1. Manages both V1 and V2 client lifecycles
2. Routes operations to the appropriate API
3. Converts between unified models and API-specific formats
4. Handles batch operation errors
5. Verifies resource existence before V2 operations

### 9.2 Constructor

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

### 9.3 Initialization

```python
async def initialize(self) -> None:
    # Initialize V1 client
    self._v1_client = TickTickV1Client(
        client_id=...,
        client_secret=...,
        redirect_uri=...,
        access_token=...,
    )

    # Initialize and authenticate V2 client
    self._v2_client = TickTickV2Client(device_id=..., timeout=...)
    if username and password:
        session = await self._v2_client.authenticate(username, password)
        self._inbox_id = session.inbox_id

    # Create router
    self._router = APIRouter(v1_client=self._v1_client, v2_client=self._v2_client)

    # Verify both clients
    verification = await self._router.verify_clients()

    # BOTH V1 and V2 are REQUIRED
    if not self._router.is_fully_configured:
        raise TickTickConfigurationError("Both V1 and V2 APIs are required")
```

### 9.4 Batch Response Error Checking

The helper function `_check_batch_response_errors()` checks V2 batch responses:

```python
_BATCH_NOT_FOUND_ERRORS = frozenset({
    "TASK_NOT_FOUND", "PROJECT_NOT_FOUND", "TAG_NOT_FOUND",
    "task not exists", "project not found",
})

_BATCH_QUOTA_ERRORS = frozenset({"EXCEED_QUOTA"})

def _check_batch_response_errors(
    response: dict,
    operation: str,
    resource_ids: list[str] | None = None,
) -> None:
    id2error = response.get("id2error", {})
    if not id2error:
        return

    for resource_id, error_msg in id2error.items():
        error_upper = error_msg.upper()

        if any(nf in error_upper for nf in _BATCH_NOT_FOUND_ERRORS):
            raise TickTickNotFoundError(f"Resource not found: {error_msg}")

        if any(qe in error_upper for qe in _BATCH_QUOTA_ERRORS):
            raise TickTickQuotaExceededError(f"Quota exceeded: {error_msg}")

        raise TickTickAPIError(f"{operation} failed: {error_msg}")
```

### 9.5 Existence Verification Pattern

V2 batch operations often silently ignore nonexistent resources. The unified API verifies existence first:

```python
async def complete_task(self, task_id: str, project_id: str) -> None:
    # V2 batch silently accepts nonexistent tasks - verify first
    await self._v2_client.get_task(task_id)  # Raises NotFoundError if missing

    response = await self._v2_client.batch_tasks(update=[{
        "id": task_id,
        "projectId": project_id,
        "status": TaskStatus.COMPLETED,
    }])
    _check_batch_response_errors(response, "complete_task", [task_id])
```

This pattern is used for:
- `complete_task`
- `delete_task`
- `move_task`
- `set_task_parent`
- `unset_task_parent`
- `delete_project`
- `delete_project_group`
- `update_tag`
- `delete_tag`

### 9.6 Model Conversion

**From API to Unified Model:**

```python
async def list_all_tasks(self) -> list[Task]:
    state = await self._v2_client.sync()
    tasks_data = state.get("syncTaskBean", {}).get("update", [])
    return [Task.from_v2(t) for t in tasks_data]
```

**From Unified Model to API:**

```python
async def update_task(self, task: Task) -> Task:
    data = task.to_v2_dict(for_update=True)
    response = await self._v2_client.batch_tasks(update=[data])
    _check_batch_response_errors(response, "update_task", [task.id])
    return await self.get_task(task.id, task.project_id)
```

### 9.7 Subtask Creation Workaround

The V2 API ignores `parentId` during task creation. The unified API works around this:

```python
async def create_task(self, ..., parent_id: str | None = None) -> Task:
    # Create task WITHOUT parent_id (it's ignored anyway)
    response = await self._v2_client.create_task(
        title=title,
        project_id=project_id,
        # parent_id NOT passed here
    )

    task_id = next(iter(response.get("id2etag", {}).keys()))

    # Set parent separately if needed
    if parent_id:
        await self._v2_client.set_task_parent(task_id, project_id, parent_id)

    return await self.get_task(task_id, project_id)
```

### 9.8 Habit Check-in with Streak Calculation

The unified API calculates streaks client-side (like the web app):

```python
async def checkin_habit(
    self, habit_id: str, value: float = 1.0, checkin_date: date | None = None
) -> Habit:
    # Step 1: Create check-in record
    await self._v2_client.create_habit_checkin(...)

    # Step 2: Fetch ALL check-in records
    checkins = await self.get_habit_checkins([habit_id], after_stamp=0)

    # Step 3: Calculate streak and total
    calculated_streak = _calculate_streak_from_checkins(checkins, date.today())
    calculated_total = _count_total_checkins(checkins)

    # Step 4: Update habit with calculated values
    await self._v2_client.update_habit(
        habit_id=habit_id,
        name=original_habit.name,  # Must preserve!
        total_checkins=calculated_total,
        current_streak=calculated_streak,
    )
```

### 9.9 Inbox ID Caching

The inbox ID (special project for tasks without a project) is cached from authentication:

```python
@property
def inbox_id(self) -> str | None:
    return self._inbox_id  # Set during V2 authentication

async def create_task(self, ..., project_id: str | None = None) -> Task:
    if project_id is None:
        project_id = self._inbox_id
    if project_id is None:
        raise TickTickConfigurationError("No project ID and inbox unknown")
```

---

## Section 10: Error Handling

**File**: `/src/ticktick_sdk/exceptions.py`

### 10.1 Exception Hierarchy

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

### 10.2 Base Exception

```python
class TickTickError(Exception):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message
```

### 10.3 Authentication Exceptions

#### TickTickOAuthError

```python
class TickTickOAuthError(TickTickAuthenticationError):
    def __init__(
        self,
        message: str,
        oauth_error: str | None = None,         # e.g., "invalid_grant"
        oauth_error_description: str | None = None,
        details: dict | None = None,
    ):
        self.oauth_error = oauth_error
        self.oauth_error_description = oauth_error_description
```

**Common OAuth Errors:**
- `invalid_grant` - Invalid or expired authorization code
- `invalid_client` - Invalid client credentials
- `invalid_state` - State mismatch (possible CSRF)

#### TickTickSessionError

```python
class TickTickSessionError(TickTickAuthenticationError):
    def __init__(
        self,
        message: str,
        requires_2fa: bool = False,
        auth_id: str | None = None,  # For completing 2FA
        details: dict | None = None,
    ):
        self.requires_2fa = requires_2fa
        self.auth_id = auth_id
```

**2FA Detection:**

```python
try:
    await client.authenticate(username, password)
except TickTickSessionError as e:
    if e.requires_2fa:
        # Prompt user for TOTP code
        totp = input("Enter 2FA code: ")
        await client.authenticate_2fa(e.auth_id, totp)
```

### 10.4 API Exceptions

#### TickTickAPIError

```python
class TickTickAPIError(TickTickError):
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: str | None = None,
        api_version: str | None = None,  # "v1" or "v2"
        endpoint: str | None = None,
        details: dict | None = None,
    ):
        self.status_code = status_code
        self.response_body = response_body
        self.api_version = api_version
        self.endpoint = endpoint
```

#### TickTickRateLimitError

```python
class TickTickRateLimitError(TickTickAPIError):
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int | None = None,  # Seconds to wait
        **kwargs,
    ):
        self.retry_after = retry_after
```

**Handling:**

```python
try:
    await api.create_task(...)
except TickTickRateLimitError as e:
    if e.retry_after:
        await asyncio.sleep(e.retry_after)
        # Retry
```

#### TickTickNotFoundError

```python
class TickTickNotFoundError(TickTickAPIError):
    def __init__(
        self,
        message: str = "Resource not found",
        resource_type: str | None = None,  # "task", "project", etc.
        resource_id: str | None = None,
        **kwargs,
    ):
        # Always sets status_code=404
        self.resource_type = resource_type
        self.resource_id = resource_id
```

#### TickTickQuotaExceededError

```python
class TickTickQuotaExceededError(TickTickAPIError):
    def __init__(
        self,
        message: str = "Account quota exceeded",
        quota_type: str | None = None,
        **kwargs,
    ):
        self.quota_type = quota_type
```

**Free Tier Limits:**

- 99 tasks per list
- 9 lists
- 19 tags
- Limited habit tracking

### 10.5 Validation and Configuration Exceptions

#### TickTickValidationError

```python
class TickTickValidationError(TickTickError):
    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
        expected: str | None = None,
        details: dict | None = None,
    ):
        self.field = field
        self.value = value
        self.expected = expected
```

#### TickTickConfigurationError

```python
class TickTickConfigurationError(TickTickError):
    def __init__(
        self,
        message: str,
        missing_config: list[str] | None = None,  # Missing env vars
        details: dict | None = None,
    ):
        self.missing_config = missing_config or []
```

**Example:**

```python
raise TickTickConfigurationError(
    "Configuration incomplete: V2 session credentials incomplete",
    missing_config=["TICKTICK_USERNAME", "TICKTICK_PASSWORD"],
)
```

#### TickTickAPIUnavailableError

```python
class TickTickAPIUnavailableError(TickTickError):
    def __init__(
        self,
        message: str,
        operation: str | None = None,
        v1_error: TickTickError | None = None,
        v2_error: TickTickError | None = None,
        details: dict | None = None,
    ):
        self.operation = operation
        self.v1_error = v1_error
        self.v2_error = v2_error
```

---

## Section 11: HTTP Request Details

### 11.1 Request Format

All requests use JSON:

```
Content-Type: application/json
Accept: application/json
```

### 11.2 V1 Request Headers

```
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:95.0) Gecko/20100101 Firefox/95.0
Content-Type: application/json
Accept: application/json
Authorization: Bearer {access_token}
```

### 11.3 V2 Request Headers

```
User-Agent: Mozilla/5.0 (rv:145.0) Firefox/145.0
Content-Type: application/json
Accept: application/json
X-Device: {"platform":"web","version":6430,"id":"678a3b2c4d5e6f7a8b9c0d1e"}
Cookie: t={session_token}; AWSALB=...; AWSALBCORS=...
```

**Critical**: V2 authentication is primarily cookie-based. The `t` cookie contains the session token.

### 11.4 Timeout Configuration

```python
DEFAULT_TIMEOUT = 30.0  # seconds

# Configurable via constructor
client = TickTickV2Client(timeout=60.0)

# Or via settings
TICKTICK_TIMEOUT=60  # Environment variable
```

### 11.5 Date/Time Formats

| Format | Usage | Example |
|--------|-------|---------|
| V2 Task Dates | Task startDate, dueDate, etc. | `2026-01-17T10:00:00.000+0000` |
| V1 Task Dates | Task dates in V1 | `2026-01-17T10:00:00+0000` |
| V2 Query Params | Closed tasks endpoint | `2026-01-17 10:00:00` |
| Statistics Dates | Focus heatmap, habit stamps | `20260117` (YYYYMMDD) |

**Format Constants:**

```python
DATETIME_FORMAT_V2 = "%Y-%m-%dT%H:%M:%S.000+0000"
DATETIME_FORMAT_V1 = "%Y-%m-%dT%H:%M:%S%z"
DATETIME_FORMAT_V2_QUERY = "%Y-%m-%d %H:%M:%S"
DATE_FORMAT_STATS = "%Y%m%d"
```

### 11.6 Retry Logic

The SDK does **not** implement automatic retries. Users should implement their own:

```python
async def with_retry(coro, max_retries=3, backoff=1.0):
    for attempt in range(max_retries):
        try:
            return await coro
        except TickTickRateLimitError as e:
            if e.retry_after:
                await asyncio.sleep(e.retry_after)
            else:
                await asyncio.sleep(backoff * (2 ** attempt))
        except TickTickServerError:
            await asyncio.sleep(backoff * (2 ** attempt))
    raise
```

---

## Section 12: Batch Operations Deep Dive

### 12.1 Why V2 Uses Batch Endpoints

V2 API is designed for sync operations. The web client batches changes and sends them periodically. Benefits:

1. **Efficiency**: Multiple operations in one request
2. **Atomicity**: Related changes sent together
3. **Offline Support**: Changes queued and synced later

### 12.2 Batch Request Structure

All batch endpoints follow the same pattern:

```json
{
  "add": [...],     // Items to create
  "update": [...],  // Items to modify
  "delete": [...]   // IDs or objects to delete
}
```

### 12.3 Batch Response Structure

```json
{
  "id2etag": {
    "resource_id_1": "new_etag_1",
    "resource_id_2": "new_etag_2"
  },
  "id2error": {
    "failed_resource_id": "ERROR_CODE"
  }
}
```

**Key Points:**

- Success: ID appears in `id2etag` with new etag
- Failure: ID appears in `id2error` with error message
- **Important**: HTTP status is 200 even when operations fail!

### 12.4 Etags and Optimistic Concurrency

Etags are 8-character lowercase alphanumeric strings (e.g., `a1b2c3d4`).

**Purpose:**
- Detect concurrent modifications
- Track resource versions

**Usage:**

```python
# Fetch task
task = await client.get_task(task_id)
print(task["etag"])  # "a1b2c3d4"

# Update task (gets new etag)
response = await client.batch_tasks(update=[...])
new_etag = response["id2etag"][task_id]  # "e5f6g7h8"
```

### 12.5 Error Codes in Batch Responses

| Error Code | Meaning | Action |
|------------|---------|--------|
| `TASK_NOT_FOUND` | Task doesn't exist | Raise NotFoundError |
| `PROJECT_NOT_FOUND` | Project doesn't exist | Raise NotFoundError |
| `TAG_NOT_FOUND` | Tag doesn't exist | Raise NotFoundError |
| `EXCEED_QUOTA` | Free tier limit | Raise QuotaExceededError |
| `task not exists` | Alt not found | Raise NotFoundError |

### 12.6 Silent Failures

**Critical**: V2 batch operations can silently succeed for nonexistent resources:

```python
# This returns HTTP 200 with empty id2etag (no error)!
await client.batch_tasks(delete=[
    {"projectId": "proj123", "taskId": "nonexistent"}
])
```

The unified API works around this by verifying existence first.

### 12.7 Batch Example: Complete Workflow

```python
# Create task
create_response = await client.batch_tasks(add=[{
    "title": "New Task",
    "projectId": "proj123",
    "tags": ["work"],
}])
task_id = list(create_response["id2etag"].keys())[0]

# Update task
update_response = await client.batch_tasks(update=[{
    "id": task_id,
    "projectId": "proj123",
    "priority": 5,
}])

# Complete task
complete_response = await client.batch_tasks(update=[{
    "id": task_id,
    "projectId": "proj123",
    "status": 2,
    "completedTime": "2026-01-17T10:00:00.000+0000",
}])

# Delete task
delete_response = await client.batch_tasks(delete=[{
    "projectId": "proj123",
    "taskId": task_id,
}])
```

---

## Section 13: Authentication Tokens and Headers

### 13.1 V1 OAuth2 Flow

```
┌─────────────┐     1. Authorization Request      ┌─────────────┐
│   Your App  │ ─────────────────────────────────>│  TickTick   │
│             │     (redirect to ticktick.com)    │   OAuth     │
│             │                                    │   Server    │
│             │<────────────────────────────────── │             │
│             │     2. Authorization Code          │             │
│             │        (callback with ?code=...)   │             │
│             │                                    │             │
│             │     3. Token Exchange              │             │
│             │ ─────────────────────────────────> │             │
│             │     POST /oauth/token              │             │
│             │     (code + client_secret)         │             │
│             │                                    │             │
│             │<────────────────────────────────── │             │
│             │     4. Access Token                │             │
└─────────────┘        + Refresh Token             └─────────────┘
```

### 13.2 V1 Authorization Header

```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### 13.3 V1 Token Refresh

When the token expires (check with 60-second buffer):

```python
if token.is_expired:
    new_token = await handler.refresh_access_token()
```

### 13.4 V2 Session Flow

```
┌─────────────┐     1. POST /api/v2/user/signon   ┌─────────────┐
│   Your App  │ ─────────────────────────────────>│  TickTick   │
│             │     (username + password)          │   API       │
│             │                                    │             │
│             │<────────────────────────────────── │             │
│             │     2. Session Token + Cookies     │             │
│             │                                    │             │
│             │     3. API Requests                │             │
│             │ ─────────────────────────────────> │             │
│             │     (Cookie: t={token})            │             │
└─────────────┘                                    └─────────────┘
```

### 13.5 V2 Required Headers

```
User-Agent: Mozilla/5.0 (rv:145.0) Firefox/145.0
X-Device: {"platform":"web","version":6430,"id":"device_id_here"}
Cookie: t={session_token}
```

### 13.6 V2 Cookie Contents

The sign-on response sets multiple cookies:

| Cookie | Purpose |
|--------|---------|
| `t` | Session token (primary authentication) |
| `AWSALB` | AWS load balancer session affinity |
| `AWSALBCORS` | AWS load balancer CORS |

### 13.7 Token Expiry

| API | Token Lifetime | Refresh Mechanism |
|-----|---------------|-------------------|
| V1 | ~180 days | Use refresh_token |
| V2 | Unknown (long-lived) | Re-authenticate |

---

## Section 14: API Endpoint Reference Tables

### 14.1 V1 API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| **Tasks** | | |
| GET | `/project/{projectId}/task/{taskId}` | Get single task |
| POST | `/task` | Create task |
| POST | `/task/{taskId}` | Update task |
| POST | `/project/{projectId}/task/{taskId}/complete` | Complete task |
| DELETE | `/project/{projectId}/task/{taskId}` | Delete task |
| **Projects** | | |
| GET | `/project` | List all projects |
| GET | `/project/{projectId}` | Get single project |
| GET | `/project/{projectId}/data` | Get project with tasks/columns |
| POST | `/project` | Create project |
| POST | `/project/{projectId}` | Update project |
| DELETE | `/project/{projectId}` | Delete project |

### 14.2 V2 API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| **Authentication** | | |
| POST | `/user/signon` | Username/password login |
| POST | `/user/sign/mfa/code/verify` | Complete 2FA |
| **User** | | |
| GET | `/user/status` | Subscription status |
| GET | `/user/profile` | User profile |
| GET | `/user/preferences/settings` | User preferences |
| GET | `/statistics/general` | Productivity statistics |
| **Sync** | | |
| GET | `/batch/check/0` | Full sync (all data) |
| **Tasks** | | |
| GET | `/task/{id}` | Get single task |
| POST | `/batch/task` | Create/update/delete tasks |
| POST | `/batch/taskProject` | Move tasks between projects |
| POST | `/batch/taskParent` | Set/unset subtask relationships |
| GET | `/project/all/closed` | Completed/abandoned tasks |
| GET | `/project/all/trash/pagination` | Deleted tasks (trash) |
| **Projects** | | |
| POST | `/batch/project` | Create/update/delete projects |
| **Project Groups** | | |
| POST | `/batch/projectGroup` | Create/update/delete folders |
| **Tags** | | |
| POST | `/batch/tag` | Create/update tags |
| PUT | `/tag/rename` | Rename tag |
| PUT | `/tag/merge` | Merge tags |
| DELETE | `/tag` | Delete tag (query param: name) |
| **Focus/Pomodoro** | | |
| GET | `/pomodoros/statistics/heatmap/{from}/{to}` | Focus heatmap |
| GET | `/pomodoros/statistics/dist/{from}/{to}` | Focus by tag |
| **Habits** | | |
| GET | `/habits` | List all habits |
| GET | `/habitSections` | List habit sections |
| GET | `/user/preferences/habit` | Habit preferences |
| POST | `/habits/batch` | Create/update/delete habits |
| POST | `/habitCheckins/query` | Query check-in records |
| POST | `/habitCheckins/batch` | Create/update/delete check-ins |

### 14.3 Operation to Endpoint Mapping

| Operation | V1 Endpoint | V2 Endpoint | Preferred |
|-----------|-------------|-------------|-----------|
| **Tasks** | | | |
| Create task | POST /task | POST /batch/task | V2 |
| Get task | GET /project/{p}/task/{t} | GET /task/{t} | V2 |
| Update task | POST /task/{t} | POST /batch/task | V2 |
| Complete task | POST /.../complete | POST /batch/task | V1 |
| Delete task | DELETE /project/{p}/task/{t} | POST /batch/task | V2 |
| List all tasks | N/A | GET /batch/check/0 | V2 only |
| Move task | N/A | POST /batch/taskProject | V2 only |
| Subtasks | N/A | POST /batch/taskParent | V2 only |
| Completed tasks | N/A | GET /project/all/closed | V2 only |
| **Projects** | | | |
| List projects | GET /project | GET /batch/check/0 | V2 |
| Get project | GET /project/{p} | GET /batch/check/0 | V1 |
| Get with tasks | GET /project/{p}/data | GET /batch/check/0 | V1 |
| Create project | POST /project | POST /batch/project | V2 |
| Update project | POST /project/{p} | POST /batch/project | V2 |
| Delete project | DELETE /project/{p} | POST /batch/project | V2 |
| **Tags** | N/A | All V2 | V2 only |
| **Habits** | N/A | All V2 | V2 only |
| **Focus** | N/A | All V2 | V2 only |
| **User** | N/A | All V2 | V2 only |

---

## Appendix A: TickTick API Quirks

### A.1 Documented Quirks

| Quirk | Description | SDK Handling |
|-------|-------------|--------------|
| Recurrence requires start_date | `repeat_flag` ignored without `start_date` | Validation in UnifiedAPI |
| Subtasks require separate call | `parent_id` in create is ignored | Two-step create in UnifiedAPI |
| Soft delete | Tasks moved to trash (deleted=1) | Still accessible via get_task |
| V1 empty response | Returns HTTP 200 with empty body for missing resources | Check in `_get_json()` |
| V2 silent batch | Batch ops succeed silently for missing resources | Verify existence first |
| Inbox is special | Cannot be deleted, fixed ID format | Cache from authentication |
| Tag name vs label | `name` is lowercase ID, `label` is display | Auto-generate name from label |

### A.2 Date/Time Quirks

- All-day tasks have time 00:00:00 in UTC
- Timezone must be set correctly for proper display
- Clearing dates requires clearing both start_date and due_date together
- Habit check-in stamps use YYYYMMDD integer format

### A.3 V2 API Quirks

- `X-Device` header must have specific format
- Cookies are the PRIMARY auth mechanism (not Bearer token alone)
- Device ID must be 24 hex characters (MongoDB ObjectId format)
- User-Agent must look like a real browser
- Version number (6430) may need updating if API changes

---

## Appendix B: Constants Reference

**File**: `/src/ticktick_sdk/constants.py`

### B.1 API URLs

```python
TICKTICK_API_BASE_V1 = "https://api.ticktick.com/open/v1"
TICKTICK_API_BASE_V2 = "https://api.ticktick.com/api/v2"
TICKTICK_OAUTH_BASE = "https://ticktick.com/oauth"
```

### B.2 Status Enums

```python
class TaskStatus(IntEnum):
    ABANDONED = -1
    ACTIVE = 0
    COMPLETED_ALT = 1
    COMPLETED = 2

class TaskPriority(IntEnum):
    NONE = 0
    LOW = 1
    MEDIUM = 3
    HIGH = 5
```

### B.3 OAuth Scopes

```python
OAUTH_SCOPES = ["tasks:read", "tasks:write"]
```

---

## Appendix C: Quick Reference

### C.1 Minimal V2 Authentication

```python
from ticktick_sdk.api.v2 import TickTickV2Client

async with TickTickV2Client() as client:
    await client.authenticate("user@example.com", "password")
    state = await client.sync()
    tasks = state["syncTaskBean"]["update"]
```

### C.2 Minimal V1 Authentication

```python
from ticktick_sdk.api.v1 import TickTickV1Client

client = TickTickV1Client(
    client_id="...",
    client_secret="...",
    redirect_uri="http://localhost:8080/callback",
)

auth_url, state = client.get_authorization_url()
# User visits auth_url, approves, redirected with ?code=...

await client.authenticate_with_code(code, state)
projects = await client.get_projects()
```

### C.3 Full UnifiedAPI Usage

```python
from ticktick_sdk.unified import UnifiedTickTickAPI

async with UnifiedTickTickAPI(
    client_id="...",
    client_secret="...",
    v1_access_token="...",
    username="user@example.com",
    password="password",
) as api:
    # Full functionality
    tasks = await api.list_all_tasks()
    tags = await api.list_tags()
    habits = await api.list_habits()
```

---

*Document generated for TickTick SDK v0.2.1*
*This is a comprehensive technical reference for the API layer internals.*
