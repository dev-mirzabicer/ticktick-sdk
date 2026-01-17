# TickTick Kanban & Columns API Documentation

This document provides comprehensive API documentation for TickTick's Kanban board and Column management features, extracted from HAR file analysis.

---

## Table of Contents

1. [Create Kanban Project](#1-create-kanban-project)
2. [Create Column (for Kanban Project)](#2-create-column-for-kanban-project)
3. [Get Columns for Project](#3-get-columns-for-project)
4. [Add Task to Kanban Column](#4-add-task-to-kanban-column)
5. [Move Task to Different Column](#5-move-task-to-different-column)
6. [Common Headers](#common-headers)
7. [Authentication](#authentication)

---

## Common Headers

All endpoints share these common headers:

| Header | Value | Required | Description |
|--------|-------|----------|-------------|
| `Host` | `api.ticktick.com` | Yes | API host |
| `Content-Type` | `application/json;charset=utf-8` | Yes (for POST) | Request body format |
| `Accept` | `application/json, text/plain, */*` | Yes | Expected response format |
| `X-Device` | JSON object (see below) | Yes | Device information |
| `hl` | `en_US` | Yes | Language/locale |
| `x-tz` | `Europe/Istanbul` | Yes | Timezone |
| `X-Csrftoken` | CSRF token string | Yes | CSRF protection token |
| `traceid` | Unique trace ID | Yes | Request tracing identifier |
| `Origin` | `https://ticktick.com` | Yes | CORS origin |
| `Referer` | `https://ticktick.com/` | Yes | Referrer URL |
| `Cookie` | Session cookies | Yes | Authentication cookies |

### X-Device Header Structure

```json
{
  "platform": "web",
  "os": "macOS 10.15",
  "device": "Firefox 146.0",
  "name": "",
  "version": 8006,
  "id": "6940c239ae8dc70ae62c4684",
  "channel": "website",
  "campaign": "",
  "websocket": "696b8a6a2c6f3277d1cc42d6"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `platform` | string | Platform type (e.g., "web") |
| `os` | string | Operating system |
| `device` | string | Browser/device name and version |
| `name` | string | Device name (can be empty) |
| `version` | integer | Client version number |
| `id` | string | Device/client ID |
| `channel` | string | Installation channel |
| `campaign` | string | Marketing campaign (can be empty) |
| `websocket` | string | WebSocket connection ID |

---

## Authentication

Authentication is handled via cookies. Key cookies include:

| Cookie | Description |
|--------|-------------|
| `t` | Primary authentication token (long hex string) |
| `SESSION` | Session identifier (base64 encoded) |
| `_csrf_token` | CSRF token for form submissions |
| `AWSALB` / `AWSALBCORS` | AWS load balancer cookies |

---

## 1. Create Kanban Project

### Endpoint
`POST /api/v2/project`

### Description
Creates a new project with Kanban view mode. This is used to create a project that displays tasks in a Kanban board format with columns.

### Request

#### Headers
See [Common Headers](#common-headers)

#### Query Parameters
None

#### Body
```json
{
  "name": "Example List",
  "color": "#FFD324",
  "groupId": "696b8c362c6f3277d1cc4381",
  "sortOrder": -549755879424,
  "inAll": true,
  "muted": false,
  "teamId": null,
  "kind": "TASK",
  "viewMode": "kanban",
  "showType": 1,
  "reminderType": 1,
  "openToTeam": false,
  "isOwner": true
}
```

#### Body Schema
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Project name (can include emoji) |
| `color` | string | No | Hex color code for the project |
| `groupId` | string | No | Parent folder/group ID |
| `sortOrder` | integer | No | Sort position (negative values for ordering) |
| `inAll` | boolean | No | Include in "All" view |
| `muted` | boolean | No | Mute notifications for this project |
| `teamId` | string/null | No | Team ID if shared project |
| `kind` | string | Yes | Project kind: "TASK" or "NOTE" |
| `viewMode` | string | Yes | View mode: "kanban" for Kanban board, "list" for list view |
| `showType` | integer | No | Display type setting |
| `reminderType` | integer | No | Reminder configuration |
| `openToTeam` | boolean | No | Whether project is open to team members |
| `isOwner` | boolean | No | Whether current user is owner |

### Response

#### Status: 200 OK
```json
{
  "id": "696b8c508f082d43263a248f",
  "name": "Example List",
  "ownerId": null,
  "viewMode": "kanban",
  "teamId": null,
  "creatorRemoved": false,
  "referId": null,
  "kind": "TASK",
  "needAudit": true,
  "barcodeNeedAudit": false,
  "openToTeam": null,
  "teamMemberPermission": null,
  "lastCloseToAll": null
}
```

#### Response Schema
| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique project identifier |
| `name` | string | Project name |
| `ownerId` | string/null | Owner user ID |
| `viewMode` | string | View mode ("kanban", "list", etc.) |
| `teamId` | string/null | Team ID if applicable |
| `creatorRemoved` | boolean | Whether creator has been removed |
| `referId` | string/null | Reference ID (for duplicated projects) |
| `kind` | string | Project kind |
| `needAudit` | boolean | Whether project needs audit |
| `barcodeNeedAudit` | boolean | Barcode audit flag |
| `openToTeam` | boolean/null | Team visibility setting |
| `teamMemberPermission` | string/null | Team member permissions |
| `lastCloseToAll` | string/null | Last close-to-all timestamp |

### Notes
- When creating a Kanban project, a default column "Not Sectioned" is automatically expected to be created via a separate API call
- The `viewMode: "kanban"` is the key differentiator from regular list projects
- Unicode characters (like emojis) are supported in project names

---

## 2. Create Column (for Kanban Project)

### Endpoint
`POST /api/v2/column`

### Description
Creates, updates, or deletes columns for a Kanban project. This endpoint uses a batch operation format supporting multiple operations in a single request.

### Request

#### Headers
See [Common Headers](#common-headers)

#### Query Parameters
None

#### Body
```json
{
  "add": [
    {
      "id": "696b8c512c6f3277d1cc4392",
      "userId": 130208689,
      "createdTime": "2026-01-17T13:19:13.152+0000",
      "name": "Not Sectioned",
      "projectId": "696b8c508f082d43263a248f",
      "sortOrder": 0
    }
  ],
  "update": [],
  "delete": []
}
```

#### Body Schema
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `add` | array | Yes | Array of columns to add |
| `update` | array | Yes | Array of columns to update |
| `delete` | array | Yes | Array of column IDs to delete |

#### Column Object Schema (for `add` array)
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique column identifier (client-generated) |
| `userId` | integer | Yes | User ID who owns the column |
| `createdTime` | string | Yes | ISO 8601 timestamp with timezone |
| `name` | string | Yes | Column display name |
| `projectId` | string | Yes | Parent project ID |
| `sortOrder` | integer | Yes | Sort position (0 for first column) |

### Response

#### Status: 200 OK
```json
{
  "id2etag": {
    "696b8c512c6f3277d1cc4392": "lo736kb6"
  },
  "id2error": {}
}
```

#### Response Schema
| Field | Type | Description |
|-------|------|-------------|
| `id2etag` | object | Map of column ID to etag (version identifier) |
| `id2error` | object | Map of column ID to error message (empty if success) |

### Notes
- The `id` field for new columns is client-generated
- The first column (default) is typically named "Not Sectioned" with `sortOrder: 0`
- The `etag` returned is used for optimistic concurrency control
- This endpoint is called immediately after creating a Kanban project to set up the default column

---

## 3. Get Columns for Project

### Endpoint
`GET /api/v2/column/project/{projectId}`

### Description
Retrieves all columns for a specific project. Used to fetch the Kanban board structure.

### Request

#### Headers
See [Common Headers](#common-headers) (excluding Content-Type for GET request)

#### URL Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `projectId` | string | Yes | The project ID to get columns for |

#### Query Parameters
None

### Response

#### Status: 200 OK
```json
[
  {
    "id": "696b92a72c6f3277d1cc4638",
    "projectId": "696b8c508f082d43263a248f",
    "name": "Next Section",
    "sortOrder": 1099511627776,
    "createdTime": "2026-01-17T13:46:15.794+0000",
    "modifiedTime": "2026-01-17T13:46:15.794+0000",
    "etag": "d0jlm4y7"
  },
  {
    "id": "696b8c512c6f3277d1cc4392",
    "projectId": "696b8c508f082d43263a248f",
    "name": "Not Sectioned",
    "sortOrder": 0,
    "createdTime": "2026-01-17T13:19:13.152+0000",
    "modifiedTime": "2026-01-17T13:19:13.152+0000",
    "etag": "lo736kb6"
  }
]
```

#### Response Schema (Array of Column objects)
| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique column identifier |
| `projectId` | string | Parent project ID |
| `name` | string | Column display name |
| `sortOrder` | integer | Sort position (lower = earlier in board) |
| `createdTime` | string | ISO 8601 creation timestamp |
| `modifiedTime` | string | ISO 8601 last modification timestamp |
| `etag` | string | Version identifier for concurrency control |

### Notes
- Columns are returned in an array, sorted by `sortOrder`
- The `sortOrder` uses large integers (powers of 2) to allow easy insertion between columns
- Lower `sortOrder` values appear first (left side in Kanban board)

---

## 4. Add Task to Kanban Column

### Endpoint
`POST /api/v2/batch/task`

### Description
Creates a new task and assigns it to a specific Kanban column. This is the same batch task endpoint used for general task operations, but includes the `columnId` field for Kanban placement.

### Request

#### Headers
See [Common Headers](#common-headers)

#### Query Parameters
None

#### Body
```json
{
  "add": [
    {
      "items": [],
      "reminders": [],
      "exDate": [],
      "dueDate": null,
      "priority": 0,
      "progress": 0,
      "assignee": null,
      "sortOrder": -1099511627776,
      "startDate": null,
      "isFloating": false,
      "columnId": "696b92a72c6f3277d1cc4638",
      "status": 0,
      "projectId": "696b8c508f082d43263a248f",
      "kind": null,
      "createdTime": "2026-01-17T13:46:43.000+0000",
      "modifiedTime": "2026-01-17T13:46:43.000+0000",
      "title": "Add task to a column in kanban",
      "tags": [],
      "timeZone": "Europe/Istanbul",
      "content": "",
      "id": "696b92c32c6f3277d1cc463e"
    }
  ],
  "update": [],
  "delete": [],
  "addAttachments": [],
  "updateAttachments": [],
  "deleteAttachments": []
}
```

#### Body Schema
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `add` | array | Yes | Array of tasks to add |
| `update` | array | Yes | Array of tasks to update |
| `delete` | array | Yes | Array of task IDs to delete |
| `addAttachments` | array | Yes | Array of attachments to add |
| `updateAttachments` | array | Yes | Array of attachments to update |
| `deleteAttachments` | array | Yes | Array of attachment IDs to delete |

#### Task Object Schema (for `add` array)
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique task identifier (client-generated) |
| `title` | string | Yes | Task title/name |
| `content` | string | No | Task description/notes |
| `projectId` | string | Yes | Parent project ID |
| `columnId` | string | Yes* | Kanban column ID (*required for Kanban projects) |
| `status` | integer | Yes | Task status: 0 = incomplete, 2 = complete |
| `priority` | integer | No | Priority level: 0 = none, 1 = low, 3 = medium, 5 = high |
| `progress` | integer | No | Task progress (0-100) |
| `sortOrder` | integer | Yes | Sort position within column |
| `dueDate` | string/null | No | ISO 8601 due date |
| `startDate` | string/null | No | ISO 8601 start date |
| `isFloating` | boolean | No | Whether task has floating time |
| `timeZone` | string | Yes | Timezone for the task |
| `createdTime` | string | Yes | ISO 8601 creation timestamp |
| `modifiedTime` | string | Yes | ISO 8601 last modification timestamp |
| `items` | array | No | Checklist items |
| `reminders` | array | No | Task reminders |
| `exDate` | array | No | Exception dates (for recurring tasks) |
| `tags` | array | No | Task tags |
| `assignee` | string/null | No | Assigned user ID |
| `kind` | string/null | No | Task kind |

### Response

#### Status: 200 OK
```json
{
  "id2etag": {
    "696b92c32c6f3277d1cc463e": "8ld0djej"
  },
  "id2error": {}
}
```

#### Response Schema
| Field | Type | Description |
|-------|------|-------------|
| `id2etag` | object | Map of task ID to etag (version identifier) |
| `id2error` | object | Map of task ID to error message (empty if success) |

### Notes
- The `columnId` field is what places the task in a specific Kanban column
- Tasks in Kanban projects MUST have a `columnId` specified
- The `sortOrder` determines vertical position within the column
- Client generates the task `id` before sending the request

---

## 5. Move Task to Different Column

### Endpoint
`POST /api/v2/batch/task`

### Description
Updates an existing task to move it to a different Kanban column. Uses the same batch endpoint as task creation, but with the task in the `update` array instead of `add`.

### Request

#### Headers
See [Common Headers](#common-headers)

#### Query Parameters
None

#### Body
```json
{
  "add": [],
  "update": [
    {
      "items": [],
      "reminders": [],
      "exDate": [],
      "dueDate": null,
      "priority": 0,
      "progress": 0,
      "assignee": null,
      "sortOrder": -2199023255552,
      "startDate": null,
      "isFloating": false,
      "columnId": "696b8c512c6f3277d1cc4392",
      "status": 0,
      "projectId": "696b8c508f082d43263a248f",
      "kind": null,
      "etag": "8ld0djej",
      "createdTime": "2026-01-17T13:46:43.000+0000",
      "modifiedTime": "2026-01-17T13:49:12.000+0000",
      "title": "Add task to a column in kanban",
      "tags": [],
      "timeZone": "Europe/Istanbul",
      "content": "",
      "id": "696b92c32c6f3277d1cc463e"
    }
  ],
  "delete": [],
  "addAttachments": [],
  "updateAttachments": [],
  "deleteAttachments": []
}
```

#### Body Schema
Same as [Add Task to Kanban Column](#4-add-task-to-kanban-column), but task goes in `update` array.

#### Additional Fields for Update
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `etag` | string | Yes | Current version tag (from previous response) |

### Response

#### Status: 200 OK
```json
{
  "id2etag": {
    "696b92c32c6f3277d1cc463e": "ca29pri6"
  },
  "id2error": {}
}
```

#### Response Schema
| Field | Type | Description |
|-------|------|-------------|
| `id2etag` | object | Map of task ID to new etag |
| `id2error` | object | Map of task ID to error message (empty if success) |

### Notes
- To move a task to a different column, change the `columnId` field
- The `etag` field is **required** for updates (optimistic concurrency control)
- The `modifiedTime` should be updated to the current time
- The `sortOrder` can be changed to position the task within the new column
- A new `etag` is returned after successful update - use this for subsequent updates

---

## ID Generation

TickTick uses a specific ID format:
- IDs appear to be 24-character hexadecimal strings
- Format resembles MongoDB ObjectIds
- Client generates IDs before sending requests
- Example: `696b92c32c6f3277d1cc463e`

## SortOrder Values

TickTick uses large integer values for sort ordering:
- Uses powers of 2 for spacing (e.g., `1099511627776` = 2^40)
- Negative values are valid and typically indicate "top of list"
- This allows easy insertion between existing items by averaging

| Example Value | Meaning |
|---------------|---------|
| `0` | Default/first position |
| `-549755879424` | High priority (top of list) |
| `-1099511627776` | Very high priority |
| `-2199023255552` | Highest priority seen |
| `1099511627776` | Lower in list |

## Timestamps

All timestamps use ISO 8601 format with timezone:
- Format: `YYYY-MM-DDTHH:mm:ss.SSS+ZZZZ`
- Example: `2026-01-17T13:46:43.000+0000`

---

## Error Handling

Errors are returned in the `id2error` field of batch responses:

```json
{
  "id2etag": {},
  "id2error": {
    "some-task-id": "Error message here"
  }
}
```

Common error scenarios:
- Invalid `etag` (concurrency conflict)
- Invalid `projectId` or `columnId`
- Missing required fields
- Authentication failure

---

## Workflow Examples

### Creating a New Kanban Project with Columns

1. `POST /api/v2/project` with `viewMode: "kanban"`
2. `POST /api/v2/column` to create default "Not Sectioned" column
3. `POST /api/v2/column` to add additional columns as needed

### Adding a Task to Kanban Board

1. `GET /api/v2/column/project/{projectId}` to get available columns
2. `POST /api/v2/batch/task` with `columnId` set to target column

### Moving a Task Between Columns

1. Fetch current task data (including `etag`)
2. `POST /api/v2/batch/task` with updated `columnId` in `update` array
3. Store new `etag` from response for future updates
