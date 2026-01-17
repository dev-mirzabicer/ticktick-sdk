# TickTick API v2 - Projects, Folders & Notes API Documentation

> **Analysis Date:** 2026-01-17
> **Source:** HAR file analysis from TickTick web application
> **Base URL:** `https://api.ticktick.com`

---

## Table of Contents

1. [Authentication & Common Headers](#authentication--common-headers)
2. [Create Folder (Project Group)](#create-folder-project-group)
3. [Create Project (Note Type)](#create-project-note-type)
4. [Create Project (Timeline Type)](#create-project-timeline-type)
5. [Create Note Task](#create-note-task)
6. [Update Note Task (Content)](#update-note-task-content)
7. [Project Types Comparison](#project-types-comparison)
8. [Field Type Reference](#field-type-reference)

---

## Authentication & Common Headers

All API requests require the following authentication and configuration headers:

### Required Headers

| Header | Value | Required | Description |
|--------|-------|----------|-------------|
| `Content-Type` | `application/json;charset=utf-8` | Yes | Content type for request body |
| `Accept` | `application/json, text/plain, */*` | Yes | Accepted response types |
| `X-Device` | JSON object (see below) | Yes | Device identification information |
| `hl` | `en_US` | Yes | Language/locale setting |
| `x-tz` | `Europe/Istanbul` (example) | Yes | User timezone (IANA format) |
| `X-Csrftoken` | CSRF token string | Yes | CSRF protection token |
| `traceid` | Unique request ID | Yes | Request tracing identifier |
| `Origin` | `https://ticktick.com` | Yes | Request origin |
| `Referer` | `https://ticktick.com/` | Yes | Request referrer |
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
| `platform` | string | Platform type (`web`, `ios`, `android`) |
| `os` | string | Operating system name and version |
| `device` | string | Browser/device name and version |
| `name` | string | Custom device name (optional) |
| `version` | integer | Client version number |
| `id` | string | Unique device identifier |
| `channel` | string | Installation channel (`website`, `app_store`, etc.) |
| `campaign` | string | Marketing campaign identifier |
| `websocket` | string | WebSocket connection identifier |

### Required Cookies

| Cookie | Description |
|--------|-------------|
| `t` | Authentication token (primary) |
| `SESSION` | Session identifier (Base64 encoded) |
| `_csrf_token` | CSRF token (must match X-Csrftoken header) |
| `AWSALB` / `AWSALBCORS` | AWS load balancer cookies |

---

## Create Folder (Project Group)

Creates a new folder (project group) to organize projects/lists.

### Endpoint
`POST /api/v2/batch/projectGroup`

### Description
Creates, updates, or deletes project groups (folders) in batch. Folders are used to organize multiple projects/lists together in the sidebar.

### Request

#### Headers
Standard authentication headers (see [Authentication & Common Headers](#authentication--common-headers))

#### Query Parameters
None

#### Body
```json
{
  "add": [
    {
      "showAll": true,
      "open": false,
      "name": "Some Folder",
      "sortOrder": -1099511693312,
      "teamId": null,
      "listType": "group",
      "sortType": "",
      "id": "696b8c362c6f3277d1cc4381"
    }
  ],
  "update": [],
  "delete": []
}
```

#### Body Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `add` | array | Yes | Array of project groups to create |
| `update` | array | Yes | Array of project groups to update |
| `delete` | array | Yes | Array of project group IDs to delete |

#### Project Group Object Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Client-generated unique identifier (24 hex chars) |
| `name` | string | Yes | Display name of the folder |
| `sortOrder` | integer (int64) | Yes | Sort position (negative values, lower = higher position) |
| `showAll` | boolean | Yes | Whether to show all projects in the folder |
| `open` | boolean | Yes | Whether the folder is expanded in UI |
| `teamId` | string \| null | No | Team ID if this is a team folder |
| `listType` | string | Yes | Type of list (`"group"` for folders) |
| `sortType` | string | No | Sort method for items within folder |

### Response

#### Status: 200 OK

```json
{
  "id2etag": {
    "696b8c362c6f3277d1cc4381": "l5f199i6"
  },
  "id2error": {}
}
```

#### Response Schema

| Field | Type | Description |
|-------|------|-------------|
| `id2etag` | object | Map of created/updated item IDs to their etags |
| `id2error` | object | Map of item IDs to error messages (empty on success) |

### Notes
- The `id` must be generated client-side (24-character hex string)
- `sortOrder` uses negative integers for positioning; more negative = higher in list
- The `etag` returned is used for optimistic concurrency control in subsequent updates
- Batch operations allow creating, updating, and deleting multiple items in a single request

---

## Create Project (Note Type)

Creates a new project/list with Note type (for note-taking rather than tasks).

### Endpoint
`POST /api/v2/project`

### Description
Creates a new project/list. This example shows creation of a Note-type list, which is designed for note-taking rather than task management.

### Request

#### Headers
Standard authentication headers (see [Authentication & Common Headers](#authentication--common-headers))

#### Query Parameters
None

#### Body
```json
{
  "name": "Another List",
  "color": null,
  "groupId": null,
  "sortOrder": -2199023321088,
  "inAll": true,
  "muted": false,
  "teamId": null,
  "kind": "NOTE",
  "viewMode": "list",
  "showType": 1,
  "reminderType": 1,
  "openToTeam": false,
  "isOwner": true
}
```

#### Body Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Display name of the project |
| `color` | string \| null | No | Color code for the project (hex or named color) |
| `groupId` | string \| null | No | ID of parent folder/group |
| `sortOrder` | integer (int64) | Yes | Sort position (negative, more negative = higher) |
| `inAll` | boolean | Yes | Whether to show items in "All" view |
| `muted` | boolean | Yes | Whether notifications are muted |
| `teamId` | string \| null | No | Team ID for shared projects |
| `kind` | string | Yes | Project type: `"NOTE"` or `"TASK"` |
| `viewMode` | string | Yes | Display mode: `"list"`, `"timeline"`, `"kanban"` |
| `showType` | integer | Yes | Show type setting (1 = default) |
| `reminderType` | integer | Yes | Reminder type setting (1 = default) |
| `openToTeam` | boolean | Yes | Whether project is open to team members |
| `isOwner` | boolean | Yes | Whether current user is owner |

### Response

#### Status: 200 OK

```json
{
  "id": "696b8cb18f082d43263a2e0f",
  "name": "Another List",
  "ownerId": null,
  "viewMode": "list",
  "teamId": null,
  "creatorRemoved": false,
  "referId": null,
  "kind": "NOTE",
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
| `id` | string | Server-generated project ID |
| `name` | string | Project display name |
| `ownerId` | string \| null | Owner user ID |
| `viewMode` | string | View mode setting |
| `teamId` | string \| null | Team ID |
| `creatorRemoved` | boolean | Whether creator has been removed |
| `referId` | string \| null | Reference ID for duplicated projects |
| `kind` | string | Project type (`"NOTE"` or `"TASK"`) |
| `needAudit` | boolean | Whether audit is required |
| `barcodeNeedAudit` | boolean | Whether barcode audit is required |
| `openToTeam` | boolean \| null | Team visibility setting |
| `teamMemberPermission` | string \| null | Team member permission level |
| `lastCloseToAll` | string \| null | Last time project was closed to all |

### Notes
- Unlike batch endpoints, the project ID is generated server-side
- `kind: "NOTE"` creates a note-focused list (no task checkboxes by default)
- The `viewMode` can be `"list"`, `"timeline"`, or `"kanban"`

---

## Create Project (Timeline Type)

Creates a new project/list with Timeline view mode (for time-based task organization).

### Endpoint
`POST /api/v2/project`

### Description
Creates a new project/list with Timeline view. Timeline lists are designed for time-based visualization of tasks with dates.

### Request

#### Headers
Standard authentication headers (see [Authentication & Common Headers](#authentication--common-headers))

#### Query Parameters
None

#### Body
```json
{
  "name": "Timeline List",
  "color": null,
  "groupId": null,
  "sortOrder": -3298534948864,
  "inAll": true,
  "muted": false,
  "teamId": null,
  "kind": "TASK",
  "viewMode": "timeline",
  "showType": 1,
  "reminderType": 1,
  "openToTeam": false,
  "isOwner": true
}
```

#### Body Schema

Same as [Create Project (Note Type)](#create-project-note-type), with these differences:

| Field | Value | Description |
|-------|-------|-------------|
| `kind` | `"TASK"` | Project type is TASK (with checkboxes) |
| `viewMode` | `"timeline"` | Timeline display mode |

### Response

#### Status: 200 OK

```json
{
  "id": "696b8cf18f082d43263a34b0",
  "name": "Timeline List",
  "ownerId": null,
  "viewMode": "timeline",
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

Same as [Create Project (Note Type)](#create-project-note-type) response.

### Notes
- `kind: "TASK"` creates a task-focused list with checkboxes
- `viewMode: "timeline"` enables the timeline visualization
- Timeline lists work best with tasks that have dates assigned

---

## Create Note Task

Creates a new note/task within a Note-type project.

### Endpoint
`POST /api/v2/batch/task`

### Description
Creates, updates, or deletes tasks/notes in batch. This example shows creating a new note within a Note-type project.

### Request

#### Headers
Standard authentication headers (see [Authentication & Common Headers](#authentication--common-headers))

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
      "isAllDay": true,
      "repeatFrom": null,
      "repeatFlag": null,
      "progress": 0,
      "assignee": null,
      "sortOrder": -1099511627776,
      "startDate": "2026-01-16T21:00:00.000+0000",
      "isFloating": false,
      "status": 0,
      "projectId": "696b8cb18f082d43263a2e0f",
      "kind": "NOTE",
      "createdTime": "2026-01-17T13:23:15.000+0000",
      "modifiedTime": "2026-01-17T13:23:15.000+0000",
      "title": "Hi this is a note!",
      "tags": [],
      "timeZone": "Europe/Istanbul",
      "content": "",
      "id": "696b8d432c6f3277d1cc43d1"
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
| `add` | array | Yes | Array of tasks/notes to create |
| `update` | array | Yes | Array of tasks/notes to update |
| `delete` | array | Yes | Array of task/note IDs to delete |
| `addAttachments` | array | Yes | Array of attachments to add |
| `updateAttachments` | array | Yes | Array of attachments to update |
| `deleteAttachments` | array | Yes | Array of attachment IDs to delete |

#### Task/Note Object Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Client-generated unique identifier (24 hex chars) |
| `projectId` | string | Yes | Parent project/list ID |
| `title` | string | Yes | Note/task title |
| `content` | string | No | Note body content (markdown supported) |
| `kind` | string | Yes | Item type: `"NOTE"` or `"TASK"` |
| `status` | integer | Yes | Status: 0 = incomplete, 2 = complete |
| `priority` | integer | Yes | Priority: 0 = none, 1 = low, 3 = medium, 5 = high |
| `progress` | integer | No | Progress percentage (0-100) |
| `startDate` | string (ISO 8601) | No | Start date/time |
| `dueDate` | string (ISO 8601) \| null | No | Due date/time |
| `isAllDay` | boolean | Yes | Whether dates are all-day (no specific time) |
| `isFloating` | boolean | Yes | Whether time is floating (no timezone) |
| `timeZone` | string | Yes | IANA timezone for dates |
| `sortOrder` | integer (int64) | Yes | Sort position within project |
| `tags` | array | No | Array of tag strings |
| `items` | array | No | Array of checklist items |
| `reminders` | array | No | Array of reminder objects |
| `exDate` | array | No | Exception dates for recurring items |
| `repeatFrom` | string \| null | No | Repeat reference point |
| `repeatFlag` | string \| null | No | Recurrence rule (RRULE format) |
| `assignee` | string \| null | No | Assigned user ID |
| `createdTime` | string (ISO 8601) | Yes | Creation timestamp |
| `modifiedTime` | string (ISO 8601) | Yes | Last modification timestamp |

### Response

#### Status: 200 OK

```json
{
  "id2etag": {
    "696b8d432c6f3277d1cc43d1": "vka8gpnd"
  },
  "id2error": {}
}
```

#### Response Schema

| Field | Type | Description |
|-------|------|-------------|
| `id2etag` | object | Map of created/updated item IDs to their etags |
| `id2error` | object | Map of item IDs to error messages (empty on success) |

### Notes
- The `id` is client-generated (24-character hex string)
- `kind: "NOTE"` matches the parent project's kind for note projects
- `startDate` is set to the start of the day in UTC (21:00 previous day for Europe/Istanbul timezone)
- The `etag` is required for subsequent updates (optimistic concurrency)

---

## Update Note Task (Content)

Updates an existing note's content.

### Endpoint
`POST /api/v2/batch/task`

### Description
Updates an existing task/note. This example shows updating the content of a note.

### Request

#### Headers
Standard authentication headers (see [Authentication & Common Headers](#authentication--common-headers))

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
      "isAllDay": true,
      "repeatFrom": null,
      "repeatFlag": null,
      "progress": 0,
      "assignee": null,
      "sortOrder": -1099511627776,
      "startDate": "2026-01-16T21:00:00.000+0000",
      "isFloating": false,
      "status": 0,
      "projectId": "696b8cb18f082d43263a2e0f",
      "kind": "NOTE",
      "etag": "cq1bv1ur",
      "createdTime": "2026-01-17T13:23:15.000+0000",
      "modifiedTime": "2026-01-17T13:24:08.000+0000",
      "title": "Hi this is a note!",
      "tags": [],
      "timeZone": "Europe/Istanbul",
      "content": "I'm writing contents of a note that i created before",
      "id": "696b8d432c6f3277d1cc43d1"
    }
  ],
  "delete": [],
  "addAttachments": [],
  "updateAttachments": [],
  "deleteAttachments": []
}
```

#### Key Differences from Create

| Field | Required for Update | Description |
|-------|---------------------|-------------|
| `id` | Yes | Existing task/note ID |
| `etag` | Yes | Current etag (from previous response) |
| `modifiedTime` | Yes | Should be updated to current time |
| `content` | Changed | The updated content |

### Response

#### Status: 200 OK

```json
{
  "id2etag": {
    "696b8d432c6f3277d1cc43d1": "xjqukfqn"
  },
  "id2error": {}
}
```

#### Response Schema

Same as create response. Note the etag has changed after the update.

### Notes
- The `etag` field is **required** for updates (optimistic locking)
- If the `etag` doesn't match the server's current value, the update will fail (conflict)
- The new `etag` from the response should be used for subsequent updates
- `modifiedTime` should be updated to reflect the actual modification time

---

## Project Types Comparison

### Kind Values

| Kind | Description | Checkbox | Primary Use |
|------|-------------|----------|-------------|
| `TASK` | Task-oriented project | Yes | To-do lists, task management |
| `NOTE` | Note-oriented project | No | Note-taking, documentation |

### View Mode Values

| View Mode | Description | Best For |
|-----------|-------------|----------|
| `list` | Standard list view | General use |
| `timeline` | Timeline/calendar view | Date-based planning |
| `kanban` | Kanban board view | Status-based workflows |

### Project Creation Comparison

| Field | Note Project | Timeline Project | Difference |
|-------|--------------|------------------|------------|
| `kind` | `"NOTE"` | `"TASK"` | Type of items |
| `viewMode` | `"list"` | `"timeline"` | Display format |
| Other fields | Same | Same | No difference |

---

## Field Type Reference

### ID Generation

Client-generated IDs follow this format:
- 24 hexadecimal characters
- Example: `696b8c362c6f3277d1cc4381`
- Appears to be MongoDB ObjectId format

### Sort Order

Sort order uses 64-bit signed integers:
- Negative values are standard
- More negative = higher position in list
- Example progression: `-1099511693312`, `-2199023321088`, `-3298534948864`
- Appears to use powers of 2 for spacing

### Date/Time Format

Dates use ISO 8601 format with timezone:
- Format: `YYYY-MM-DDTHH:mm:ss.sss+0000`
- Example: `2026-01-17T13:23:15.000+0000`
- Timezone offset uses `+0000` format (not `Z`)

### Etag Format

Etags are 8-character alphanumeric strings:
- Example: `l5f199i6`, `vka8gpnd`, `xjqukfqn`
- Used for optimistic concurrency control

### Priority Values

| Value | Level |
|-------|-------|
| 0 | None |
| 1 | Low |
| 3 | Medium |
| 5 | High |

### Status Values

| Value | Status |
|-------|--------|
| 0 | Incomplete |
| 2 | Complete |

---

## Error Handling

### Response Structure for Errors

Errors are returned in the `id2error` field:

```json
{
  "id2etag": {},
  "id2error": {
    "696b8d432c6f3277d1cc43d1": {
      "code": "ETAG_MISMATCH",
      "message": "Item has been modified by another client"
    }
  }
}
```

### Common Error Scenarios

1. **Etag Mismatch**: Update attempted with stale etag
2. **Invalid Project ID**: Referenced project doesn't exist
3. **Permission Denied**: User lacks access to the resource
4. **Invalid Fields**: Required fields missing or invalid format

---

## SDK Implementation Notes

### Creating a Folder

1. Generate a client-side ID (24 hex chars)
2. Calculate sort order based on existing folders
3. POST to `/api/v2/batch/projectGroup` with `add` array
4. Store returned etag for future updates

### Creating a Project

1. Determine project kind (`NOTE` or `TASK`) and viewMode
2. Calculate sort order based on existing projects
3. POST to `/api/v2/project`
4. Store returned ID for creating tasks/notes

### Creating a Note/Task

1. Generate a client-side ID (24 hex chars)
2. Set `kind` to match parent project's kind
3. POST to `/api/v2/batch/task` with `add` array
4. Store returned etag for future updates

### Updating a Note/Task

1. Include current `etag` from previous response
2. Update `modifiedTime` to current timestamp
3. POST to `/api/v2/batch/task` with `update` array
4. Store new etag from response

### Batch Operations

The batch endpoints support multiple operations in a single request:
- Create multiple items in `add`
- Update multiple items in `update`
- Delete multiple items in `delete`
- All operations in the same request are atomic
