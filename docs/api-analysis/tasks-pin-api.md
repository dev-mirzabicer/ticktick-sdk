# TickTick Tasks & Pin API Documentation

This document provides comprehensive API documentation extracted from HAR files capturing TickTick web application requests related to task operations including creating, updating, pinning, and marking tasks as "won't do".

---

## Table of Contents
1. [Batch Task Endpoint Overview](#batch-task-endpoint-overview)
2. [Pin Task Operation](#pin-task-operation)
3. [Mark Task as Won't Do](#mark-task-as-wont-do)
4. [Create Task (Basic)](#create-task-basic)
5. [Create Task with Daily Repeat (COUNT-based)](#create-task-with-daily-repeat-count-based)
6. [Create Task with Daily Repeat (Indefinite)](#create-task-with-daily-repeat-indefinite)
7. [Complete Recurring Task Instance](#complete-recurring-task-instance)
8. [Common Headers](#common-headers)
9. [Task Object Schema](#task-object-schema)
10. [Key Field Analysis](#key-field-analysis)

---

## Batch Task Endpoint Overview

All task operations use a single unified endpoint that supports batch operations for adding, updating, and deleting tasks.

### Endpoint
`POST /api/v2/batch/task`

### Base URL
`https://api.ticktick.com`

### Full URL
`https://api.ticktick.com/api/v2/batch/task`

---

## Pin Task Operation

### Description
Pins a task to keep it prominently displayed. The pin operation is implemented by updating the task with a `pinnedTime` field set to an ISO 8601 timestamp.

### Endpoint
`POST /api/v2/batch/task`

### Source File
`(pin a task) ticktick.com_api_v2_batch_task_Archive [26-01-17 16-54-03].har`

### Request

#### Headers
| Header | Value | Required |
|--------|-------|----------|
| Host | api.ticktick.com | Yes |
| Content-Type | application/json;charset=utf-8 | Yes |
| X-Device | JSON object with device info | Yes |
| X-Csrftoken | CSRF token string | Yes |
| Cookie | Session cookies including `t`, `SESSION`, `_csrf_token` | Yes |
| hl | en_US | No |
| x-tz | Europe/Istanbul (timezone) | No |
| traceid | Unique trace ID | No |

#### Body
```json
{
  "add": [],
  "update": [
    {
      "items": [],
      "reminders": [
        {
          "id": "696b8a802c6f3277d1cc4311",
          "trigger": "TRIGGER:PT0S"
        }
      ],
      "exDate": [],
      "dueDate": null,
      "priority": 5,
      "isAllDay": false,
      "pinnedTime": "2026-01-17T13:53:44.000+0000",
      "repeatFrom": "0",
      "repeatFlag": "RRULE:FREQ=WEEKLY;INTERVAL=1;WKST=SU;UNTIL=20260227;BYDAY=SA",
      "progress": 0,
      "assignee": null,
      "sortOrder": -1099512217600,
      "startDate": "2026-01-17T15:00:00.000+0000",
      "isFloating": false,
      "columnId": "6940c3d51263b74715346bd9",
      "status": 0,
      "projectId": "6940c3d51263b74715346bd6",
      "kind": null,
      "etag": "v9b9ap2e",
      "createdTime": "2026-01-17T13:12:28.000+0000",
      "modifiedTime": "2026-01-17T13:53:44.000+0000",
      "title": "test task",
      "tags": [],
      "timeZone": "Europe/Istanbul",
      "content": "",
      "id": "696b8abc2c6f3277d1cc431a"
    }
  ],
  "delete": [],
  "addAttachments": [],
  "updateAttachments": [],
  "deleteAttachments": []
}
```

#### Key Pin Field
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| **pinnedTime** | string (ISO 8601) | **CRITICAL** | The timestamp when the task was pinned. Set to an ISO 8601 datetime string to pin, set to `null` to unpin. |

### Response
#### Status: 200 OK
```json
{
  "id2etag": {
    "696b8abc2c6f3277d1cc431a": "wxbn3q91"
  },
  "id2error": {}
}
```

#### Response Schema
| Field | Type | Description |
|-------|------|-------------|
| id2etag | object | Map of task IDs to their new etag values after the operation |
| id2error | object | Map of task IDs to error messages (empty on success) |

### Notes
- **CRITICAL DISCOVERY**: The `pinnedTime` field is the key to pinning/unpinning tasks
- To PIN a task: Set `pinnedTime` to an ISO 8601 timestamp (e.g., `"2026-01-17T13:53:44.000+0000"`)
- To UNPIN a task: Set `pinnedTime` to `null`
- The `pinnedTime` value represents when the task was pinned, not when it should be unpinned
- The `etag` field must be included when updating to prevent conflicts

---

## Mark Task as Won't Do

### Description
Marks a task as "won't do" (abandoned). This is achieved by setting `status` to `-1` and providing a `completedTime` and `completedUserId`.

### Endpoint
`POST /api/v2/batch/task`

### Source File
`(mark as wont do) ticktick.com_api_v2_batch_task_Archive [26-01-17 16-17-05].har`

### Request

#### Body
```json
{
  "add": [
    {
      "items": [],
      "reminders": [
        {
          "id": "696b8ba82c6f3277d1cc4342",
          "trigger": "TRIGGER:PT0S"
        }
      ],
      "exDate": [],
      "dueDate": "2026-01-17T15:00:00.000+0000",
      "priority": 0,
      "isAllDay": false,
      "repeatFrom": "2",
      "repeatFlag": null,
      "progress": 0,
      "assignee": null,
      "sortOrder": -3298535473152,
      "startDate": "2026-01-17T14:00:00.000+0000",
      "isFloating": false,
      "attachments": [],
      "completedUserId": 130208689,
      "columnId": "6940c3d51263b74715346bd9",
      "remindTime": null,
      "status": -1,
      "projectId": "6940c3d51263b74715346bd6",
      "kind": null,
      "createdTime": "2026-01-17T13:15:27.000+0000",
      "modifiedTime": "2026-01-17T13:16:05.000+0000",
      "title": "last test task",
      "completedTime": "2026-01-17T13:16:24.000+0000",
      "tags": [],
      "timeZone": "Europe/Istanbul",
      "content": "",
      "id": "696b8ba82c6f3277d1cc4341",
      "repeatTaskId": "696b8b6f2c6f3277d1cc4331"
    }
  ],
  "update": [
    {
      "items": [],
      "reminders": [
        {
          "id": "696b8b602c6f3277d1cc432f",
          "trigger": "TRIGGER:PT0S"
        }
      ],
      "exDate": [],
      "dueDate": "2026-01-18T15:00:00.000+0000",
      "priority": 0,
      "isAllDay": false,
      "repeatFrom": "2",
      "repeatFlag": "RRULE:FREQ=DAILY;INTERVAL=1",
      "focusSummaries": [],
      "progress": 0,
      "assignee": null,
      "sortOrder": -3298535473152,
      "startDate": "2026-01-18T14:00:00.000+0000",
      "isFloating": false,
      "columnId": "6940c3d51263b74715346bd9",
      "remindTime": null,
      "status": 0,
      "projectId": "6940c3d51263b74715346bd6",
      "kind": null,
      "etag": "q1j2ogah",
      "createdTime": "2026-01-17T13:15:27.000+0000",
      "modifiedTime": "2026-01-17T13:16:24.000+0000",
      "title": "last test task",
      "tags": [],
      "timeZone": "Europe/Istanbul",
      "content": "",
      "id": "696b8b6f2c6f3277d1cc4331"
    }
  ],
  "delete": [],
  "addAttachments": [],
  "updateAttachments": [],
  "deleteAttachments": []
}
```

#### Key Won't Do Fields
| Field | Type | Value | Description |
|-------|------|-------|-------------|
| **status** | integer | **-1** | Status code for "won't do" / abandoned task |
| **completedTime** | string (ISO 8601) | Timestamp | When the task was marked as won't do |
| **completedUserId** | integer | User ID | ID of the user who marked it as won't do |
| repeatTaskId | string | Task ID | Reference to parent recurring task (if applicable) |

### Response
#### Status: 200 OK
```json
{
  "id2etag": {
    "696b8b6f2c6f3277d1cc4331": "8nlrfpks",
    "696b8ba82c6f3277d1cc4341": "imm8jglm"
  },
  "id2error": {}
}
```

### Notes
- **Status Values**:
  - `0` = Active/Incomplete task
  - `2` = Completed task (normal completion)
  - `-1` = Won't Do / Abandoned task
- When marking a recurring task instance as "won't do", a new task record is created in the `add` array with `status: -1`
- The parent recurring task is updated in the `update` array to advance to the next occurrence
- The `repeatTaskId` field links the completed/abandoned instance to its parent recurring task

---

## Create Task (Basic)

### Description
Creates a new task with basic fields including recurring schedule using weekly RRULE with an end date.

### Source File
`ticktick.com_api_v2_batch_task_Archive [26-01-17 16-13-35].har`

### Request Body
```json
{
  "add": [
    {
      "items": [],
      "reminders": [
        {
          "id": "696b8a802c6f3277d1cc4311",
          "trigger": "TRIGGER:PT0S"
        }
      ],
      "exDate": [],
      "dueDate": null,
      "priority": 5,
      "isAllDay": false,
      "repeatFrom": "0",
      "repeatFlag": "RRULE:FREQ=WEEKLY;INTERVAL=1;WKST=SU;UNTIL=20260227;BYDAY=SA",
      "progress": 0,
      "assignee": null,
      "sortOrder": -1099512217600,
      "startDate": "2026-01-17T15:00:00.000+0000",
      "isFloating": false,
      "columnId": "6940c3d51263b74715346bd9",
      "status": 0,
      "projectId": "6940c3d51263b74715346bd6",
      "kind": null,
      "createdTime": "2026-01-17T13:12:28.000+0000",
      "modifiedTime": "2026-01-17T13:12:28.000+0000",
      "title": "test task",
      "tags": [],
      "timeZone": "Europe/Istanbul",
      "content": "",
      "id": "696b8abc2c6f3277d1cc431a"
    }
  ],
  "update": [],
  "delete": [],
  "addAttachments": [],
  "updateAttachments": [],
  "deleteAttachments": []
}
```

### Response
#### Status: 200 OK
```json
{
  "id2etag": {
    "696b8abc2c6f3277d1cc431a": "v9b9ap2e"
  },
  "id2error": {}
}
```

---

## Create Task with Daily Repeat (COUNT-based)

### Description
Creates a task with daily recurrence using COUNT to limit number of occurrences.

### Source File
`ticktick.com_api_v2_batch_task_Archive [26-01-17 16-14-47].har`

### Request Body
```json
{
  "add": [
    {
      "items": [],
      "reminders": [
        {
          "id": "696b8b192c6f3277d1cc4324",
          "trigger": "TRIGGER:PT0S"
        }
      ],
      "exDate": [],
      "dueDate": null,
      "priority": 0,
      "isAllDay": false,
      "repeatFrom": "1",
      "repeatFlag": "RRULE:FREQ=DAILY;INTERVAL=1;COUNT=30",
      "progress": 0,
      "assignee": null,
      "sortOrder": -2199023845376,
      "startDate": "2026-01-17T15:00:00.000+0000",
      "isFloating": false,
      "columnId": "6940c3d51263b74715346bd9",
      "status": 0,
      "projectId": "6940c3d51263b74715346bd6",
      "kind": null,
      "createdTime": "2026-01-17T13:14:39.000+0000",
      "modifiedTime": "2026-01-17T13:14:39.000+0000",
      "title": "another test task",
      "tags": [],
      "timeZone": "Europe/Istanbul",
      "content": "",
      "id": "696b8b3f2c6f3277d1cc4326"
    }
  ],
  "update": [],
  "delete": [],
  "addAttachments": [],
  "updateAttachments": [],
  "deleteAttachments": []
}
```

### Response
#### Status: 200 OK
```json
{
  "id2etag": {
    "696b8b3f2c6f3277d1cc4326": "nzjdpzpk"
  },
  "id2error": {}
}
```

### Notes on repeatFrom
| Value | Meaning |
|-------|---------|
| "0" | Repeat from due date |
| "1" | Repeat from start date |
| "2" | Repeat from completion date |

---

## Create Task with Daily Repeat (Indefinite)

### Description
Creates a task with indefinite daily recurrence (no end date or count limit).

### Source File
`ticktick.com_api_v2_batch_task_Archive [26-01-17 16-15-35].har`

### Request Body
```json
{
  "add": [
    {
      "items": [],
      "reminders": [
        {
          "id": "696b8b602c6f3277d1cc432f",
          "trigger": "TRIGGER:PT0S"
        }
      ],
      "exDate": [],
      "dueDate": "2026-01-17T15:00:00.000+0000",
      "priority": 0,
      "isAllDay": false,
      "repeatFrom": "2",
      "repeatFlag": "RRULE:FREQ=DAILY;INTERVAL=1",
      "progress": 0,
      "assignee": null,
      "sortOrder": -3298535473152,
      "startDate": "2026-01-17T14:00:00.000+0000",
      "isFloating": false,
      "columnId": "6940c3d51263b74715346bd9",
      "status": 0,
      "projectId": "6940c3d51263b74715346bd6",
      "kind": null,
      "createdTime": "2026-01-17T13:15:27.000+0000",
      "modifiedTime": "2026-01-17T13:15:27.000+0000",
      "title": "last test task",
      "tags": [],
      "timeZone": "Europe/Istanbul",
      "content": "",
      "id": "696b8b6f2c6f3277d1cc4331"
    }
  ],
  "update": [],
  "delete": [],
  "addAttachments": [],
  "updateAttachments": [],
  "deleteAttachments": []
}
```

### Response
#### Status: 200 OK
```json
{
  "id2etag": {
    "696b8b6f2c6f3277d1cc4331": "q1j2ogah"
  },
  "id2error": {}
}
```

---

## Complete Recurring Task Instance

### Description
When completing an instance of a recurring task, the API creates a new completed task record and updates the parent task to the next occurrence.

### Observed Behavior (from "mark as wont do" HAR)
The same pattern applies to both completing and marking as "won't do":

1. **Add array**: Contains a new task with:
   - Same properties as the original
   - `repeatFlag: null` (no longer repeating)
   - `status: 2` (completed) or `status: -1` (won't do)
   - `completedTime`: timestamp of completion
   - `completedUserId`: user who completed it
   - `repeatTaskId`: ID of the parent recurring task
   - New unique `id`

2. **Update array**: Contains the parent recurring task with:
   - Advanced `startDate` and `dueDate` to next occurrence
   - Updated `modifiedTime`
   - `status: 0` (still active)
   - Same `repeatFlag`

---

## Common Headers

All requests share these common headers:

| Header | Value | Description |
|--------|-------|-------------|
| Host | api.ticktick.com | API host |
| User-Agent | Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:146.0) Gecko/20100101 Firefox/146.0 | Browser user agent |
| Accept | application/json, text/plain, */* | Accepted response types |
| Accept-Language | en-US,en;q=0.5 | Language preference |
| Accept-Encoding | gzip, deflate, br, zstd | Compression support |
| Content-Type | application/json;charset=utf-8 | Request body type |
| Referer | https://ticktick.com/ | Referring page |
| Origin | https://ticktick.com | CORS origin |
| X-Device | JSON object | Device identification |
| hl | en_US | UI language |
| x-tz | Europe/Istanbul | User timezone |
| X-Csrftoken | Token string | CSRF protection token |
| traceid | Unique ID | Request trace ID |
| Cookie | Multiple cookies | Session and auth cookies |

### X-Device Header Format
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
  "websocket": "696b93b42c6f3230e0a4238f"
}
```

### Required Cookies for Authentication
| Cookie | Description |
|--------|-------------|
| t | Authentication token (long hex string) |
| SESSION | Session ID (base64 encoded) |
| _csrf_token | CSRF token for request validation |
| AWSALB | AWS load balancer sticky session |
| AWSALBCORS | AWS CORS load balancer cookie |

---

## Task Object Schema

### Complete Task Object
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| id | string | Yes | Unique task identifier (24-char hex) |
| projectId | string | Yes | Project/list ID the task belongs to |
| title | string | Yes | Task title/name |
| content | string | No | Task description/notes |
| status | integer | Yes | Task status: 0=active, 2=complete, -1=won't do |
| priority | integer | No | Priority level: 0=none, 1=low, 3=medium, 5=high |
| sortOrder | integer | No | Sort position (negative numbers, lower = higher) |
| columnId | string | No | Kanban column ID |
| tags | array | No | Array of tag strings |
| timeZone | string | No | Task timezone (e.g., "Europe/Istanbul") |
| isAllDay | boolean | No | Whether this is an all-day task |
| isFloating | boolean | No | Whether time is floating (no timezone) |
| startDate | string | No | Start datetime (ISO 8601) |
| dueDate | string | No | Due datetime (ISO 8601) |
| createdTime | string | Yes | Creation timestamp (ISO 8601) |
| modifiedTime | string | Yes | Last modification timestamp (ISO 8601) |
| completedTime | string | No | Completion timestamp (ISO 8601) |
| completedUserId | integer | No | User ID who completed the task |
| repeatFlag | string | No | RRULE recurrence rule |
| repeatFrom | string | No | Repeat calculation basis: "0"=due, "1"=start, "2"=completed |
| repeatTaskId | string | No | Parent recurring task ID (for completed instances) |
| exDate | array | No | Exception dates for recurrence |
| reminders | array | No | Array of reminder objects |
| items | array | No | Subtasks/checklist items |
| assignee | string | No | Assigned user ID |
| progress | integer | No | Progress percentage (0-100) |
| kind | string | No | Task kind/type |
| etag | string | Required for updates | Entity tag for concurrency control |
| attachments | array | No | Array of attachment objects |
| focusSummaries | array | No | Array of focus session summaries |
| remindTime | string | No | Specific remind time |
| **pinnedTime** | string | No | **PIN FIELD**: ISO 8601 timestamp when pinned, null when unpinned |

### Reminder Object Schema
| Field | Type | Description |
|-------|------|-------------|
| id | string | Unique reminder identifier |
| trigger | string | iCal TRIGGER format (e.g., "TRIGGER:PT0S" for at time) |

### Trigger Format Examples
| Trigger | Meaning |
|---------|---------|
| TRIGGER:PT0S | At the scheduled time |
| TRIGGER:-PT5M | 5 minutes before |
| TRIGGER:-PT1H | 1 hour before |
| TRIGGER:-P1D | 1 day before |

---

## Key Field Analysis

### Priority Values
| Value | Level |
|-------|-------|
| 0 | None |
| 1 | Low |
| 3 | Medium |
| 5 | High |

### Status Values
| Value | Meaning |
|-------|---------|
| 0 | Active/Incomplete |
| 2 | Completed |
| -1 | Won't Do / Abandoned |

### repeatFrom Values
| Value | Behavior |
|-------|----------|
| "0" | Calculate next occurrence from due date |
| "1" | Calculate next occurrence from start date |
| "2" | Calculate next occurrence from completion date |

### RRULE Examples (repeatFlag)
| RRULE | Description |
|-------|-------------|
| `RRULE:FREQ=DAILY;INTERVAL=1` | Every day, indefinitely |
| `RRULE:FREQ=DAILY;INTERVAL=1;COUNT=30` | Every day for 30 occurrences |
| `RRULE:FREQ=WEEKLY;INTERVAL=1;WKST=SU;UNTIL=20260227;BYDAY=SA` | Every Saturday until Feb 27, 2026 |

### sortOrder
- Uses large negative numbers
- Lower (more negative) values appear first
- Examples observed: -1099512217600, -2199023845376, -3298535473152

---

## Implementation Notes for SDK

### To Pin a Task
```python
# Include pinnedTime in the update payload
update_data = {
    "update": [{
        "id": task_id,
        "pinnedTime": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000+0000"),
        "etag": current_etag,
        # ... other required fields
    }]
}
```

### To Unpin a Task
```python
# Set pinnedTime to null
update_data = {
    "update": [{
        "id": task_id,
        "pinnedTime": None,  # or null in JSON
        "etag": current_etag,
        # ... other required fields
    }]
}
```

### To Mark Task as Won't Do
```python
# Set status to -1 and include completion metadata
update_data = {
    "update": [{
        "id": task_id,
        "status": -1,
        "completedTime": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000+0000"),
        "completedUserId": user_id,
        "etag": current_etag,
        # ... other required fields
    }]
}
```

### Fields Not Previously Tracked (Potentially Missing from SDK)
1. **pinnedTime** - Critical for pin functionality
2. **completedUserId** - User who completed the task
3. **repeatTaskId** - Links completed instances to parent recurring task
4. **focusSummaries** - Focus/Pomodoro session data
5. **remindTime** - Specific reminder time
6. **attachments** - Task attachments array
7. **columnId** - Kanban board column assignment
8. **kind** - Task kind/type classification
9. **isFloating** - Floating time indicator
