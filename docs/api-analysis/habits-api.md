# TickTick Habits API Documentation

This document provides comprehensive API documentation for TickTick's Habits feature, reverse-engineered from HAR file analysis.

## Table of Contents
1. [Overview](#overview)
2. [Authentication & Headers](#authentication--headers)
3. [Endpoints](#endpoints)
   - [Create Habit](#create-habit)
   - [Update Habit](#update-habit)
   - [Habit Check-in (Two-Request Pattern)](#habit-check-in-two-request-pattern)
4. [Data Models](#data-models)
5. [Workflow Sequences](#workflow-sequences)

---

## Overview

The TickTick Habits API uses two primary endpoints:
- `/api/v2/habits/batch` - For creating and updating habit definitions
- `/api/v2/habitCheckins/batch` - For recording habit check-ins/progress

**CRITICAL OBSERVATION**: Habit check-ins require a **TWO-REQUEST PATTERN**. A single user action to check in a habit triggers:
1. A request to `/api/v2/habits/batch` to update the habit metadata (totalCheckIns, currentStreak, etc.)
2. A request to `/api/v2/habitCheckins/batch` to record the actual check-in entry

Both endpoints follow a batch operation structure supporting `add`, `update`, and `delete` arrays.

---

## Authentication & Headers

### Required Headers

| Header | Value | Required | Description |
|--------|-------|----------|-------------|
| `Host` | `api.ticktick.com` | Yes | API host |
| `Content-Type` | `application/json;charset=utf-8` | Yes | Request content type |
| `X-Device` | JSON object (see below) | Yes | Device identification |
| `hl` | `en_US` | Yes | Language/locale code |
| `x-tz` | `Europe/Istanbul` | Yes | Timezone (IANA format) |
| `X-Csrftoken` | CSRF token string | Yes | CSRF protection token |
| `traceid` | Unique ID per request | Yes | Request tracing ID |
| `Cookie` | Session cookies | Yes | Authentication cookies |
| `Origin` | `https://ticktick.com` | Yes | CORS origin |
| `Referer` | `https://ticktick.com/` | Yes | Referer header |

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
| `platform` | string | Platform type: "web", "ios", "android" |
| `os` | string | Operating system info |
| `device` | string | Browser/device name and version |
| `name` | string | Device name (empty for web) |
| `version` | integer | Client version number |
| `id` | string | Unique device ID |
| `channel` | string | Installation channel |
| `campaign` | string | Marketing campaign (if any) |
| `websocket` | string | WebSocket session ID |

### Authentication Cookies

| Cookie | Description |
|--------|-------------|
| `t` | Primary authentication token (long encrypted string) |
| `SESSION` | Session ID (base64 encoded UUID) |
| `_csrf_token` | CSRF token for request validation |
| `AWSALB` / `AWSALBCORS` | AWS load balancer cookies |

---

## Endpoints

---

## Create Habit

### Endpoint
`POST /api/v2/habits/batch`

### Description
Creates a new habit. Uses the `add` array in the batch request body.

### Request

#### Headers
See [Authentication & Headers](#authentication--headers)

#### Query Parameters
None

#### Body
```json
{
  "add": [
    {
      "color": "#97E38B",
      "iconRes": "habit_daily_check_in",
      "createdTime": "2026-01-17T13:36:16.000+0000",
      "encouragement": "",
      "etag": "",
      "goal": 5,
      "id": "696b90502c6f3277d1cc44a9",
      "modifiedTime": "2026-01-17T13:36:16.000+0000",
      "name": "Some new habit",
      "recordEnable": false,
      "reminders": ["20:00"],
      "repeatRule": "RRULE:FREQ=DAILY;INTERVAL=2",
      "sortOrder": -1099511627776,
      "status": 0,
      "step": 1,
      "totalCheckIns": 0,
      "type": "Real",
      "unit": "Count",
      "sectionId": "6940c47c16b2c87db11a567f",
      "targetDays": 100,
      "targetStartDate": 20260117,
      "completedCycles": 0,
      "exDates": [],
      "currentStreak": 0,
      "style": 1
    }
  ],
  "update": [],
  "delete": []
}
```

#### Body Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `add` | array | Yes | Array of habit objects to create |
| `update` | array | Yes | Array of habit objects to update (empty for create) |
| `delete` | array | Yes | Array of habit IDs to delete (empty for create) |

#### Habit Object Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique habit ID (client-generated) |
| `name` | string | Yes | Habit name/title |
| `color` | string | Yes | Hex color code (e.g., "#97E38B") |
| `iconRes` | string | Yes | Icon resource name (e.g., "habit_daily_check_in") |
| `goal` | integer | Yes | Target value to achieve per check-in period |
| `step` | integer | Yes | Increment value per check-in action |
| `unit` | string | Yes | Measurement unit (e.g., "Count") |
| `type` | string | Yes | Habit type: "Real" |
| `status` | integer | Yes | Habit status: 0 = active |
| `style` | integer | Yes | Display style: 1 |
| `repeatRule` | string | Yes | iCalendar RRULE format (e.g., "RRULE:FREQ=DAILY;INTERVAL=2") |
| `reminders` | array | No | Array of reminder times in "HH:MM" format |
| `sectionId` | string | No | ID of the habit section/group |
| `targetDays` | integer | No | Total target days for the habit goal |
| `targetStartDate` | integer | No | Start date in YYYYMMDD format |
| `sortOrder` | integer | Yes | Sort order (negative large number) |
| `totalCheckIns` | integer | Yes | Total number of check-ins (0 for new) |
| `completedCycles` | integer | Yes | Number of completed cycles (0 for new) |
| `currentStreak` | integer | Yes | Current streak count (0 for new) |
| `exDates` | array | No | Array of excluded dates |
| `recordEnable` | boolean | No | Whether records are enabled |
| `encouragement` | string | No | Encouragement message |
| `etag` | string | No | Entity tag for optimistic concurrency (empty for new) |
| `createdTime` | string | Yes | ISO 8601 timestamp with timezone |
| `modifiedTime` | string | Yes | ISO 8601 timestamp with timezone |

### Response

#### Status: 200 OK
```json
{
  "id2etag": {
    "696b90502c6f3277d1cc44a9": "0wmx5gyn"
  },
  "id2error": {}
}
```

#### Response Schema

| Field | Type | Description |
|-------|------|-------------|
| `id2etag` | object | Map of habit ID to new etag value |
| `id2error` | object | Map of habit ID to error message (empty on success) |

### Notes
- The `id` is client-generated before sending the request
- The `etag` is returned by the server and must be used in subsequent updates
- `repeatRule` follows iCalendar RRULE specification
- `sortOrder` uses large negative numbers for ordering

---

## Update Habit

### Endpoint
`POST /api/v2/habits/batch`

### Description
Updates an existing habit. Uses the `update` array in the batch request body.

### Request

#### Headers
See [Authentication & Headers](#authentication--headers)

#### Query Parameters
None

#### Body
```json
{
  "add": [],
  "update": [
    {
      "color": "#97E38B",
      "iconRes": "habit_daily_check_in",
      "createdTime": "2026-01-17T13:36:16.000+0000",
      "encouragement": "",
      "etag": "",
      "goal": 5,
      "id": "696b90502c6f3277d1cc44a9",
      "modifiedTime": "2026-01-17T13:37:49.000+0000",
      "name": "Some new habit",
      "recordEnable": false,
      "reminders": ["20:00"],
      "repeatRule": "RRULE:FREQ=DAILY;INTERVAL=2",
      "sortOrder": -1099511627776,
      "status": 0,
      "step": 1,
      "totalCheckIns": 0,
      "type": "Real",
      "unit": "Count",
      "sectionId": "6940c47c16b2c87db11a567f",
      "targetDays": 100,
      "targetStartDate": 20260117,
      "completedCycles": 0,
      "exDates": [],
      "currentStreak": 0,
      "style": 1
    }
  ],
  "delete": []
}
```

### Response

#### Status: 200 OK
```json
{
  "id2etag": {
    "696b90502c6f3277d1cc44a9": "qaf97ag9"
  },
  "id2error": {}
}
```

### Notes
- The `etag` should be included from the previous response for optimistic concurrency
- All fields must be sent (full object replacement, not partial update)
- `modifiedTime` should be updated to the current time

---

## Habit Check-in (Two-Request Pattern)

### Overview

**IMPORTANT**: A habit check-in requires TWO sequential API calls:

1. **Request 1**: Update habit metadata via `/api/v2/habits/batch`
2. **Request 2**: Record the check-in via `/api/v2/habitCheckins/batch`

This pattern is observed both for initial partial check-ins and for completing a habit goal.

---

### Phase 1: Partial Check-in (Increment Progress)

When a user clicks to increment habit progress (but hasn't yet reached the goal):

#### Request 1: Update Habit Metadata

**Endpoint**: `POST /api/v2/habits/batch`

**Body**:
```json
{
  "add": [],
  "update": [
    {
      "color": "#97E38B",
      "iconRes": "habit_daily_check_in",
      "createdTime": "2026-01-17T13:36:16.000+0000",
      "encouragement": "",
      "etag": "",
      "goal": 5,
      "id": "696b90502c6f3277d1cc44a9",
      "modifiedTime": "2026-01-17T13:37:49.000+0000",
      "name": "Some new habit",
      "recordEnable": false,
      "reminders": ["20:00"],
      "repeatRule": "RRULE:FREQ=DAILY;INTERVAL=2",
      "sortOrder": -1099511627776,
      "status": 0,
      "step": 1,
      "totalCheckIns": 0,
      "type": "Real",
      "unit": "Count",
      "sectionId": "6940c47c16b2c87db11a567f",
      "targetDays": 100,
      "targetStartDate": 20260117,
      "completedCycles": 0,
      "exDates": [],
      "currentStreak": 0,
      "style": 1
    }
  ],
  "delete": []
}
```

**Response**:
```json
{
  "id2etag": {
    "696b90502c6f3277d1cc44a9": "qaf97ag9"
  },
  "id2error": {}
}
```

#### Request 2: Create Check-in Record

**Endpoint**: `POST /api/v2/habitCheckins/batch`

**Body**:
```json
{
  "add": [
    {
      "checkinStamp": 20260116,
      "opTime": "2026-01-17T13:37:49.000+0000",
      "goal": 5,
      "habitId": "696b90502c6f3277d1cc44a9",
      "id": "696b90862c6f3277d1cc451e",
      "status": 0,
      "value": 1
    }
  ],
  "update": [],
  "delete": []
}
```

#### Check-in Object Schema (Add - Initial)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique check-in ID (client-generated) |
| `habitId` | string | Yes | ID of the parent habit |
| `checkinStamp` | integer | Yes | Date stamp in YYYYMMDD format |
| `opTime` | string | Yes | Operation time (ISO 8601) |
| `goal` | integer | Yes | The goal value from the habit |
| `value` | integer | Yes | Current progress value |
| `status` | integer | Yes | 0 = in progress, 2 = completed |

**Response**:
```json
{
  "id2etag": {},
  "id2error": {}
}
```

---

### Phase 2: Complete Check-in (Reach Goal)

When a user completes the habit goal (reaches target value):

#### Request 1: Update Habit Metadata (with updated stats)

**Endpoint**: `POST /api/v2/habits/batch`

**Body**:
```json
{
  "add": [],
  "update": [
    {
      "color": "#97E38B",
      "iconRes": "habit_daily_check_in",
      "createdTime": "2026-01-17T13:36:16.000+0000",
      "encouragement": "",
      "etag": "sln4qiy2",
      "goal": 5,
      "id": "696b90502c6f3277d1cc44a9",
      "modifiedTime": "2026-01-17T13:39:00.000+0000",
      "name": "Some new habit",
      "recordEnable": false,
      "reminders": ["20:00"],
      "repeatRule": "RRULE:FREQ=DAILY;INTERVAL=2",
      "sortOrder": -1099511627776,
      "status": 0,
      "step": 1,
      "totalCheckIns": 1,
      "type": "Real",
      "unit": "Count",
      "sectionId": "6940c47c16b2c87db11a567f",
      "targetDays": 100,
      "targetStartDate": 20260117,
      "completedCycles": 0,
      "exDates": [],
      "currentStreak": 1,
      "style": 1
    }
  ],
  "delete": []
}
```

**Key differences from partial check-in**:
- `etag` is populated with value from previous response
- `totalCheckIns` incremented from 0 to 1
- `currentStreak` incremented from 0 to 1
- `modifiedTime` updated

**Response**:
```json
{
  "id2etag": {
    "696b90502c6f3277d1cc44a9": "jitu6mpc"
  },
  "id2error": {}
}
```

#### Request 2: Update Check-in Record (Mark Complete)

**Endpoint**: `POST /api/v2/habitCheckins/batch`

**Body**:
```json
{
  "add": [],
  "update": [
    {
      "checkinStamp": 20260116,
      "checkinTime": "2026-01-17T13:39:00.000+0000",
      "opTime": "2026-01-17T13:39:00.000+0000",
      "goal": 5,
      "habitId": "696b90502c6f3277d1cc44a9",
      "id": "696b90862c6f3277d1cc451e",
      "status": 2,
      "value": 5
    }
  ],
  "delete": []
}
```

#### Check-in Object Schema (Update - Complete)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Check-in ID (from initial add) |
| `habitId` | string | Yes | ID of the parent habit |
| `checkinStamp` | integer | Yes | Date stamp in YYYYMMDD format |
| `checkinTime` | string | Yes | Time when check-in was completed (ISO 8601) |
| `opTime` | string | Yes | Operation time (ISO 8601) |
| `goal` | integer | Yes | The goal value |
| `value` | integer | Yes | Final value (should equal goal when complete) |
| `status` | integer | Yes | 2 = completed |

**Response**:
```json
{
  "id2etag": {},
  "id2error": {}
}
```

---

## Data Models

### Habit Object

```typescript
interface Habit {
  // Identification
  id: string;                    // Unique habit ID (client-generated)
  name: string;                  // Habit name

  // Appearance
  color: string;                 // Hex color code (e.g., "#97E38B")
  iconRes: string;               // Icon resource identifier
  style: number;                 // Display style (1)

  // Goal Configuration
  goal: number;                  // Target value per period
  step: number;                  // Increment per action
  unit: string;                  // Unit of measurement ("Count")
  type: string;                  // Habit type ("Real")

  // Schedule
  repeatRule: string;            // iCalendar RRULE format
  reminders: string[];           // Array of "HH:MM" times
  exDates: string[];             // Excluded dates

  // Target Tracking
  targetDays: number;            // Total target days
  targetStartDate: number;       // Start date (YYYYMMDD)

  // Statistics
  totalCheckIns: number;         // Total completed check-ins
  completedCycles: number;       // Completed goal cycles
  currentStreak: number;         // Current streak count

  // Organization
  sectionId: string;             // Parent section ID
  sortOrder: number;             // Sort order (negative number)

  // State
  status: number;                // 0 = active
  recordEnable: boolean;         // Records enabled
  encouragement: string;         // Encouragement message

  // Sync
  etag: string;                  // Optimistic concurrency tag
  createdTime: string;           // ISO 8601 timestamp
  modifiedTime: string;          // ISO 8601 timestamp
}
```

### HabitCheckin Object

```typescript
interface HabitCheckin {
  // Identification
  id: string;                    // Unique check-in ID (client-generated)
  habitId: string;               // Parent habit ID

  // Check-in Data
  checkinStamp: number;          // Date in YYYYMMDD format
  checkinTime?: string;          // Completion time (ISO 8601) - only when complete
  opTime: string;                // Operation time (ISO 8601)

  // Progress
  goal: number;                  // Target goal value
  value: number;                 // Current progress value
  status: number;                // 0 = in progress, 2 = completed
}
```

### BatchRequest Object

```typescript
interface BatchRequest<T> {
  add: T[];                      // Objects to create
  update: T[];                   // Objects to update
  delete: string[];              // IDs to delete
}
```

### BatchResponse Object

```typescript
interface BatchResponse {
  id2etag: Record<string, string>;   // Map of ID to new etag
  id2error: Record<string, string>;  // Map of ID to error message
}
```

---

## Workflow Sequences

### Creating a New Habit

```
1. Generate unique ID (client-side)
2. POST /api/v2/habits/batch
   - add: [habit object with all fields]
3. Store returned etag for future updates
```

### Checking In a Habit (Full Workflow)

```
Step 1: Initial Progress Increment
   1a. POST /api/v2/habits/batch
       - update: [habit with modifiedTime updated]
   1b. POST /api/v2/habitCheckins/batch
       - add: [checkin with status=0, value=step]

Step 2: Continue Incrementing (repeat until goal reached)
   2a. POST /api/v2/habits/batch
       - update: [habit with modifiedTime updated]
   2b. POST /api/v2/habitCheckins/batch
       - update: [checkin with value incremented, status=0]

Step 3: Complete (when value >= goal)
   3a. POST /api/v2/habits/batch
       - update: [habit with totalCheckIns++, currentStreak++, new etag]
   3b. POST /api/v2/habitCheckins/batch
       - update: [checkin with value=goal, status=2, checkinTime set]
```

### Timeline Example (from HAR files)

```
16:37:10 - Create new habit "Some new habit"
           POST /api/v2/habits/batch (add)

16:37:52 - First check-in increment (value: 0 -> 1)
           POST /api/v2/habits/batch (update habit metadata)
           POST /api/v2/habitCheckins/batch (add checkin)

16:39:03 - Complete check-in (value: 1 -> 5, status: 0 -> 2)
           POST /api/v2/habits/batch (update: totalCheckIns=1, currentStreak=1)
           POST /api/v2/habitCheckins/batch (update: value=5, status=2)
```

---

## Status Codes

### Habit Status
| Value | Meaning |
|-------|---------|
| 0 | Active |

### Check-in Status
| Value | Meaning |
|-------|---------|
| 0 | In Progress |
| 2 | Completed |

---

## ID Generation

IDs appear to be MongoDB ObjectId-like hex strings (24 characters). They are generated client-side before the request is sent.

Example: `696b90502c6f3277d1cc44a9`

---

## Date/Time Formats

| Format | Example | Usage |
|--------|---------|-------|
| ISO 8601 | `2026-01-17T13:36:16.000+0000` | createdTime, modifiedTime, opTime, checkinTime |
| YYYYMMDD | `20260117` | checkinStamp, targetStartDate |
| HH:MM | `20:00` | reminders |

---

## RRULE Format

The `repeatRule` field uses iCalendar RRULE specification:

Examples:
- `RRULE:FREQ=DAILY;INTERVAL=1` - Every day
- `RRULE:FREQ=DAILY;INTERVAL=2` - Every 2 days
- `RRULE:FREQ=WEEKLY;BYDAY=MO,WE,FR` - Monday, Wednesday, Friday

---

## Error Handling

Errors are returned in the `id2error` map in the response:

```json
{
  "id2etag": {},
  "id2error": {
    "696b90502c6f3277d1cc44a9": "Error message here"
  }
}
```

---

## SDK Implementation Notes

1. **Always send both requests**: A habit check-in is not complete without both the habits/batch and habitCheckins/batch requests.

2. **Maintain etag**: Store and update the etag from each response for subsequent updates.

3. **Generate IDs client-side**: Both habit and check-in IDs must be generated before the request.

4. **Send full objects**: Updates require the complete object, not just changed fields.

5. **Track statistics client-side**: When completing a check-in, update `totalCheckIns` and `currentStreak` on the habit object.

6. **Handle timezone**: The `x-tz` header and timestamps are important for correct date handling.

7. **Date stamp vs Date**: `checkinStamp` uses YYYYMMDD integer format, while other date fields use ISO 8601 strings.
