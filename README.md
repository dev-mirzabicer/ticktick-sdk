# ticktick-mcp

A comprehensive async Python library for [TickTick](https://ticktick.com) with [MCP](https://modelcontextprotocol.io/) (Model Context Protocol) server support.

## Features

- **Full-Featured Python Client**: Async API for tasks, projects, tags, folders, focus tracking, habits, and more
- **MCP Server**: 36 tools for AI assistant integration (Claude, etc.)
- **Unified V1/V2 API**: Combines official OAuth2 API with unofficial session API for maximum functionality
- **Type-Safe**: Full Pydantic v2 validation with comprehensive type hints
- **Well-Tested**: 290+ tests covering both mock and live API interactions

## Why Both APIs?

TickTick has two APIs with different capabilities:

| Feature | V1 (OAuth2) | V2 (Session) |
|---------|-------------|--------------|
| Official/Documented | Yes | No |
| Tags | No | Yes |
| Folders (Project Groups) | No | Yes |
| Focus/Pomodoro Data | No | Yes |
| User Statistics | No | Yes |
| Move Tasks | No | Yes |
| Subtasks | Limited | Full |
| Get Project with Tasks | Yes | No |

This library combines both APIs to give you **everything**.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/dev-mirzabicer/ticktick-mcp.git
cd ticktick-mcp

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .

# For development
pip install -e ".[dev]"
```

---

## Authentication Setup

This library requires **both** V1 and V2 authentication for full functionality.

### Step 1: Register Your App (V1 OAuth2)

1. Go to the [TickTick Developer Portal](https://developer.ticktick.com/manage)
2. Click **"Create App"**
3. Fill in:
   - **App Name**: e.g., "My TickTick App"
   - **Redirect URI**: `http://127.0.0.1:8080/callback`
4. Save your **Client ID** and **Client Secret**

### Step 2: Create Your .env File

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
# V1 API (OAuth2) - REQUIRED
TICKTICK_CLIENT_ID=your_client_id_here
TICKTICK_CLIENT_SECRET=your_client_secret_here
TICKTICK_REDIRECT_URI=http://127.0.0.1:8080/callback
TICKTICK_ACCESS_TOKEN=  # Will be filled in Step 3

# V2 API (Session) - REQUIRED
TICKTICK_USERNAME=your_ticktick_email@example.com
TICKTICK_PASSWORD=your_ticktick_password

# Optional
TICKTICK_TIMEOUT=30
```

### Step 3: Get Your OAuth2 Access Token

Run the included helper script:

```bash
# Activate your virtual environment first
source .venv/bin/activate

# Run the OAuth flow
python scripts/get_oauth_token.py
```

This will:
1. Open your browser to TickTick's authorization page
2. Wait for you to authorize the app
3. Print the access token

**Copy the access token** to your `.env` file:

```bash
TICKTICK_ACCESS_TOKEN=paste_your_token_here
```

> **SSH/Headless Users**: Use `python scripts/get_oauth_token.py --manual` for a text-based flow.

### Step 4: Verify Setup

```bash
# Run a quick test
python -c "
import asyncio
from ticktick_mcp import TickTickClient

async def test():
    async with TickTickClient.from_settings() as client:
        profile = await client.get_profile()
        print(f'Connected as: {profile.display_name}')

asyncio.run(test())
"
```

---

## Usage: Python Library

### Quick Start

```python
import asyncio
from ticktick_mcp import TickTickClient

async def main():
    async with TickTickClient.from_settings() as client:
        # List all tasks
        tasks = await client.get_all_tasks()
        for task in tasks[:5]:
            print(f"- {task.title}")

asyncio.run(main())
```

### Creating Tasks

```python
from datetime import datetime, timedelta
from ticktick_mcp import TickTickClient

async with TickTickClient.from_settings() as client:
    # Simple task
    task = await client.create_task(title="Buy groceries")

    # Task with due date and priority
    task = await client.create_task(
        title="Submit report",
        due_date=datetime.now() + timedelta(days=1),
        priority="high",  # none, low, medium, high
    )

    # Task with tags
    task = await client.create_task(
        title="Review PR",
        tags=["work", "urgent"],
    )

    # Recurring task (MUST include start_date!)
    task = await client.create_task(
        title="Daily standup",
        start_date=datetime(2025, 1, 20, 9, 0),
        recurrence="RRULE:FREQ=DAILY;BYDAY=MO,TU,WE,TH,FR",
    )

    # Task with reminder
    task = await client.create_task(
        title="Meeting with team",
        due_date=datetime(2025, 1, 20, 14, 0),
        reminders=["TRIGGER:-PT15M"],  # 15 minutes before
    )
```

### Managing Tasks

```python
async with TickTickClient.from_settings() as client:
    # Get a specific task
    task = await client.get_task(task_id="...")

    # Update a task
    task.title = "Updated title"
    task.priority = 5  # high
    await client.update_task(task)

    # Complete a task
    await client.complete_task(task_id="...", project_id="...")

    # Delete a task
    await client.delete_task(task_id="...", project_id="...")

    # Move task to another project
    await client.move_task(
        task_id="...",
        from_project_id="...",
        to_project_id="...",
    )

    # Make a task a subtask
    await client.make_subtask(
        task_id="child_task_id",
        parent_id="parent_task_id",
        project_id="...",
    )
```

### Querying Tasks

```python
async with TickTickClient.from_settings() as client:
    # All active tasks
    all_tasks = await client.get_all_tasks()

    # Tasks due today
    today_tasks = await client.get_today_tasks()

    # Overdue tasks
    overdue = await client.get_overdue_tasks()

    # Tasks by tag
    work_tasks = await client.get_tasks_by_tag("work")

    # Tasks by priority
    high_priority = await client.get_tasks_by_priority("high")

    # Search tasks
    results = await client.search_tasks("meeting")

    # Recently completed tasks
    completed = await client.get_completed_tasks(days=7, limit=50)
```

### Projects

```python
async with TickTickClient.from_settings() as client:
    # List all projects
    projects = await client.get_all_projects()

    # Get a project with its tasks
    project_data = await client.get_project_tasks(project_id="...")
    print(f"Project: {project_data.project.name}")
    print(f"Tasks: {len(project_data.tasks)}")

    # Create a project
    project = await client.create_project(
        name="New Project",
        color="#F18181",
        view_mode="kanban",  # list, kanban, timeline
    )

    # Delete a project
    await client.delete_project(project_id="...")
```

### Folders (Project Groups)

```python
async with TickTickClient.from_settings() as client:
    # List all folders
    folders = await client.get_all_folders()

    # Create a folder
    folder = await client.create_folder(name="Work Projects")

    # Create project in folder
    project = await client.create_project(
        name="Q1 Goals",
        folder_id=folder.id,
    )

    # Delete a folder
    await client.delete_folder(folder_id="...")
```

### Tags

```python
async with TickTickClient.from_settings() as client:
    # List all tags
    tags = await client.get_all_tags()

    # Create a tag
    tag = await client.create_tag(
        name="urgent",
        color="#FF0000",
    )

    # Create nested tag
    child_tag = await client.create_tag(
        name="critical",
        parent="urgent",
    )

    # Rename a tag
    await client.rename_tag(old_name="urgent", new_name="priority")

    # Update a tag (change color or parent)
    await client.update_tag(name="urgent", color="#FF5500")

    # Merge tags
    await client.merge_tags(source="old-tag", target="new-tag")

    # Delete a tag
    await client.delete_tag(name="obsolete")
```

### User Information

```python
async with TickTickClient.from_settings() as client:
    # User profile
    profile = await client.get_profile()
    print(f"Username: {profile.username}")
    print(f"Display Name: {profile.display_name}")

    # Account status (Pro, inbox ID, etc.)
    status = await client.get_status()
    print(f"Is Pro: {status.is_pro}")
    print(f"Inbox ID: {status.inbox_id}")

    # Productivity statistics
    stats = await client.get_statistics()
    print(f"Level: {stats.level}")
    print(f"Score: {stats.score}")
    print(f"Tasks completed today: {stats.today_completed}")
    print(f"Total pomodoros: {stats.total_pomo_count}")

    # User preferences
    prefs = await client.get_preferences()
    print(f"Timezone: {prefs.get('timeZone')}")
    print(f"Week starts on: {prefs.get('weekStartDay')}")  # 0=Sunday
```

### Focus/Pomodoro Data

```python
from datetime import date, timedelta

async with TickTickClient.from_settings() as client:
    # Focus heatmap
    heatmap = await client.get_focus_heatmap(
        start_date=date.today() - timedelta(days=30),
        end_date=date.today(),
    )

    # Focus time by tag
    by_tag = await client.get_focus_by_tag(days=30)
    for tag, seconds in by_tag.items():
        hours = seconds / 3600
        print(f"{tag}: {hours:.1f} hours")
```

### Habits

```python
async with TickTickClient.from_settings() as client:
    # Get habits from sync
    state = await client.sync()
    habits = state.get("habits", [])
    for habit in habits:
        print(f"Habit: {habit.get('name')} (ID: {habit.get('id')})")

    # Get habit check-ins
    if habits:
        habit_ids = [h["id"] for h in habits[:3]]
        checkins = await client.get_habit_checkins(habit_ids)
        print(f"Check-ins: {checkins}")
```

### Advanced Task Queries

```python
async with TickTickClient.from_settings() as client:
    # Get abandoned tasks ("won't do")
    abandoned = await client.get_abandoned_tasks(days=30)
    print(f"Abandoned tasks: {len(abandoned)}")

    # Get deleted tasks (in trash)
    deleted = await client.get_deleted_tasks(limit=50)
    print(f"Deleted tasks: {len(deleted)}")

    # Unparent a subtask (make it top-level)
    await client.unparent_subtask(task_id="...", project_id="...")
```

### Full Sync

```python
async with TickTickClient.from_settings() as client:
    # Get complete account state
    state = await client.sync()

    projects = state.get("projectProfiles", [])
    tasks = state.get("syncTaskBean", {}).get("update", [])
    tags = state.get("tags", [])

    print(f"Projects: {len(projects)}")
    print(f"Tasks: {len(tasks)}")
    print(f"Tags: {len(tags)}")
```

### Error Handling

```python
from ticktick_mcp import (
    TickTickClient,
    TickTickError,
    TickTickNotFoundError,
    TickTickAuthenticationError,
    TickTickRateLimitError,
)

async with TickTickClient.from_settings() as client:
    try:
        task = await client.get_task("nonexistent-id")
    except TickTickNotFoundError:
        print("Task not found")
    except TickTickAuthenticationError:
        print("Authentication failed - check credentials")
    except TickTickRateLimitError:
        print("Rate limited - wait and retry")
    except TickTickError as e:
        print(f"TickTick error: {e}")
```

---

## Usage: MCP Server

The MCP server enables AI assistants (like Claude) to manage your TickTick tasks.

### Running the Server

```bash
# Make sure your .env is configured
ticktick-mcp
```

### Claude Desktop Integration

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "ticktick": {
      "command": "/path/to/ticktick-mcp/.venv/bin/python",
      "args": ["-m", "ticktick_mcp.server"],
      "env": {
        "TICKTICK_CLIENT_ID": "your_client_id",
        "TICKTICK_CLIENT_SECRET": "your_client_secret",
        "TICKTICK_ACCESS_TOKEN": "your_access_token",
        "TICKTICK_USERNAME": "your_email",
        "TICKTICK_PASSWORD": "your_password"
      }
    }
  }
}
```

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `ticktick_create_task` | Create a new task |
| `ticktick_get_task` | Get task details |
| `ticktick_list_tasks` | List tasks with filters |
| `ticktick_update_task` | Update task properties |
| `ticktick_complete_task` | Mark task complete |
| `ticktick_delete_task` | Delete a task |
| `ticktick_move_task` | Move task between projects |
| `ticktick_make_subtask` | Create parent-child relationship |
| `ticktick_unparent_subtask` | Remove parent-child relationship |
| `ticktick_completed_tasks` | List recently completed tasks |
| `ticktick_abandoned_tasks` | List abandoned ("won't do") tasks |
| `ticktick_deleted_tasks` | List deleted tasks (in trash) |
| `ticktick_search_tasks` | Search tasks by text |
| `ticktick_list_projects` | List all projects |
| `ticktick_get_project` | Get project details |
| `ticktick_create_project` | Create a new project |
| `ticktick_update_project` | Update project properties |
| `ticktick_delete_project` | Delete a project |
| `ticktick_list_folders` | List all folders |
| `ticktick_create_folder` | Create a folder |
| `ticktick_rename_folder` | Rename a folder |
| `ticktick_delete_folder` | Delete a folder |
| `ticktick_list_tags` | List all tags |
| `ticktick_create_tag` | Create a tag |
| `ticktick_update_tag` | Update tag properties |
| `ticktick_delete_tag` | Delete a tag |
| `ticktick_rename_tag` | Rename a tag |
| `ticktick_merge_tags` | Merge two tags |
| `ticktick_get_profile` | Get user profile |
| `ticktick_get_status` | Get account status |
| `ticktick_get_statistics` | Get productivity stats |
| `ticktick_get_preferences` | Get user preferences |
| `ticktick_focus_heatmap` | Get focus heatmap |
| `ticktick_focus_by_tag` | Get focus time by tag |
| `ticktick_habit_checkins` | Get habit check-in data |
| `ticktick_sync` | Full account sync |

---

## Important: TickTick API Quirks

TickTick's API has several unique behaviors you should know about:

### 1. Recurrence Requires start_date

**If you create a recurring task without a start_date, TickTick silently ignores the recurrence rule.**

```python
# WRONG - recurrence will be ignored!
task = await client.create_task(
    title="Daily standup",
    recurrence="RRULE:FREQ=DAILY",
)

# CORRECT
task = await client.create_task(
    title="Daily standup",
    start_date=datetime(2025, 1, 20, 9, 0),
    recurrence="RRULE:FREQ=DAILY",
)
```

### 2. Subtasks Require Separate Call

Setting `parent_id` during task creation is **ignored** by the API. Create the task first, then make it a subtask:

```python
# Create the child task
child = await client.create_task(title="Subtask")

# Then make it a subtask
await client.make_subtask(
    task_id=child.id,
    parent_id="parent_task_id",
    project_id=child.project_id,
)
```

### 3. Soft Delete

Deleting tasks moves them to trash (sets `deleted=1`) rather than permanently removing them. Deleted tasks remain accessible via `get_task`.

### 4. Date Clearing

To clear a task's `due_date`, you must also clear `start_date`. If you only clear `due_date`, TickTick will restore it from `start_date`.

```python
# Clear both dates together
task.due_date = None
task.start_date = None
await client.update_task(task)
```

### 5. Tag Order Not Preserved

The API does not preserve tag order - tags may be returned in any order.

### 6. Inbox is Special

The inbox is a special project that cannot be deleted. Get its ID via `client.inbox_id` or `await client.get_status()`.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `TICKTICK_CLIENT_ID` | Yes | OAuth2 client ID from developer portal |
| `TICKTICK_CLIENT_SECRET` | Yes | OAuth2 client secret |
| `TICKTICK_ACCESS_TOKEN` | Yes | OAuth2 access token (from setup script) |
| `TICKTICK_USERNAME` | Yes | Your TickTick email |
| `TICKTICK_PASSWORD` | Yes | Your TickTick password |
| `TICKTICK_REDIRECT_URI` | No | OAuth2 redirect URI (default: `http://127.0.0.1:8080/callback`) |
| `TICKTICK_TIMEOUT` | No | Request timeout in seconds (default: `30`) |
| `TICKTICK_DEVICE_ID` | No | Device ID for V2 API (auto-generated) |

---

## Running Tests

```bash
# All tests (mock mode - no API calls)
pytest

# With verbose output
pytest -v

# Specific test file
pytest tests/test_client_tasks.py

# Live tests (requires .env with valid credentials)
pytest --live

# With coverage
pytest --cov=ticktick_mcp --cov-report=term-missing
```

---

## API Reference

### Models

| Model | Description |
|-------|-------------|
| `Task` | Task with title, dates, priority, tags, subtasks, etc. |
| `Project` | Project/list container for tasks |
| `ProjectGroup` | Folder for organizing projects |
| `Tag` | Tag with name, color, and optional parent |
| `User` | User profile information |
| `UserStatus` | Account status (Pro, inbox ID, etc.) |
| `UserStatistics` | Productivity statistics |
| `ChecklistItem` | Subtask/checklist item within a task |

### Enums

| Enum | Values |
|------|--------|
| `TaskStatus` | `ABANDONED (-1)`, `ACTIVE (0)`, `COMPLETED (2)` |
| `TaskPriority` | `NONE (0)`, `LOW (1)`, `MEDIUM (3)`, `HIGH (5)` |
| `TaskKind` | `TEXT`, `NOTE`, `CHECKLIST` |
| `ProjectKind` | `TASK`, `NOTE` |
| `ViewMode` | `LIST`, `KANBAN`, `TIMELINE` |

### Exceptions

| Exception | Description |
|-----------|-------------|
| `TickTickError` | Base exception for all errors |
| `TickTickAuthenticationError` | Authentication failed |
| `TickTickNotFoundError` | Resource not found |
| `TickTickValidationError` | Invalid input data |
| `TickTickRateLimitError` | Rate limit exceeded |
| `TickTickConfigurationError` | Missing configuration |
| `TickTickForbiddenError` | Access denied |
| `TickTickServerError` | Server-side error |

---

## Troubleshooting

### "Token exchange failed"
- Verify your Client ID and Client Secret are correct
- Ensure the Redirect URI matches exactly (including trailing slashes)

### "Authentication failed"
- Check your TickTick username and password
- Try logging into ticktick.com to verify credentials

### "V2 initialization failed"
- Your password may contain special characters that need escaping
- Check for 2FA (not currently supported)

### "Rate limit exceeded"
- Wait 30-60 seconds before retrying
- Reduce the frequency of API calls

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass (`pytest`)
5. Submit a pull request

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [TickTick](https://ticktick.com) for the excellent task management app
- [Model Context Protocol](https://modelcontextprotocol.io/) for the AI integration standard
- [FastMCP](https://github.com/anthropics/anthropic-sdk-python) for the MCP framework
