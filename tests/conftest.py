"""
Pytest Configuration and Fixtures for TickTick Client Tests.

This module provides comprehensive fixtures, mock factories, and shared
utilities for testing the TickTick Client.

Architecture:
    - MockUnifiedAPI: Async mock for UnifiedTickTickAPI
    - Factories: Generate test data (tasks, projects, tags, etc.)
    - Fixtures: Provide configured clients and mock data
    - Markers: Custom pytest markers for test categorization

Live Mode:
    Run tests against the real TickTick API with:
        pytest --live

    This requires a .env file with valid credentials.
    Tests marked with @pytest.mark.mock_only will be skipped in live mode.
    Tests marked with @pytest.mark.live_only will only run in live mode.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ticktick_sdk.client import TickTickClient
from ticktick_sdk.models import (
    Task,
    ChecklistItem,
    Project,
    ProjectGroup,
    ProjectData,
    Column,
    Tag,
    User,
    UserStatus,
    UserStatistics,
    Habit,
    HabitSection,
    HabitCheckin,
    HabitPreferences,
)
from ticktick_sdk.constants import TaskStatus, TaskPriority, ProjectKind, ViewMode


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run tests against real TickTick API (requires .env credentials)",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "tasks: Task-related tests")
    config.addinivalue_line("markers", "projects: Project-related tests")
    config.addinivalue_line("markers", "tags: Tag-related tests")
    config.addinivalue_line("markers", "user: User-related tests")
    config.addinivalue_line("markers", "focus: Focus/Pomodoro tests")
    config.addinivalue_line("markers", "habits: Habit-related tests")
    config.addinivalue_line("markers", "sync: Sync-related tests")
    config.addinivalue_line("markers", "errors: Error handling tests")
    config.addinivalue_line("markers", "lifecycle: Client lifecycle tests")
    config.addinivalue_line("markers", "mock_only: Tests that only work with mocks (skipped in --live mode)")
    config.addinivalue_line("markers", "live_only: Tests that only run in --live mode")


def pytest_collection_modifyitems(config, items):
    """Handle live/mock test filtering and event loop scope."""
    live_mode = config.getoption("--live")

    skip_in_live = pytest.mark.skip(reason="Test only works with mocks (use without --live)")
    skip_in_mock = pytest.mark.skip(reason="Test only works with real API (use --live)")

    # In live mode, use session-scoped event loop to share httpx client
    session_loop_marker = pytest.mark.asyncio(loop_scope="session")

    for item in items:
        if live_mode:
            # In live mode, skip mock_only tests
            if "mock_only" in item.keywords:
                item.add_marker(skip_in_live)
            # Use session-scoped event loop for live tests
            # This allows sharing the httpx client across tests
            if asyncio.iscoroutinefunction(item.obj):
                item.add_marker(session_loop_marker)
        else:
            # In mock mode, skip live_only tests
            if "live_only" in item.keywords:
                item.add_marker(skip_in_mock)


# =============================================================================
# Time Utilities
# =============================================================================


def utc_now() -> datetime:
    """Get current UTC time."""
    return datetime.now(timezone.utc)


def days_ago(n: int) -> datetime:
    """Get datetime n days ago."""
    return utc_now() - timedelta(days=n)


def days_from_now(n: int) -> datetime:
    """Get datetime n days from now."""
    return utc_now() + timedelta(days=n)


# =============================================================================
# ID Generators
# =============================================================================


class IDGenerator:
    """Thread-safe ID generator for test objects."""

    _counter: int = 0

    @classmethod
    def reset(cls) -> None:
        """Reset counter (call in fixtures)."""
        cls._counter = 0

    @classmethod
    def next_id(cls, prefix: str = "") -> str:
        """Generate next unique ID."""
        cls._counter += 1
        hex_part = f"{cls._counter:024x}"
        return f"{prefix}{hex_part}" if prefix else hex_part

    @classmethod
    def task_id(cls) -> str:
        """Generate task ID."""
        return cls.next_id()

    @classmethod
    def project_id(cls) -> str:
        """Generate project ID."""
        return cls.next_id()

    @classmethod
    def folder_id(cls) -> str:
        """Generate folder/project group ID."""
        return cls.next_id()

    @classmethod
    def inbox_id(cls) -> str:
        """Generate inbox ID."""
        cls._counter += 1
        return f"inbox{cls._counter}"


# =============================================================================
# Test Data Factories
# =============================================================================


class TaskFactory:
    """Factory for creating Task test objects."""

    @staticmethod
    def create(
        id: str | None = None,
        project_id: str | None = None,
        title: str = "Test Task",
        content: str | None = None,
        status: int = TaskStatus.ACTIVE,
        priority: int = TaskPriority.NONE,
        start_date: datetime | None = None,
        due_date: datetime | None = None,
        completed_time: datetime | None = None,
        tags: list[str] | None = None,
        items: list[ChecklistItem] | None = None,
        parent_id: str | None = None,
        child_ids: list[str] | None = None,
        time_zone: str = "America/Los_Angeles",
        is_all_day: bool = False,
        repeat_flag: str | None = None,
        **kwargs,
    ) -> Task:
        """Create a Task with sensible defaults."""
        return Task(
            id=id or IDGenerator.task_id(),
            project_id=project_id or IDGenerator.project_id(),
            title=title,
            content=content,
            status=status,
            priority=priority,
            start_date=start_date,
            due_date=due_date,
            completed_time=completed_time,
            tags=tags or [],
            items=items or [],
            parent_id=parent_id,
            child_ids=child_ids or [],
            time_zone=time_zone,
            is_all_day=is_all_day,
            repeat_flag=repeat_flag,
            created_time=utc_now(),
            modified_time=utc_now(),
            sort_order=0,
            **kwargs,
        )

    @staticmethod
    def create_with_due_date(days_offset: int = 1, **kwargs) -> Task:
        """Create task with due date relative to today."""
        due = days_from_now(days_offset) if days_offset >= 0 else days_ago(-days_offset)
        return TaskFactory.create(due_date=due, **kwargs)

    @staticmethod
    def create_completed(**kwargs) -> Task:
        """Create a completed task."""
        return TaskFactory.create(
            status=TaskStatus.COMPLETED,
            completed_time=utc_now(),
            **kwargs,
        )

    @staticmethod
    def create_overdue(**kwargs) -> Task:
        """Create an overdue task."""
        return TaskFactory.create(
            due_date=days_ago(3),
            status=TaskStatus.ACTIVE,
            **kwargs,
        )

    @staticmethod
    def create_with_subtasks(subtask_count: int = 3, **kwargs) -> Task:
        """Create task with checklist items."""
        items = [
            ChecklistItem(
                id=IDGenerator.task_id(),
                title=f"Subtask {i+1}",
                status=0,
                sort_order=i,
            )
            for i in range(subtask_count)
        ]
        return TaskFactory.create(items=items, **kwargs)

    @staticmethod
    def create_with_tags(tags: list[str], **kwargs) -> Task:
        """Create task with specified tags."""
        return TaskFactory.create(tags=tags, **kwargs)

    @staticmethod
    def create_recurring(rrule: str = "RRULE:FREQ=DAILY;INTERVAL=1", **kwargs) -> Task:
        """Create recurring task."""
        return TaskFactory.create(repeat_flag=rrule, **kwargs)

    @staticmethod
    def create_child_task(parent_id: str, **kwargs) -> Task:
        """Create a child task (subtask)."""
        return TaskFactory.create(parent_id=parent_id, **kwargs)

    @staticmethod
    def create_batch(count: int, **kwargs) -> list[Task]:
        """Create multiple tasks."""
        return [TaskFactory.create(title=f"Task {i+1}", **kwargs) for i in range(count)]

    @staticmethod
    def create_priority_set() -> list[Task]:
        """Create one task of each priority level."""
        return [
            TaskFactory.create(title="No Priority", priority=TaskPriority.NONE),
            TaskFactory.create(title="Low Priority", priority=TaskPriority.LOW),
            TaskFactory.create(title="Medium Priority", priority=TaskPriority.MEDIUM),
            TaskFactory.create(title="High Priority", priority=TaskPriority.HIGH),
        ]


class ProjectFactory:
    """Factory for creating Project test objects."""

    @staticmethod
    def create(
        id: str | None = None,
        name: str = "Test Project",
        color: str | None = "#F18181",
        kind: str = ProjectKind.TASK,
        view_mode: str = ViewMode.LIST,
        group_id: str | None = None,
        closed: bool = False,
        sort_order: int = 0,
        **kwargs,
    ) -> Project:
        """Create a Project with sensible defaults."""
        return Project(
            id=id or IDGenerator.project_id(),
            name=name,
            color=color,
            kind=kind,
            view_mode=view_mode,
            group_id=group_id,
            closed=closed,
            sort_order=sort_order,
            **kwargs,
        )

    @staticmethod
    def create_inbox(user_id: int = 123456789) -> Project:
        """Create an inbox project."""
        return ProjectFactory.create(
            id=f"inbox{user_id}",
            name="Inbox",
            color=None,
        )

    @staticmethod
    def create_note_project(**kwargs) -> Project:
        """Create a NOTE type project."""
        return ProjectFactory.create(kind=ProjectKind.NOTE, **kwargs)

    @staticmethod
    def create_kanban_project(**kwargs) -> Project:
        """Create a kanban view project."""
        return ProjectFactory.create(view_mode=ViewMode.KANBAN, **kwargs)

    @staticmethod
    def create_in_folder(folder_id: str, **kwargs) -> Project:
        """Create project in a folder."""
        return ProjectFactory.create(group_id=folder_id, **kwargs)

    @staticmethod
    def create_batch(count: int, **kwargs) -> list[Project]:
        """Create multiple projects."""
        return [ProjectFactory.create(name=f"Project {i+1}", **kwargs) for i in range(count)]


class FolderFactory:
    """Factory for creating ProjectGroup (folder) test objects."""

    @staticmethod
    def create(
        id: str | None = None,
        name: str = "Test Folder",
        sort_order: int = 0,
        **kwargs,
    ) -> ProjectGroup:
        """Create a ProjectGroup with sensible defaults."""
        return ProjectGroup(
            id=id or IDGenerator.folder_id(),
            name=name,
            sort_order=sort_order,
            **kwargs,
        )

    @staticmethod
    def create_batch(count: int, **kwargs) -> list[ProjectGroup]:
        """Create multiple folders."""
        return [FolderFactory.create(name=f"Folder {i+1}", **kwargs) for i in range(count)]


class TagFactory:
    """Factory for creating Tag test objects."""

    @staticmethod
    def create(
        name: str | None = None,
        label: str = "TestTag",
        color: str | None = "#86BB6D",
        parent: str | None = None,
        sort_order: int = 0,
        **kwargs,
    ) -> Tag:
        """Create a Tag with sensible defaults."""
        tag_name = name or label.lower().replace(" ", "")
        return Tag(
            name=tag_name,
            label=label,
            color=color,
            parent=parent,
            sort_order=sort_order,
            **kwargs,
        )

    @staticmethod
    def create_nested(parent_label: str, child_labels: list[str]) -> list[Tag]:
        """Create a parent tag with children."""
        parent = TagFactory.create(label=parent_label)
        children = [
            TagFactory.create(label=label, parent=parent.name)
            for label in child_labels
        ]
        return [parent] + children

    @staticmethod
    def create_batch(labels: list[str]) -> list[Tag]:
        """Create multiple tags from labels."""
        return [TagFactory.create(label=label) for label in labels]


class UserFactory:
    """Factory for creating User test objects."""

    @staticmethod
    def create(
        username: str = "testuser@example.com",
        display_name: str = "Test User",
        name: str = "Test",
        email: str | None = None,
        locale: str = "en_US",
        verified_email: bool = True,
        **kwargs,
    ) -> User:
        """Create a User with sensible defaults."""
        return User(
            username=username,
            display_name=display_name,
            name=name,
            email=email or username,
            locale=locale,
            verified_email=verified_email,
            **kwargs,
        )


class UserStatusFactory:
    """Factory for creating UserStatus test objects."""

    @staticmethod
    def create(
        user_id: str = "123456789",
        username: str = "testuser@example.com",
        inbox_id: str = "inbox123456789",
        is_pro: bool = True,
        team_user: bool = False,
        pro_end_date: str | None = None,
        **kwargs,
    ) -> UserStatus:
        """Create a UserStatus with sensible defaults."""
        return UserStatus(
            user_id=user_id,
            username=username,
            inbox_id=inbox_id,
            is_pro=is_pro,
            team_user=team_user,
            pro_end_date=pro_end_date or "2026-12-31",
            **kwargs,
        )

    @staticmethod
    def create_free_user(**kwargs) -> UserStatus:
        """Create a free tier user status."""
        return UserStatusFactory.create(is_pro=False, pro_end_date=None, **kwargs)


class UserStatisticsFactory:
    """Factory for creating UserStatistics test objects."""

    @staticmethod
    def create(
        score: int = 1000,
        level: int = 5,
        today_completed: int = 3,
        yesterday_completed: int = 5,
        total_completed: int = 500,
        today_pomo_count: int = 2,
        yesterday_pomo_count: int = 4,
        total_pomo_count: int = 100,
        today_pomo_duration: int = 3000,
        yesterday_pomo_duration: int = 6000,
        total_pomo_duration: int = 150000,
        **kwargs,
    ) -> UserStatistics:
        """Create UserStatistics with sensible defaults."""
        return UserStatistics(
            score=score,
            level=level,
            today_completed=today_completed,
            yesterday_completed=yesterday_completed,
            total_completed=total_completed,
            today_pomo_count=today_pomo_count,
            yesterday_pomo_count=yesterday_pomo_count,
            total_pomo_count=total_pomo_count,
            today_pomo_duration=today_pomo_duration,
            yesterday_pomo_duration=yesterday_pomo_duration,
            total_pomo_duration=total_pomo_duration,
            **kwargs,
        )


class ProjectDataFactory:
    """Factory for creating ProjectData test objects."""

    @staticmethod
    def create(
        project: Project | None = None,
        tasks: list[Task] | None = None,
        columns: list[Column] | None = None,
    ) -> ProjectData:
        """Create ProjectData with sensible defaults."""
        proj = project or ProjectFactory.create()
        return ProjectData(
            project=proj,
            tasks=tasks or TaskFactory.create_batch(3, project_id=proj.id),
            columns=columns or [],
        )


class ColumnFactory:
    """Factory for creating Column test objects."""

    @staticmethod
    def create(
        id: str | None = None,
        project_id: str | None = None,
        name: str = "To Do",
        sort_order: int = 0,
    ) -> Column:
        """Create a Column with sensible defaults."""
        return Column(
            id=id or IDGenerator.next_id(),
            project_id=project_id or IDGenerator.project_id(),
            name=name,
            sort_order=sort_order,
        )

    @staticmethod
    def create_kanban_set(project_id: str) -> list[Column]:
        """Create a standard kanban column set."""
        return [
            ColumnFactory.create(project_id=project_id, name="To Do", sort_order=0),
            ColumnFactory.create(project_id=project_id, name="In Progress", sort_order=1),
            ColumnFactory.create(project_id=project_id, name="Done", sort_order=2),
        ]


# =============================================================================
# Mock API Classes
# =============================================================================


class MockUnifiedAPI:
    """
    Comprehensive mock for UnifiedTickTickAPI.

    This mock provides configurable behavior for all API operations,
    allowing tests to simulate various scenarios including success,
    failure, and edge cases.
    """

    def __init__(self):
        """Initialize mock with default data stores."""
        self.tasks: dict[str, Task] = {}
        self.projects: dict[str, Project] = {}
        self.folders: dict[str, ProjectGroup] = {}
        self.tags: dict[str, Tag] = {}
        self.user: User = UserFactory.create()
        self.user_status: UserStatus = UserStatusFactory.create()
        self.user_statistics: UserStatistics = UserStatisticsFactory.create()
        self.user_preferences: dict = {"timeZone": "UTC", "weekStartDay": 0}
        self.inbox_id: str = "inbox123456789"
        self._initialized: bool = False

        # Mock data for special queries
        self.abandoned_tasks: list = []  # Raw dict data for abandoned tasks
        self.deleted_tasks: list = []    # Raw dict data for deleted tasks

        # Habit data
        self._habits: dict[str, Habit] = {}
        self._habit_sections: list[HabitSection] = [
            HabitSection(id="section_morning", name="_morning", sort_order=-196608),
            HabitSection(id="section_afternoon", name="_afternoon", sort_order=-131072),
            HabitSection(id="section_night", name="_night", sort_order=-65536),
        ]
        self._habit_preferences: HabitPreferences = HabitPreferences(
            show_in_calendar=True,
            show_in_today=True,
            enabled=True,
            default_section_order=0,
        )
        self._habit_checkins: dict[str, list[HabitCheckin]] = {}

        # Track method calls for verification
        self.call_history: list[tuple[str, tuple, dict]] = []

        # Configurable behaviors
        self.should_fail: dict[str, Exception | None] = {}
        self.delays: dict[str, float] = {}

    def _record_call(self, method: str, args: tuple, kwargs: dict) -> None:
        """Record method call for verification."""
        self.call_history.append((method, args, kwargs))

    def _check_failure(self, method: str) -> None:
        """Check if method should raise an exception."""
        if method in self.should_fail and self.should_fail[method]:
            raise self.should_fail[method]

    async def initialize(self) -> None:
        """Mock initialization."""
        self._record_call("initialize", (), {})
        self._check_failure("initialize")
        self._initialized = True

    async def close(self) -> None:
        """Mock close."""
        self._record_call("close", (), {})
        self._initialized = False

    # -------------------------------------------------------------------------
    # Task Operations
    # -------------------------------------------------------------------------

    async def create_task(
        self,
        title: str,
        project_id: str | None = None,
        **kwargs,
    ) -> Task:
        """Mock task creation."""
        self._record_call("create_task", (title,), {"project_id": project_id, **kwargs})
        self._check_failure("create_task")

        # Filter out None values to allow factory defaults to apply
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}

        task = TaskFactory.create(
            title=title,
            project_id=project_id or self.inbox_id,
            **filtered_kwargs,
        )
        self.tasks[task.id] = task
        return task

    async def get_task(self, task_id: str, project_id: str | None = None) -> Task:
        """Mock get task."""
        self._record_call("get_task", (task_id,), {"project_id": project_id})
        self._check_failure("get_task")

        if task_id not in self.tasks:
            from ticktick_sdk.exceptions import TickTickNotFoundError
            raise TickTickNotFoundError(f"Task not found: {task_id}")
        return self.tasks[task_id]

    async def update_task(self, task: Task) -> Task:
        """Mock update task."""
        self._record_call("update_task", (task,), {})
        self._check_failure("update_task")

        if task.id not in self.tasks:
            from ticktick_sdk.exceptions import TickTickNotFoundError
            raise TickTickNotFoundError(f"Task not found: {task.id}")

        task.modified_time = utc_now()
        self.tasks[task.id] = task
        return task

    async def complete_task(self, task_id: str, project_id: str) -> None:
        """Mock complete task."""
        self._record_call("complete_task", (task_id, project_id), {})
        self._check_failure("complete_task")

        if task_id not in self.tasks:
            from ticktick_sdk.exceptions import TickTickNotFoundError
            raise TickTickNotFoundError(f"Task not found: {task_id}")

        task = self.tasks[task_id]
        task.status = TaskStatus.COMPLETED
        task.completed_time = utc_now()

    async def delete_task(self, task_id: str, project_id: str) -> None:
        """Mock delete task (soft delete - set deleted=1).

        Matches actual API behavior: tasks go to trash with deleted=1,
        not permanently removed.
        """
        self._record_call("delete_task", (task_id, project_id), {})
        self._check_failure("delete_task")

        if task_id not in self.tasks:
            from ticktick_sdk.exceptions import TickTickNotFoundError
            raise TickTickNotFoundError(f"Task not found: {task_id}")

        # Soft delete - set deleted flag instead of removing
        self.tasks[task_id].deleted = 1

    async def list_all_tasks(self) -> list[Task]:
        """Mock list all tasks (excludes deleted tasks)."""
        self._record_call("list_all_tasks", (), {})
        self._check_failure("list_all_tasks")

        # Return active, non-deleted tasks
        return [t for t in self.tasks.values() if t.status == TaskStatus.ACTIVE and t.deleted == 0]

    async def list_completed_tasks(
        self,
        from_date: datetime,
        to_date: datetime,
        limit: int = 100,
    ) -> list[Task]:
        """Mock list completed tasks."""
        self._record_call("list_completed_tasks", (from_date, to_date), {"limit": limit})
        self._check_failure("list_completed_tasks")

        # Normalize dates to be timezone-aware for comparison
        from_aware = from_date.replace(tzinfo=timezone.utc) if from_date.tzinfo is None else from_date
        to_aware = to_date.replace(tzinfo=timezone.utc) if to_date.tzinfo is None else to_date

        completed = []
        for t in self.tasks.values():
            if t.status == TaskStatus.COMPLETED and t.completed_time:
                ct = t.completed_time
                ct_aware = ct.replace(tzinfo=timezone.utc) if ct.tzinfo is None else ct
                if from_aware <= ct_aware <= to_aware:
                    completed.append(t)

        return completed[:limit]

    async def list_abandoned_tasks(
        self,
        from_date: datetime,
        to_date: datetime,
        limit: int = 100,
    ) -> list[Task]:
        """Mock list abandoned (won't do) tasks."""
        self._record_call("list_abandoned_tasks", (from_date, to_date), {"limit": limit})
        self._check_failure("list_abandoned_tasks")

        # Return pre-configured abandoned tasks (converted from raw dict data)
        return [Task.from_v2(t) for t in self.abandoned_tasks[:limit]]

    async def list_deleted_tasks(
        self,
        start: int = 0,
        limit: int = 100,
    ) -> list[Task]:
        """Mock list deleted tasks (trash)."""
        self._record_call("list_deleted_tasks", (start,), {"limit": limit})
        self._check_failure("list_deleted_tasks")

        # Return pre-configured deleted tasks (converted from raw dict data)
        return [Task.from_v2(t) for t in self.deleted_tasks[start:start + limit]]

    async def move_task(
        self,
        task_id: str,
        from_project_id: str,
        to_project_id: str,
    ) -> None:
        """Mock move task."""
        self._record_call("move_task", (task_id, from_project_id, to_project_id), {})
        self._check_failure("move_task")

        if task_id not in self.tasks:
            from ticktick_sdk.exceptions import TickTickNotFoundError
            raise TickTickNotFoundError(f"Task not found: {task_id}")

        self.tasks[task_id].project_id = to_project_id

    async def set_task_parent(
        self,
        task_id: str,
        project_id: str,
        parent_id: str,
    ) -> None:
        """Mock set task parent."""
        self._record_call("set_task_parent", (task_id, project_id, parent_id), {})
        self._check_failure("set_task_parent")

        if task_id not in self.tasks:
            from ticktick_sdk.exceptions import TickTickNotFoundError
            raise TickTickNotFoundError(f"Task not found: {task_id}")

        self.tasks[task_id].parent_id = parent_id

        if parent_id in self.tasks:
            parent = self.tasks[parent_id]
            if task_id not in parent.child_ids:
                parent.child_ids.append(task_id)

    async def unset_task_parent(
        self,
        task_id: str,
        project_id: str,
    ) -> None:
        """Mock unset task parent (remove subtask from parent)."""
        self._record_call("unset_task_parent", (task_id, project_id), {})
        self._check_failure("unset_task_parent")

        if task_id not in self.tasks:
            from ticktick_sdk.exceptions import TickTickNotFoundError
            raise TickTickNotFoundError(f"Task not found: {task_id}")

        task = self.tasks[task_id]
        parent_id = task.parent_id

        if not parent_id:
            from ticktick_sdk.exceptions import TickTickAPIError
            raise TickTickAPIError(f"Task {task_id} is not a subtask (has no parent)")

        # Remove from parent's child list
        if parent_id in self.tasks:
            parent = self.tasks[parent_id]
            if task_id in parent.child_ids:
                parent.child_ids.remove(task_id)

        # Clear parent reference
        task.parent_id = None

    # -------------------------------------------------------------------------
    # Project Operations
    # -------------------------------------------------------------------------

    async def list_projects(self) -> list[Project]:
        """Mock list projects."""
        self._record_call("list_projects", (), {})
        self._check_failure("list_projects")
        return list(self.projects.values())

    async def get_project(self, project_id: str) -> Project:
        """Mock get project."""
        self._record_call("get_project", (project_id,), {})
        self._check_failure("get_project")

        if project_id not in self.projects:
            from ticktick_sdk.exceptions import TickTickNotFoundError
            raise TickTickNotFoundError(f"Project not found: {project_id}")
        return self.projects[project_id]

    async def get_project_with_data(self, project_id: str) -> ProjectData:
        """Mock get project with data."""
        self._record_call("get_project_with_data", (project_id,), {})
        self._check_failure("get_project_with_data")

        if project_id not in self.projects:
            from ticktick_sdk.exceptions import TickTickNotFoundError
            raise TickTickNotFoundError(f"Project not found: {project_id}")

        project = self.projects[project_id]
        tasks = [t for t in self.tasks.values() if t.project_id == project_id]
        return ProjectData(project=project, tasks=tasks, columns=[])

    async def create_project(
        self,
        name: str,
        color: str | None = None,
        kind: str = "TASK",
        view_mode: str = "list",
        group_id: str | None = None,
    ) -> Project:
        """Mock create project."""
        self._record_call("create_project", (name,), {
            "color": color, "kind": kind, "view_mode": view_mode, "group_id": group_id
        })
        self._check_failure("create_project")

        project = ProjectFactory.create(
            name=name,
            color=color,
            kind=kind,
            view_mode=view_mode,
            group_id=group_id,
        )
        self.projects[project.id] = project
        return project

    async def delete_project(self, project_id: str) -> None:
        """Mock delete project."""
        self._record_call("delete_project", (project_id,), {})
        self._check_failure("delete_project")

        if project_id not in self.projects:
            from ticktick_sdk.exceptions import TickTickNotFoundError
            raise TickTickNotFoundError(f"Project not found: {project_id}")

        del self.projects[project_id]
        # Also delete associated tasks
        self.tasks = {k: v for k, v in self.tasks.items() if v.project_id != project_id}

    # -------------------------------------------------------------------------
    # Folder Operations
    # -------------------------------------------------------------------------

    async def list_project_groups(self) -> list[ProjectGroup]:
        """Mock list project groups (folders)."""
        self._record_call("list_project_groups", (), {})
        self._check_failure("list_project_groups")
        return list(self.folders.values())

    async def create_project_group(self, name: str) -> ProjectGroup:
        """Mock create project group."""
        self._record_call("create_project_group", (name,), {})
        self._check_failure("create_project_group")

        folder = FolderFactory.create(name=name)
        self.folders[folder.id] = folder
        return folder

    async def delete_project_group(self, group_id: str) -> None:
        """Mock delete project group.

        Note: TickTick does NOT automatically ungroup projects when their folder
        is deleted. Projects retain their group_id as a "dangling reference".
        """
        self._record_call("delete_project_group", (group_id,), {})
        self._check_failure("delete_project_group")

        if group_id not in self.folders:
            from ticktick_sdk.exceptions import TickTickNotFoundError
            raise TickTickNotFoundError(f"Folder not found: {group_id}")

        del self.folders[group_id]
        # Note: Do NOT ungroup projects - TickTick leaves group_id as-is

    # -------------------------------------------------------------------------
    # Tag Operations
    # -------------------------------------------------------------------------

    async def list_tags(self) -> list[Tag]:
        """Mock list tags."""
        self._record_call("list_tags", (), {})
        self._check_failure("list_tags")
        return list(self.tags.values())

    async def create_tag(
        self,
        name: str,
        color: str | None = None,
        parent: str | None = None,
    ) -> Tag:
        """Mock create tag."""
        self._record_call("create_tag", (name,), {"color": color, "parent": parent})
        self._check_failure("create_tag")

        tag = TagFactory.create(label=name, color=color, parent=parent)
        self.tags[tag.name] = tag
        return tag

    async def delete_tag(self, name: str) -> None:
        """Mock delete tag."""
        self._record_call("delete_tag", (name,), {})
        self._check_failure("delete_tag")

        tag_name = name.lower()
        if tag_name not in self.tags:
            from ticktick_sdk.exceptions import TickTickNotFoundError
            raise TickTickNotFoundError(f"Tag not found: {name}")

        del self.tags[tag_name]
        # Remove tag from tasks
        for task in self.tasks.values():
            task.tags = [t for t in task.tags if t.lower() != tag_name]

    async def rename_tag(self, old_name: str, new_name: str) -> None:
        """Mock rename tag."""
        self._record_call("rename_tag", (old_name, new_name), {})
        self._check_failure("rename_tag")

        old_tag_name = old_name.lower()
        if old_tag_name not in self.tags:
            from ticktick_sdk.exceptions import TickTickNotFoundError
            raise TickTickNotFoundError(f"Tag not found: {old_name}")

        tag = self.tags.pop(old_tag_name)
        new_tag_name = new_name.lower()
        tag.name = new_tag_name
        tag.label = new_name
        self.tags[new_tag_name] = tag

        # Update tag references in tasks
        for task in self.tasks.values():
            task.tags = [new_tag_name if t.lower() == old_tag_name else t for t in task.tags]

    async def merge_tags(self, source: str, target: str) -> None:
        """Mock merge tags."""
        self._record_call("merge_tags", (source, target), {})
        self._check_failure("merge_tags")

        source_name = source.lower()
        target_name = target.lower()

        if source_name not in self.tags:
            from ticktick_sdk.exceptions import TickTickNotFoundError
            raise TickTickNotFoundError(f"Source tag not found: {source}")
        if target_name not in self.tags:
            from ticktick_sdk.exceptions import TickTickNotFoundError
            raise TickTickNotFoundError(f"Target tag not found: {target}")

        # Move tasks from source to target
        for task in self.tasks.values():
            if source_name in [t.lower() for t in task.tags]:
                task.tags = [t for t in task.tags if t.lower() != source_name]
                if target_name not in [t.lower() for t in task.tags]:
                    task.tags.append(target_name)

        del self.tags[source_name]

    # -------------------------------------------------------------------------
    # User Operations
    # -------------------------------------------------------------------------

    async def get_user_profile(self) -> User:
        """Mock get user profile."""
        self._record_call("get_user_profile", (), {})
        self._check_failure("get_user_profile")
        return self.user

    async def get_user_status(self) -> UserStatus:
        """Mock get user status."""
        self._record_call("get_user_status", (), {})
        self._check_failure("get_user_status")
        return self.user_status

    async def get_user_statistics(self) -> UserStatistics:
        """Mock get user statistics."""
        self._record_call("get_user_statistics", (), {})
        self._check_failure("get_user_statistics")
        return self.user_statistics

    # -------------------------------------------------------------------------
    # Focus Operations
    # -------------------------------------------------------------------------

    async def get_focus_heatmap(
        self,
        start_date,
        end_date,
    ) -> list[dict[str, Any]]:
        """Mock get focus heatmap."""
        self._record_call("get_focus_heatmap", (start_date, end_date), {})
        self._check_failure("get_focus_heatmap")
        return [{"duration": 3600}, {"duration": 7200}]

    async def get_focus_by_tag(
        self,
        start_date,
        end_date,
    ) -> dict[str, int]:
        """Mock get focus by tag."""
        self._record_call("get_focus_by_tag", (start_date, end_date), {})
        self._check_failure("get_focus_by_tag")
        return {"work": 7200, "study": 3600}

    # -------------------------------------------------------------------------
    # Habit Operations
    # -------------------------------------------------------------------------

    async def list_habits(self) -> list[Habit]:
        """Mock list all habits."""
        self._record_call("list_habits", (), {})
        self._check_failure("list_habits")
        return list(self._habits.values())

    async def get_habit(self, habit_id: str) -> Habit:
        """Mock get a habit by ID."""
        self._record_call("get_habit", (habit_id,), {})
        self._check_failure("get_habit")

        if habit_id not in self._habits:
            from ticktick_sdk.exceptions import TickTickNotFoundError
            raise TickTickNotFoundError(f"Habit not found: {habit_id}")
        return self._habits[habit_id]

    async def list_habit_sections(self) -> list[HabitSection]:
        """Mock list habit sections."""
        self._record_call("list_habit_sections", (), {})
        self._check_failure("list_habit_sections")
        return self._habit_sections

    async def get_habit_preferences(self) -> HabitPreferences:
        """Mock get habit preferences."""
        self._record_call("get_habit_preferences", (), {})
        self._check_failure("get_habit_preferences")
        return self._habit_preferences

    async def create_habit(
        self,
        name: str,
        *,
        habit_type: str = "Boolean",
        goal: float = 1.0,
        step: float = 0.0,
        unit: str = "Count",
        icon: str = "habit_daily_check_in",
        color: str = "#97E38B",
        section_id: str | None = None,
        repeat_rule: str = "RRULE:FREQ=WEEKLY;BYDAY=SU,MO,TU,WE,TH,FR,SA",
        reminders: list[str] | None = None,
        target_days: int = 0,
        encouragement: str = "",
    ) -> Habit:
        """Mock create a habit."""
        self._record_call("create_habit", (name,), {
            "habit_type": habit_type, "goal": goal, "step": step, "unit": unit,
            "icon": icon, "color": color, "section_id": section_id,
            "repeat_rule": repeat_rule, "reminders": reminders,
            "target_days": target_days, "encouragement": encouragement,
        })
        self._check_failure("create_habit")

        import secrets
        habit_id = secrets.token_hex(12)
        habit = Habit(
            id=habit_id,
            name=name,
            habit_type=habit_type,
            goal=goal,
            step=step,
            unit=unit,
            icon=icon,
            color=color,
            section_id=section_id,
            repeat_rule=repeat_rule,
            reminders=reminders or [],
            target_days=target_days,
            encouragement=encouragement,
            total_checkins=0,
            current_streak=0,
            status=0,
        )
        self._habits[habit_id] = habit
        return habit

    async def update_habit(
        self,
        habit_id: str,
        *,
        name: str | None = None,
        goal: float | None = None,
        step: float | None = None,
        unit: str | None = None,
        icon: str | None = None,
        color: str | None = None,
        section_id: str | None = None,
        repeat_rule: str | None = None,
        reminders: list[str] | None = None,
        target_days: int | None = None,
        encouragement: str | None = None,
    ) -> Habit:
        """Mock update a habit."""
        self._record_call("update_habit", (habit_id,), {
            "name": name, "goal": goal, "step": step, "unit": unit,
            "icon": icon, "color": color, "section_id": section_id,
            "repeat_rule": repeat_rule, "reminders": reminders,
            "target_days": target_days, "encouragement": encouragement,
        })
        self._check_failure("update_habit")

        if habit_id not in self._habits:
            from ticktick_sdk.exceptions import TickTickNotFoundError
            raise TickTickNotFoundError(f"Habit not found: {habit_id}")

        habit = self._habits[habit_id]
        if name is not None:
            habit.name = name
        if goal is not None:
            habit.goal = goal
        if step is not None:
            habit.step = step
        if unit is not None:
            habit.unit = unit
        if icon is not None:
            habit.icon = icon
        if color is not None:
            habit.color = color
        if section_id is not None:
            habit.section_id = section_id
        if repeat_rule is not None:
            habit.repeat_rule = repeat_rule
        if reminders is not None:
            habit.reminders = reminders
        if target_days is not None:
            habit.target_days = target_days
        if encouragement is not None:
            habit.encouragement = encouragement

        return habit

    async def delete_habit(self, habit_id: str) -> None:
        """Mock delete a habit."""
        self._record_call("delete_habit", (habit_id,), {})
        self._check_failure("delete_habit")

        if habit_id not in self._habits:
            from ticktick_sdk.exceptions import TickTickNotFoundError
            raise TickTickNotFoundError(f"Habit not found: {habit_id}")

        del self._habits[habit_id]

    async def checkin_habit(
        self,
        habit_id: str,
        value: float = 1.0,
    ) -> Habit:
        """Mock check in a habit."""
        self._record_call("checkin_habit", (habit_id,), {"value": value})
        self._check_failure("checkin_habit")

        if habit_id not in self._habits:
            from ticktick_sdk.exceptions import TickTickNotFoundError
            raise TickTickNotFoundError(f"Habit not found: {habit_id}")

        habit = self._habits[habit_id]
        habit.total_checkins += int(value)
        habit.current_streak += 1
        return habit

    async def archive_habit(self, habit_id: str) -> Habit:
        """Mock archive a habit."""
        self._record_call("archive_habit", (habit_id,), {})
        self._check_failure("archive_habit")

        if habit_id not in self._habits:
            from ticktick_sdk.exceptions import TickTickNotFoundError
            raise TickTickNotFoundError(f"Habit not found: {habit_id}")

        habit = self._habits[habit_id]
        habit.status = 2  # Archived
        return habit

    async def unarchive_habit(self, habit_id: str) -> Habit:
        """Mock unarchive a habit."""
        self._record_call("unarchive_habit", (habit_id,), {})
        self._check_failure("unarchive_habit")

        if habit_id not in self._habits:
            from ticktick_sdk.exceptions import TickTickNotFoundError
            raise TickTickNotFoundError(f"Habit not found: {habit_id}")

        habit = self._habits[habit_id]
        habit.status = 0  # Active
        return habit

    async def get_habit_checkins(
        self,
        habit_ids: list[str],
        after_stamp: int = 0,
    ) -> dict[str, list[HabitCheckin]]:
        """Mock get habit check-ins."""
        self._record_call("get_habit_checkins", (habit_ids,), {"after_stamp": after_stamp})
        self._check_failure("get_habit_checkins")

        result: dict[str, list[HabitCheckin]] = {}
        for habit_id in habit_ids:
            result[habit_id] = self._habit_checkins.get(habit_id, [])
        return result

    # -------------------------------------------------------------------------
    # Sync Operations
    # -------------------------------------------------------------------------

    async def sync_all(self) -> dict[str, Any]:
        """Mock full sync."""
        self._record_call("sync_all", (), {})
        self._check_failure("sync_all")

        return {
            "inboxId": self.inbox_id,
            "projectProfiles": [p.model_dump() for p in self.projects.values()],
            "syncTaskBean": {
                "update": [t.model_dump() for t in self.tasks.values()],
            },
            "tags": [t.model_dump() for t in self.tags.values()],
            "projectGroups": [f.model_dump() for f in self.folders.values()],
        }

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def seed_data(
        self,
        tasks: int = 5,
        projects: int = 3,
        folders: int = 2,
        tags: int = 4,
    ) -> None:
        """Seed mock with test data."""
        # Create folders
        for i in range(folders):
            folder = FolderFactory.create(name=f"Folder {i+1}")
            self.folders[folder.id] = folder

        # Create projects
        folder_ids = list(self.folders.keys())
        for i in range(projects):
            group_id = folder_ids[i % len(folder_ids)] if folder_ids else None
            project = ProjectFactory.create(name=f"Project {i+1}", group_id=group_id)
            self.projects[project.id] = project

        # Create tags
        tag_labels = ["work", "personal", "urgent", "later"][:tags]
        for label in tag_labels:
            tag = TagFactory.create(label=label)
            self.tags[tag.name] = tag

        # Create tasks
        project_ids = list(self.projects.keys())
        tag_names = list(self.tags.keys())
        for i in range(tasks):
            project_id = project_ids[i % len(project_ids)] if project_ids else self.inbox_id
            task_tags = [tag_names[i % len(tag_names)]] if tag_names else []
            task = TaskFactory.create(
                title=f"Task {i+1}",
                project_id=project_id,
                tags=task_tags,
                priority=i % 4 * 2 if i % 4 < 3 else 5,  # Cycles through priorities
            )
            self.tasks[task.id] = task

    def clear_call_history(self) -> None:
        """Clear recorded method calls."""
        self.call_history.clear()

    def get_calls(self, method_name: str) -> list[tuple[tuple, dict]]:
        """Get all calls to a specific method."""
        return [(args, kwargs) for name, args, kwargs in self.call_history if name == method_name]

    def assert_called(self, method_name: str, times: int | None = None) -> None:
        """Assert a method was called (optionally a specific number of times)."""
        calls = self.get_calls(method_name)
        if times is not None:
            assert len(calls) == times, f"Expected {method_name} to be called {times} times, got {len(calls)}"
        else:
            assert len(calls) > 0, f"Expected {method_name} to be called at least once"

    def assert_not_called(self, method_name: str) -> None:
        """Assert a method was not called."""
        calls = self.get_calls(method_name)
        assert len(calls) == 0, f"Expected {method_name} not to be called, but was called {len(calls)} times"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def id_generator():
    """Reset and provide ID generator."""
    IDGenerator.reset()
    return IDGenerator


@pytest.fixture
def live_mode(request) -> bool:
    """Check if running in live mode."""
    return request.config.getoption("--live")


@pytest.fixture
def mock_api() -> MockUnifiedAPI:
    """Create a fresh mock API instance."""
    return MockUnifiedAPI()


@pytest.fixture
def seeded_mock_api() -> MockUnifiedAPI:
    """Create a mock API instance with seeded test data."""
    api = MockUnifiedAPI()
    api.seed_data(tasks=10, projects=5, folders=3, tags=5)
    return api


# =============================================================================
# Session-Scoped Event Loop (for live mode)
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop.

    This is necessary for live tests where we share an httpx client
    across all tests. Without this, each test gets a new event loop,
    causing "Event loop is closed" errors when the shared client
    tries to use connections from a closed event loop.
    """
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Live Client Management
# =============================================================================

# Global storage for live client (session-wide)
_live_client: TickTickClient | None = None
_live_client_event_loop: asyncio.AbstractEventLoop | None = None


async def _get_live_client() -> TickTickClient:
    """Get or create a live client for the current event loop.

    ALWAYS creates a new client for each request to avoid event loop issues.
    This means re-authenticating each time, but it's the only reliable way
    to avoid "bound to a different event loop" errors.

    For live tests, we accept the overhead of re-authentication because:
    1. The alternative (sharing clients across event loops) simply doesn't work
    2. Re-authentication is fast (~1 second)
    """
    from dotenv import load_dotenv
    load_dotenv()

    # Check for required credentials
    required_vars = [
        "TICKTICK_CLIENT_ID",
        "TICKTICK_CLIENT_SECRET",
        "TICKTICK_ACCESS_TOKEN",
        "TICKTICK_USERNAME",
        "TICKTICK_PASSWORD",
    ]
    missing = [var for var in required_vars if not os.environ.get(var)]
    if missing:
        pytest.skip(f"Live mode requires environment variables: {', '.join(missing)}")

    # Always create a fresh client for the current event loop
    client = TickTickClient(
        client_id=os.environ["TICKTICK_CLIENT_ID"],
        client_secret=os.environ["TICKTICK_CLIENT_SECRET"],
        v1_access_token=os.environ["TICKTICK_ACCESS_TOKEN"],
        username=os.environ["TICKTICK_USERNAME"],
        password=os.environ["TICKTICK_PASSWORD"],
        redirect_uri=os.environ.get("TICKTICK_REDIRECT_URI", "http://127.0.0.1:8080/callback"),
    )

    await client.connect()
    return client


def _create_live_client_sync() -> TickTickClient:
    """Create a live client (sync wrapper for async creation)."""
    from dotenv import load_dotenv
    load_dotenv()

    # Check for required credentials
    required_vars = [
        "TICKTICK_CLIENT_ID",
        "TICKTICK_CLIENT_SECRET",
        "TICKTICK_ACCESS_TOKEN",
        "TICKTICK_USERNAME",
        "TICKTICK_PASSWORD",
    ]
    missing = [var for var in required_vars if not os.environ.get(var)]
    if missing:
        pytest.skip(f"Live mode requires environment variables: {', '.join(missing)}")

    return TickTickClient(
        client_id=os.environ["TICKTICK_CLIENT_ID"],
        client_secret=os.environ["TICKTICK_CLIENT_SECRET"],
        v1_access_token=os.environ["TICKTICK_ACCESS_TOKEN"],
        username=os.environ["TICKTICK_USERNAME"],
        password=os.environ["TICKTICK_PASSWORD"],
        redirect_uri=os.environ.get("TICKTICK_REDIRECT_URI", "http://127.0.0.1:8080/callback"),
    )


# =============================================================================
# Live Mode Resource Cleanup
# =============================================================================


class LiveResourceTracker:
    """Tracks resources created during live tests for cleanup.

    This ensures we don't exceed TickTick's resource limits (e.g., 9 projects
    for free accounts) by cleaning up after each test.

    Features:
    - Tracks tasks, projects, folders, and tags
    - Handles renamed tags (updates tracked name)
    - Handles merged tags (removes source from tracking)
    - Handles moved tasks (updates project_id)
    - Handles explicit deletions (removes from tracking to avoid double-delete)
    - Cleans up in dependency order (tasks -> projects -> folders -> tags)
    """

    def __init__(self):
        self.tasks: dict[str, str] = {}  # task_id -> project_id
        self.projects: set[str] = set()  # project_ids
        self.folders: set[str] = set()  # folder_ids
        self.tags: set[str] = set()  # tag_names (lowercase)

    # -------------------------------------------------------------------------
    # Track operations (called when resources are created)
    # -------------------------------------------------------------------------

    def track_task(self, task_id: str, project_id: str) -> None:
        """Track a created task for cleanup."""
        self.tasks[task_id] = project_id

    def track_project(self, project_id: str) -> None:
        """Track a created project for cleanup."""
        self.projects.add(project_id)

    def track_folder(self, folder_id: str) -> None:
        """Track a created folder for cleanup."""
        self.folders.add(folder_id)

    def track_tag(self, tag_name: str) -> None:
        """Track a created tag for cleanup."""
        # Tags are stored lowercase (TickTick normalizes them)
        self.tags.add(tag_name.lower())

    # -------------------------------------------------------------------------
    # Update operations (called when resources are modified)
    # -------------------------------------------------------------------------

    def update_task_project(self, task_id: str, new_project_id: str) -> None:
        """Update the project_id for a moved task."""
        if task_id in self.tasks:
            self.tasks[task_id] = new_project_id

    def rename_tag(self, old_name: str, new_name: str) -> None:
        """Update tracking when a tag is renamed."""
        old_lower = old_name.lower()
        new_lower = new_name.lower()
        if old_lower in self.tags:
            self.tags.discard(old_lower)
            self.tags.add(new_lower)

    def merge_tags(self, source_name: str, target_name: str) -> None:
        """Update tracking when tags are merged.

        The source tag is deleted by TickTick, so remove it from tracking.
        The target tag remains (add it if not already tracked).
        """
        source_lower = source_name.lower()
        target_lower = target_name.lower()
        self.tags.discard(source_lower)
        # Don't add target - it might be a pre-existing tag we shouldn't delete

    # -------------------------------------------------------------------------
    # Untrack operations (called when resources are explicitly deleted)
    # -------------------------------------------------------------------------

    def untrack_task(self, task_id: str) -> None:
        """Remove a task from tracking (it was explicitly deleted)."""
        self.tasks.pop(task_id, None)

    def untrack_project(self, project_id: str) -> None:
        """Remove a project from tracking (it was explicitly deleted)."""
        self.projects.discard(project_id)

    def untrack_folder(self, folder_id: str) -> None:
        """Remove a folder from tracking (it was explicitly deleted)."""
        self.folders.discard(folder_id)

    def untrack_tag(self, tag_name: str) -> None:
        """Remove a tag from tracking (it was explicitly deleted)."""
        self.tags.discard(tag_name.lower())

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    async def cleanup(self, client: TickTickClient) -> None:
        """Delete all tracked resources in the correct order.

        Order: tasks -> projects -> folders -> tags
        (respects dependencies - can't delete project with tasks, etc.)
        """
        # Delete tasks first (they belong to projects)
        # Note: We fetch each task first to get the CURRENT project_id,
        # because it may have been moved since we tracked it.
        for task_id, tracked_project_id in list(self.tasks.items()):
            try:
                # Try to get the task's current state for accurate project_id
                task = await client.get_task(task_id)
                actual_project_id = task.project_id
                await client.delete_task(task_id, actual_project_id)
            except Exception:
                # If get_task fails, try with tracked project_id as fallback
                try:
                    await client.delete_task(task_id, tracked_project_id)
                except Exception:
                    pass  # Ignore errors during cleanup (already deleted, etc.)

        # Delete projects (they may belong to folders)
        for project_id in list(self.projects):
            try:
                await client.delete_project(project_id)
            except Exception:
                pass

        # Delete folders
        for folder_id in list(self.folders):
            try:
                await client.delete_folder(folder_id)
            except Exception:
                pass

        # Delete tags
        for tag_name in list(self.tags):
            try:
                await client.delete_tag(tag_name)
            except Exception:
                pass

        # Clear all tracking
        self.tasks.clear()
        self.projects.clear()
        self.folders.clear()
        self.tags.clear()


class TrackingClientWrapper:
    """Wraps TickTickClient to automatically track created/modified resources.

    This wrapper intercepts all resource-modifying operations to maintain
    accurate tracking for cleanup. It handles:

    - create_* methods: Track new resources
    - delete_* methods: Untrack deleted resources (avoid double-delete)
    - rename_tag: Update tag name in tracking
    - merge_tags: Remove source tag from tracking
    - move_task: Update task's project_id in tracking
    """

    def __init__(self, client: TickTickClient, tracker: LiveResourceTracker):
        self._client = client
        self._tracker = tracker

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped client."""
        return getattr(self._client, name)

    # -------------------------------------------------------------------------
    # Create operations - track new resources
    # -------------------------------------------------------------------------

    async def create_task(self, *args, **kwargs) -> Task:
        """Create a task and track it for cleanup."""
        task = await self._client.create_task(*args, **kwargs)
        self._tracker.track_task(task.id, task.project_id)
        return task

    async def quick_add(self, *args, **kwargs) -> Task:
        """Quick add a task and track it for cleanup."""
        task = await self._client.quick_add(*args, **kwargs)
        self._tracker.track_task(task.id, task.project_id)
        return task

    async def create_project(self, *args, **kwargs) -> Project:
        """Create a project and track it for cleanup."""
        project = await self._client.create_project(*args, **kwargs)
        self._tracker.track_project(project.id)
        return project

    async def create_folder(self, *args, **kwargs) -> ProjectGroup:
        """Create a folder and track it for cleanup."""
        folder = await self._client.create_folder(*args, **kwargs)
        self._tracker.track_folder(folder.id)
        return folder

    async def create_tag(self, *args, **kwargs) -> Tag:
        """Create a tag and track it for cleanup."""
        tag = await self._client.create_tag(*args, **kwargs)
        self._tracker.track_tag(tag.name)
        return tag

    # -------------------------------------------------------------------------
    # Delete operations - untrack to avoid double-delete during cleanup
    # -------------------------------------------------------------------------

    async def delete_task(self, task_id: str, project_id: str) -> None:
        """Delete a task and untrack it."""
        await self._client.delete_task(task_id, project_id)
        self._tracker.untrack_task(task_id)

    async def delete_project(self, project_id: str) -> None:
        """Delete a project and untrack it."""
        await self._client.delete_project(project_id)
        self._tracker.untrack_project(project_id)

    async def delete_folder(self, folder_id: str) -> None:
        """Delete a folder and untrack it."""
        await self._client.delete_folder(folder_id)
        self._tracker.untrack_folder(folder_id)

    async def delete_tag(self, tag_name: str) -> None:
        """Delete a tag and untrack it."""
        await self._client.delete_tag(tag_name)
        self._tracker.untrack_tag(tag_name)

    # -------------------------------------------------------------------------
    # Modify operations - update tracking to reflect changes
    # -------------------------------------------------------------------------

    async def rename_tag(self, old_name: str, new_name: str) -> None:
        """Rename a tag and update tracking."""
        await self._client.rename_tag(old_name, new_name)
        self._tracker.rename_tag(old_name, new_name)

    async def merge_tags(self, source_name: str, target_name: str) -> None:
        """Merge tags and update tracking (source is deleted)."""
        await self._client.merge_tags(source_name, target_name)
        self._tracker.merge_tags(source_name, target_name)

    async def move_task(
        self, task_id: str, from_project_id: str, to_project_id: str
    ) -> None:
        """Move a task and update tracking."""
        await self._client.move_task(task_id, from_project_id, to_project_id)
        self._tracker.update_task_project(task_id, to_project_id)


@pytest.fixture
async def client(request) -> AsyncIterator[TickTickClient]:
    """
    Create a TickTickClient.

    In normal mode: Uses mocked API (no real API calls).
    In live mode (--live): Uses real TickTick API with credentials from .env.
                           Creates a new client for each test.
                           Automatically cleans up created resources after test.

    Live mode requires these environment variables:
        - TICKTICK_CLIENT_ID
        - TICKTICK_CLIENT_SECRET
        - TICKTICK_ACCESS_TOKEN
        - TICKTICK_USERNAME
        - TICKTICK_PASSWORD
    """
    live_mode = request.config.getoption("--live")

    if live_mode:
        # Live mode: Create client with resource tracking for cleanup
        real_client = _create_live_client_sync()
        await real_client.connect()

        # Wrap client to track created resources
        tracker = LiveResourceTracker()
        wrapped_client = TrackingClientWrapper(real_client, tracker)

        yield wrapped_client  # type: ignore  # Tests use the wrapper

        # Clean up all created resources after test completes
        await tracker.cleanup(real_client)
        await real_client.disconnect()
    else:
        # Mock mode: Get mock_api via request.getfixturevalue to avoid eager evaluation
        mock_api = request.getfixturevalue("mock_api")

        with patch("ticktick_sdk.client.client.UnifiedTickTickAPI") as MockAPIClass:
            MockAPIClass.return_value = mock_api

            # Create client with dummy credentials
            client = TickTickClient(
                client_id="test_client_id",
                client_secret="test_client_secret",
                v1_access_token="test_access_token",
                username="test@example.com",
                password="test_password",
            )

            # Replace the internal API with our mock
            client._api = mock_api

            await client.connect()
            yield client
            await client.disconnect()


@pytest.fixture
async def seeded_client(request, seeded_mock_api: MockUnifiedAPI) -> AsyncIterator[TickTickClient]:
    """Create a TickTickClient with seeded test data (mock mode only)."""
    live_mode = request.config.getoption("--live")

    if live_mode:
        # In live mode, seeded_client uses real client with cleanup
        real_client = _create_live_client_sync()
        await real_client.connect()

        tracker = LiveResourceTracker()
        wrapped_client = TrackingClientWrapper(real_client, tracker)

        yield wrapped_client  # type: ignore

        await tracker.cleanup(real_client)
        await real_client.disconnect()
    else:
        with patch("ticktick_sdk.client.client.UnifiedTickTickAPI") as MockAPIClass:
            MockAPIClass.return_value = seeded_mock_api

            client = TickTickClient(
                client_id="test_client_id",
                client_secret="test_client_secret",
                v1_access_token="test_access_token",
                username="test@example.com",
                password="test_password",
            )

            client._api = seeded_mock_api

            await client.connect()
            yield client
            await client.disconnect()


# =============================================================================
# Factory Fixtures
# =============================================================================


@pytest.fixture
def task_factory() -> type[TaskFactory]:
    """Provide TaskFactory class."""
    return TaskFactory


@pytest.fixture
def project_factory() -> type[ProjectFactory]:
    """Provide ProjectFactory class."""
    return ProjectFactory


@pytest.fixture
def folder_factory() -> type[FolderFactory]:
    """Provide FolderFactory class."""
    return FolderFactory


@pytest.fixture
def tag_factory() -> type[TagFactory]:
    """Provide TagFactory class."""
    return TagFactory


@pytest.fixture
def user_factory() -> type[UserFactory]:
    """Provide UserFactory class."""
    return UserFactory


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_task() -> Task:
    """Create a sample task."""
    return TaskFactory.create(title="Sample Task")


@pytest.fixture
def sample_project() -> Project:
    """Create a sample project."""
    return ProjectFactory.create(name="Sample Project")


@pytest.fixture
def sample_folder() -> ProjectGroup:
    """Create a sample folder."""
    return FolderFactory.create(name="Sample Folder")


@pytest.fixture
def sample_tag() -> Tag:
    """Create a sample tag."""
    return TagFactory.create(label="SampleTag")


@pytest.fixture
def sample_user() -> User:
    """Create a sample user."""
    return UserFactory.create()


@pytest.fixture
def sample_tasks() -> list[Task]:
    """Create a variety of sample tasks."""
    return [
        TaskFactory.create(title="Normal Task"),
        TaskFactory.create_with_due_date(1, title="Due Tomorrow"),
        TaskFactory.create_with_due_date(-1, title="Overdue Task"),
        TaskFactory.create_completed(title="Completed Task"),
        TaskFactory.create_with_subtasks(3, title="Task with Subtasks"),
        TaskFactory.create_with_tags(["work", "urgent"], title="Tagged Task"),
        TaskFactory.create_recurring(title="Recurring Task"),
        TaskFactory.create(title="High Priority", priority=TaskPriority.HIGH),
        TaskFactory.create(title="Low Priority", priority=TaskPriority.LOW),
    ]


@pytest.fixture
def priority_tasks() -> list[Task]:
    """Create tasks of each priority level."""
    return TaskFactory.create_priority_set()


# =============================================================================
# Parametrization Helpers
# =============================================================================


PRIORITY_LEVELS = [
    (TaskPriority.NONE, "none"),
    (TaskPriority.LOW, "low"),
    (TaskPriority.MEDIUM, "medium"),
    (TaskPriority.HIGH, "high"),
]

PRIORITY_NAMES = ["none", "low", "medium", "high"]
PRIORITY_VALUES = [0, 1, 3, 5]

VIEW_MODES = ["list", "kanban", "timeline"]
PROJECT_KINDS = ["TASK", "NOTE"]

TASK_STATUSES = [
    (TaskStatus.ABANDONED, "abandoned"),
    (TaskStatus.ACTIVE, "active"),
    (TaskStatus.COMPLETED, "completed"),
]
