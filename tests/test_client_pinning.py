"""
Tests for task pinning functionality.

Tests the pin_task() and unpin_task() methods in TickTickClient.

Live Mode:
    These tests work in both mock and live mode.
    In live mode, tasks are created via the client and cleaned up automatically.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ticktick_sdk import TickTickClient
from ticktick_sdk.exceptions import TickTickNotFoundError

from .conftest import MockUnifiedAPI, TaskFactory

pytestmark = [pytest.mark.pinning, pytest.mark.unit]


class TestTaskPinning:
    """Tests for task pinning operations."""

    async def test_pin_task(self, client: TickTickClient):
        """Test pinning a task sets pinned_time."""
        # Create a real task through the client
        task = await client.create_task(title="Task to Pin")
        assert task.pinned_time is None
        assert task.is_pinned is False

        # Pin the task
        pinned_task = await client.pin_task(task.id, task.project_id)

        assert pinned_task.pinned_time is not None
        assert pinned_task.is_pinned is True
        assert isinstance(pinned_task.pinned_time, datetime)

    async def test_unpin_task(self, client: TickTickClient):
        """Test unpinning a task clears pinned_time."""
        # Create and pin a task
        task = await client.create_task(title="Pinned Task")
        pinned_task = await client.pin_task(task.id, task.project_id)
        assert pinned_task.is_pinned is True

        # Unpin the task
        unpinned_task = await client.unpin_task(task.id, task.project_id)

        assert unpinned_task.pinned_time is None
        assert unpinned_task.is_pinned is False

    async def test_pin_already_pinned_task(self, client: TickTickClient):
        """Test pinning an already pinned task updates the pinned_time."""
        # Create and pin a task
        task = await client.create_task(title="Already Pinned Task")
        first_pin = await client.pin_task(task.id, task.project_id)
        first_pinned_time = first_pin.pinned_time

        # Pin again
        second_pin = await client.pin_task(task.id, task.project_id)

        # Should still be pinned (with potentially updated time)
        assert second_pin.is_pinned is True
        assert second_pin.pinned_time is not None

    async def test_unpin_not_pinned_task(self, client: TickTickClient):
        """Test unpinning a task that's not pinned."""
        # Create a task that's not pinned
        task = await client.create_task(title="Not Pinned Task")
        assert task.is_pinned is False

        # Unpin (should work fine, just no-op)
        unpinned_task = await client.unpin_task(task.id, task.project_id)

        assert unpinned_task.pinned_time is None
        assert unpinned_task.is_pinned is False

    async def test_pin_nonexistent_task(self, client: TickTickClient):
        """Test pinning a task that doesn't exist raises NotFoundError."""
        with pytest.raises(TickTickNotFoundError):
            await client.pin_task("nonexistent_id_00000000", "some_project_id_000000")

    async def test_unpin_nonexistent_task(self, client: TickTickClient):
        """Test unpinning a task that doesn't exist raises NotFoundError."""
        with pytest.raises(TickTickNotFoundError):
            await client.unpin_task("nonexistent_id_00000000", "some_project_id_000000")

    async def test_pin_preserves_other_task_fields(self, client: TickTickClient):
        """Test that pinning preserves other task fields."""
        # Create a task with various fields set
        task = await client.create_task(
            title="Full Task",
            content="Task description",
            priority=5,
            tags=["important", "work"],
        )
        original_title = task.title
        original_content = task.content
        original_priority = task.priority
        original_tags = task.tags.copy() if task.tags else []

        # Pin the task
        pinned_task = await client.pin_task(task.id, task.project_id)

        # Verify other fields are preserved
        assert pinned_task.title == original_title
        assert pinned_task.content == original_content
        assert pinned_task.priority == original_priority
        assert pinned_task.tags == original_tags
        assert pinned_task.is_pinned is True

    async def test_unpin_preserves_other_task_fields(self, client: TickTickClient):
        """Test that unpinning preserves other task fields."""
        # Create and pin a task with various fields set
        task = await client.create_task(
            title="Pinned Full Task",
            content="Task description",
            priority=3,
            tags=["personal"],
        )
        pinned_task = await client.pin_task(task.id, task.project_id)
        original_title = pinned_task.title
        original_content = pinned_task.content
        original_priority = pinned_task.priority
        original_tags = pinned_task.tags.copy() if pinned_task.tags else []

        # Unpin the task
        unpinned_task = await client.unpin_task(task.id, task.project_id)

        # Verify other fields are preserved
        assert unpinned_task.title == original_title
        assert unpinned_task.content == original_content
        assert unpinned_task.priority == original_priority
        assert unpinned_task.tags == original_tags
        assert unpinned_task.is_pinned is False


@pytest.mark.mock_only
class TestTaskPinningAPIInteraction:
    """Tests for pin/unpin API call tracking (mock mode only)."""

    async def test_pin_task_calls_api(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test that pin_task makes the correct API call."""
        task = TaskFactory.create(title="API Test Task")
        mock_api.tasks[task.id] = task

        await client.pin_task(task.id, task.project_id)

        # Verify API was called with correct arguments
        calls = mock_api.get_calls("pin_task")
        assert len(calls) == 1
        args, kwargs = calls[0]
        assert args == (task.id, task.project_id)

    async def test_unpin_task_calls_api(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test that unpin_task makes the correct API call."""
        task = TaskFactory.create(title="API Test Task")
        task.pinned_time = datetime.now(timezone.utc)
        mock_api.tasks[task.id] = task

        await client.unpin_task(task.id, task.project_id)

        # Verify API was called with correct arguments
        calls = mock_api.get_calls("unpin_task")
        assert len(calls) == 1
        args, kwargs = calls[0]
        assert args == (task.id, task.project_id)


class TestTaskPinningWithProjects:
    """Tests for pinning tasks in different project contexts."""

    async def test_pin_task_in_specific_project(self, client: TickTickClient):
        """Test pinning a task in a specific project."""
        # Create a project and task
        project = await client.create_project(name="Test Pin Project")
        task = await client.create_task(title="Project Task", project_id=project.id)

        # Pin the task
        pinned_task = await client.pin_task(task.id, project.id)

        assert pinned_task.is_pinned is True
        assert pinned_task.project_id == project.id

    async def test_pin_multiple_tasks_in_same_project(self, client: TickTickClient):
        """Test pinning multiple tasks in the same project."""
        project = await client.create_project(name="Multi-Pin Project")

        # Create and pin multiple tasks
        task1 = await client.create_task(title="Task 1", project_id=project.id)
        task2 = await client.create_task(title="Task 2", project_id=project.id)
        task3 = await client.create_task(title="Task 3", project_id=project.id)

        # Pin all three tasks
        pinned1 = await client.pin_task(task1.id, project.id)
        pinned2 = await client.pin_task(task2.id, project.id)
        pinned3 = await client.pin_task(task3.id, project.id)

        assert pinned1.is_pinned is True
        assert pinned2.is_pinned is True
        assert pinned3.is_pinned is True
