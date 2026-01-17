"""
Tests for kanban column management functionality.

Tests the column CRUD operations and task-column assignment
in TickTickClient.

Live Mode:
    These tests work in both mock and live mode.
    In live mode, projects/tasks/columns are created via the client
    and cleaned up automatically.
"""

from __future__ import annotations

import pytest

from ticktick_sdk import TickTickClient
from ticktick_sdk.exceptions import TickTickNotFoundError
from ticktick_sdk.models import Column

from .conftest import ColumnFactory, MockUnifiedAPI, TaskFactory

pytestmark = [pytest.mark.columns, pytest.mark.unit]


class TestColumnListing:
    """Tests for listing kanban columns."""

    async def test_list_columns_empty(self, client: TickTickClient):
        """Test listing columns when project has no columns."""
        project = await client.create_project(name="Empty Kanban", view_mode="kanban")

        columns = await client.get_columns(project.id)

        assert columns == []

    async def test_list_columns_with_data(self, client: TickTickClient):
        """Test listing columns when project has columns."""
        project = await client.create_project(name="Kanban Project", view_mode="kanban")

        # Create columns
        await client.create_column(project.id, "To Do", sort_order=0)
        await client.create_column(project.id, "In Progress", sort_order=1)
        await client.create_column(project.id, "Done", sort_order=2)

        columns = await client.get_columns(project.id)

        assert len(columns) == 3
        column_names = [c.name for c in columns]
        assert "To Do" in column_names
        assert "In Progress" in column_names
        assert "Done" in column_names

    @pytest.mark.mock_only
    async def test_list_columns_verifies_api_call(
        self, client: TickTickClient, mock_api: MockUnifiedAPI
    ):
        """Test that list_columns makes the correct API call (mock only)."""
        project = await client.create_project(name="Kanban Project", view_mode="kanban")

        await client.get_columns(project.id)

        calls = mock_api.get_calls("list_columns")
        assert len(calls) == 1
        args, _ = calls[0]
        assert args == (project.id,)


class TestColumnCreation:
    """Tests for creating kanban columns."""

    async def test_create_column_minimal(self, client: TickTickClient):
        """Test creating a column with only required fields."""
        project = await client.create_project(name="Kanban Project", view_mode="kanban")

        column = await client.create_column(project.id, "To Do")

        assert column is not None
        assert column.name == "To Do"
        assert column.project_id == project.id
        assert column.id is not None

    async def test_create_column_with_sort_order(self, client: TickTickClient):
        """Test creating a column with explicit sort order."""
        project = await client.create_project(name="Kanban Project", view_mode="kanban")

        column = await client.create_column(project.id, "In Progress", sort_order=5)

        assert column.name == "In Progress"
        assert column.sort_order == 5

    async def test_create_multiple_columns(self, client: TickTickClient):
        """Test creating multiple columns in a project."""
        project = await client.create_project(name="Kanban Project", view_mode="kanban")

        await client.create_column(project.id, "Backlog", sort_order=0)
        await client.create_column(project.id, "To Do", sort_order=1)
        await client.create_column(project.id, "In Progress", sort_order=2)
        await client.create_column(project.id, "Review", sort_order=3)
        await client.create_column(project.id, "Done", sort_order=4)

        columns = await client.get_columns(project.id)

        assert len(columns) == 5
        # Verify all columns exist
        column_names = [c.name for c in columns]
        assert set(column_names) == {"Backlog", "To Do", "In Progress", "Review", "Done"}

    @pytest.mark.mock_only
    async def test_create_column_verifies_api_call(
        self, client: TickTickClient, mock_api: MockUnifiedAPI
    ):
        """Test that create_column makes the correct API call (mock only)."""
        project = await client.create_project(name="Kanban Project", view_mode="kanban")

        await client.create_column(project.id, "My Column", sort_order=10)

        calls = mock_api.get_calls("create_column")
        assert len(calls) == 1
        args, kwargs = calls[0]
        assert args == (project.id, "My Column")
        assert kwargs.get("sort_order") == 10


class TestColumnUpdate:
    """Tests for updating kanban columns."""

    async def test_update_column_name(self, client: TickTickClient):
        """Test updating a column's name."""
        project = await client.create_project(name="Kanban Project", view_mode="kanban")
        column = await client.create_column(project.id, "Original Name")

        updated = await client.update_column(column.id, project.id, name="New Name")

        assert updated.name == "New Name"
        assert updated.id == column.id

    async def test_update_column_sort_order(self, client: TickTickClient):
        """Test updating a column's sort order."""
        project = await client.create_project(name="Kanban Project", view_mode="kanban")
        column = await client.create_column(project.id, "Column", sort_order=0)

        updated = await client.update_column(column.id, project.id, sort_order=10)

        assert updated.sort_order == 10
        assert updated.id == column.id

    async def test_update_column_both_fields(self, client: TickTickClient):
        """Test updating both name and sort order."""
        project = await client.create_project(name="Kanban Project", view_mode="kanban")
        column = await client.create_column(project.id, "Old Name", sort_order=0)

        updated = await client.update_column(
            column.id, project.id, name="New Name", sort_order=99
        )

        assert updated.name == "New Name"
        assert updated.sort_order == 99

    async def test_update_nonexistent_column(self, client: TickTickClient):
        """Test updating a column that doesn't exist."""
        project = await client.create_project(name="Kanban Project", view_mode="kanban")

        with pytest.raises(TickTickNotFoundError):
            await client.update_column(
                "nonexistent_column_id_000", project.id, name="New Name"
            )


class TestColumnDeletion:
    """Tests for deleting kanban columns."""

    async def test_delete_column(self, client: TickTickClient):
        """Test deleting a column."""
        project = await client.create_project(name="Kanban Project", view_mode="kanban")
        column = await client.create_column(project.id, "To Delete")

        # Verify column exists
        columns_before = await client.get_columns(project.id)
        assert len(columns_before) == 1

        # Delete the column
        await client.delete_column(column.id, project.id)

        # Verify column is deleted
        columns_after = await client.get_columns(project.id)
        assert len(columns_after) == 0

    async def test_delete_one_of_multiple_columns(self, client: TickTickClient):
        """Test deleting one column while others remain."""
        project = await client.create_project(name="Kanban Project", view_mode="kanban")
        col1 = await client.create_column(project.id, "Keep", sort_order=0)
        col2 = await client.create_column(project.id, "Delete", sort_order=1)
        col3 = await client.create_column(project.id, "Also Keep", sort_order=2)

        # Delete the middle column
        await client.delete_column(col2.id, project.id)

        # Verify only two columns remain
        columns = await client.get_columns(project.id)
        assert len(columns) == 2
        column_ids = [c.id for c in columns]
        assert col1.id in column_ids
        assert col2.id not in column_ids
        assert col3.id in column_ids

    @pytest.mark.mock_only
    async def test_delete_column_verifies_api_call(
        self, client: TickTickClient, mock_api: MockUnifiedAPI
    ):
        """Test that delete_column makes the correct API call (mock only)."""
        project = await client.create_project(name="Kanban Project", view_mode="kanban")
        column = await client.create_column(project.id, "To Delete")

        await client.delete_column(column.id, project.id)

        calls = mock_api.get_calls("delete_column")
        assert len(calls) == 1
        args, _ = calls[0]
        assert args == (column.id, project.id)


class TestMoveTaskToColumn:
    """Tests for moving tasks to kanban columns."""

    async def test_move_task_to_column(self, client: TickTickClient):
        """Test moving a task to a specific column."""
        project = await client.create_project(name="Kanban Project", view_mode="kanban")
        column = await client.create_column(project.id, "To Do")
        task = await client.create_task(title="Task to Move", project_id=project.id)

        updated_task = await client.move_task_to_column(task.id, project.id, column.id)

        assert updated_task.column_id == column.id

    async def test_remove_task_from_column(self, client: TickTickClient):
        """Test removing a task from its column (set column_id to None)."""
        project = await client.create_project(name="Kanban Project", view_mode="kanban")
        column = await client.create_column(project.id, "To Do")
        task = await client.create_task(title="Task in Column", project_id=project.id)

        # Move to column first
        await client.move_task_to_column(task.id, project.id, column.id)

        # Remove from column by setting column_id to None
        updated_task = await client.move_task_to_column(task.id, project.id, None)

        assert updated_task.column_id is None

    async def test_move_task_between_columns(self, client: TickTickClient):
        """Test moving a task from one column to another."""
        project = await client.create_project(name="Kanban Project", view_mode="kanban")
        col1 = await client.create_column(project.id, "To Do", sort_order=0)
        col2 = await client.create_column(project.id, "In Progress", sort_order=1)
        task = await client.create_task(title="Moving Task", project_id=project.id)

        # Move to first column
        await client.move_task_to_column(task.id, project.id, col1.id)

        # Move to second column
        updated_task = await client.move_task_to_column(task.id, project.id, col2.id)

        assert updated_task.column_id == col2.id

    async def test_move_nonexistent_task_to_column(self, client: TickTickClient):
        """Test moving a nonexistent task raises NotFoundError."""
        project = await client.create_project(name="Kanban Project", view_mode="kanban")
        column = await client.create_column(project.id, "To Do")

        with pytest.raises(TickTickNotFoundError):
            await client.move_task_to_column(
                "nonexistent_task_id_0000", project.id, column.id
            )

    @pytest.mark.mock_only
    async def test_move_task_to_column_verifies_api_call(
        self, client: TickTickClient, mock_api: MockUnifiedAPI
    ):
        """Test that move_task_to_column makes the correct API call (mock only)."""
        project = await client.create_project(name="Kanban Project", view_mode="kanban")
        column = await client.create_column(project.id, "To Do")
        task = TaskFactory.create(title="API Test Task", project_id=project.id)
        mock_api.tasks[task.id] = task

        await client.move_task_to_column(task.id, project.id, column.id)

        calls = mock_api.get_calls("move_task_to_column")
        assert len(calls) == 1
        args, _ = calls[0]
        assert args == (task.id, project.id, column.id)


class TestColumnFactory:
    """Tests for the ColumnFactory test utility."""

    def test_column_factory_create_default(self):
        """Test ColumnFactory creates valid default column."""
        column = ColumnFactory.create()

        assert column is not None
        assert isinstance(column, Column)
        assert column.id is not None
        assert column.project_id is not None
        assert column.name == "To Do"  # Default name
        assert column.sort_order == 0

    def test_column_factory_create_custom(self):
        """Test ColumnFactory with custom values."""
        column = ColumnFactory.create(
            id="custom_column_id_0000001",
            project_id="custom_project_id_0001",
            name="Custom Column",
            sort_order=5,
        )

        assert column.id == "custom_column_id_0000001"
        assert column.project_id == "custom_project_id_0001"
        assert column.name == "Custom Column"
        assert column.sort_order == 5

    def test_column_factory_create_kanban_set(self):
        """Test ColumnFactory.create_kanban_set creates standard columns."""
        project_id = "test_project_id_0000001"
        columns = ColumnFactory.create_kanban_set(project_id)

        assert len(columns) == 3
        assert all(c.project_id == project_id for c in columns)

        column_names = [c.name for c in columns]
        assert "To Do" in column_names
        assert "In Progress" in column_names
        assert "Done" in column_names

        # Verify sort order
        sorted_columns = sorted(columns, key=lambda c: c.sort_order or 0)
        assert sorted_columns[0].name == "To Do"
        assert sorted_columns[1].name == "In Progress"
        assert sorted_columns[2].name == "Done"


class TestKanbanWorkflow:
    """Integration tests for complete kanban workflows."""

    async def test_full_kanban_workflow(self, client: TickTickClient):
        """Test complete kanban workflow: create columns, add tasks, move through stages."""
        # Create kanban project
        project = await client.create_project(
            name="Sprint Board", view_mode="kanban"
        )

        # Create columns
        todo = await client.create_column(project.id, "To Do", sort_order=0)
        progress = await client.create_column(project.id, "In Progress", sort_order=1)
        done = await client.create_column(project.id, "Done", sort_order=2)

        # Create tasks
        task1 = await client.create_task(title="Implement feature", project_id=project.id)
        task2 = await client.create_task(title="Write tests", project_id=project.id)
        task3 = await client.create_task(title="Review code", project_id=project.id)

        # Move tasks to To Do
        await client.move_task_to_column(task1.id, project.id, todo.id)
        await client.move_task_to_column(task2.id, project.id, todo.id)
        await client.move_task_to_column(task3.id, project.id, todo.id)

        # Move task1 to In Progress
        updated_task1 = await client.move_task_to_column(task1.id, project.id, progress.id)
        assert updated_task1.column_id == progress.id

        # Move task1 to Done
        final_task1 = await client.move_task_to_column(task1.id, project.id, done.id)
        assert final_task1.column_id == done.id

        # Verify column structure
        columns = await client.get_columns(project.id)
        assert len(columns) == 3

    async def test_reorganize_columns(self, client: TickTickClient):
        """Test reorganizing column order."""
        project = await client.create_project(name="Kanban", view_mode="kanban")

        # Create columns in initial order
        await client.create_column(project.id, "A", sort_order=0)
        await client.create_column(project.id, "B", sort_order=1)
        col3 = await client.create_column(project.id, "C", sort_order=2)

        # Reorganize: move C to the front
        await client.update_column(col3.id, project.id, sort_order=-1)

        # Verify the update was applied
        columns = await client.get_columns(project.id)
        updated_col3 = next(c for c in columns if c.id == col3.id)
        assert updated_col3.sort_order == -1


class TestGetTasksByColumn:
    """Tests for retrieving tasks by column."""

    async def test_get_tasks_by_column_basic(self, client: TickTickClient):
        """Test getting tasks from a column."""
        project = await client.create_project(name="Kanban Project", view_mode="kanban")
        column = await client.create_column(project.id, "To Do")

        # Create tasks and move them to the column
        task1 = await client.create_task(title="Task 1", project_id=project.id)
        task2 = await client.create_task(title="Task 2", project_id=project.id)
        task3 = await client.create_task(title="Task 3", project_id=project.id)

        await client.move_task_to_column(task1.id, project.id, column.id)
        await client.move_task_to_column(task2.id, project.id, column.id)
        await client.move_task_to_column(task3.id, project.id, column.id)

        # Get tasks by column
        tasks = await client.get_tasks_by_column(column.id)

        assert len(tasks) == 3
        task_ids = [t.id for t in tasks]
        assert task1.id in task_ids
        assert task2.id in task_ids
        assert task3.id in task_ids

    async def test_get_tasks_by_column_with_project(self, client: TickTickClient):
        """Test filtering by both column and project."""
        project1 = await client.create_project(name="Project 1", view_mode="kanban")
        project2 = await client.create_project(name="Project 2", view_mode="kanban")

        col1 = await client.create_column(project1.id, "To Do")
        col2 = await client.create_column(project2.id, "In Progress")

        # Create tasks in different projects
        task1 = await client.create_task(title="Task 1", project_id=project1.id)
        task2 = await client.create_task(title="Task 2", project_id=project1.id)
        task3 = await client.create_task(title="Task 3", project_id=project2.id)

        await client.move_task_to_column(task1.id, project1.id, col1.id)
        await client.move_task_to_column(task2.id, project1.id, col1.id)
        await client.move_task_to_column(task3.id, project2.id, col2.id)

        # Get tasks from col1 with project1 filter
        tasks = await client.get_tasks_by_column(col1.id, project_id=project1.id)

        assert len(tasks) == 2
        task_ids = [t.id for t in tasks]
        assert task1.id in task_ids
        assert task2.id in task_ids
        assert task3.id not in task_ids

    async def test_get_tasks_by_column_empty(self, client: TickTickClient):
        """Test with column that has no tasks."""
        project = await client.create_project(name="Kanban Project", view_mode="kanban")
        column = await client.create_column(project.id, "Empty Column")

        # Don't add any tasks to the column
        tasks = await client.get_tasks_by_column(column.id)

        assert tasks == []

    async def test_get_tasks_by_column_invalid_column_id(self, client: TickTickClient):
        """Test with invalid column_id returns empty list."""
        tasks = await client.get_tasks_by_column("nonexistent_column_id")
        assert tasks == []

    @pytest.mark.mock_only
    async def test_get_tasks_by_column_verifies_api_call(
        self, client: TickTickClient, mock_api: MockUnifiedAPI
    ):
        """Test that get_tasks_by_column makes the correct API call (mock only)."""
        project = await client.create_project(name="Kanban Project", view_mode="kanban")
        column = await client.create_column(project.id, "To Do")

        await client.get_tasks_by_column(column.id)

        calls = mock_api.get_calls("list_tasks_by_column")
        assert len(calls) == 1
        args, kwargs = calls[0]
        assert args == (column.id,)
        assert kwargs.get("project_id") is None
