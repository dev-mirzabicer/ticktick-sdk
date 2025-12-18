"""
Comprehensive Folder (Project Group) Operation Tests for TickTick Client.

This module tests all folder-related functionality including:
- Create, Delete
- List all folders
- Folder-project relationships
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.conftest import MockUnifiedAPI, FolderFactory
    from ticktick_sdk.client import TickTickClient


pytestmark = [pytest.mark.unit]


# =============================================================================
# Folder Creation Tests
# =============================================================================


class TestFolderCreation:
    """Tests for folder creation functionality."""

    async def test_create_folder_minimal(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test creating a folder with only name."""
        folder = await client.create_folder(name="Simple Folder")

        assert folder is not None
        assert folder.name == "Simple Folder"

    async def test_create_multiple_folders(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test creating multiple folders."""
        folders = []
        for i in range(5):
            folder = await client.create_folder(name=f"MultiFolder {i}")
            folders.append(folder)

        assert len(folders) == 5

        # All IDs should be unique
        ids = [f.id for f in folders]
        assert len(ids) == len(set(ids))

    async def test_create_folder_with_same_name(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test creating folders with the same name (allowed)."""
        folder1 = await client.create_folder(name="Same Name")
        folder2 = await client.create_folder(name="Same Name")

        assert folder1.id != folder2.id
        assert folder1.name == folder2.name

    async def test_create_folder_with_special_characters(self, client: TickTickClient):
        """Test creating folder with special characters in name."""
        folder = await client.create_folder(name="Work & Personal / Projects")

        assert folder.name == "Work & Personal / Projects"


# =============================================================================
# Folder Retrieval Tests
# =============================================================================


class TestFolderRetrieval:
    """Tests for folder retrieval functionality."""

    async def test_get_all_folders(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test getting all folders."""
        folder1 = await client.create_folder(name="Folder 1")
        folder2 = await client.create_folder(name="Folder 2")
        folder3 = await client.create_folder(name="Folder 3")

        folders = await client.get_all_folders()

        # Verify created folders are in the list
        folder_ids = [f.id for f in folders]
        assert folder1.id in folder_ids
        assert folder2.id in folder_ids
        assert folder3.id in folder_ids
        assert len(folders) >= 3

    @pytest.mark.mock_only
    async def test_get_all_folders_empty(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test getting folders when none exist.

        Mock-only because live accounts have existing data.
        """
        folders = await client.get_all_folders()

        assert folders == []


# =============================================================================
# Folder Deletion Tests
# =============================================================================


class TestFolderDeletion:
    """Tests for folder deletion functionality."""

    async def test_delete_folder(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test deleting a folder."""
        folder = await client.create_folder(name="Folder to Delete")
        folder_id = folder.id

        await client.delete_folder(folder_id)

        # Verify folder is no longer in the list
        folders = await client.get_all_folders()
        folder_ids = [f.id for f in folders]
        assert folder_id not in folder_ids

    async def test_delete_folder_projects_still_exist(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test that deleting a folder doesn't delete its projects.

        Note: TickTick does NOT automatically ungroup projects when their folder
        is deleted. Projects retain their group_id as a "dangling reference" to
        the deleted folder. The important behavior is that projects remain accessible.
        """
        folder = await client.create_folder(name="Folder")
        project1 = await client.create_project(name="Project 1", folder_id=folder.id)
        project2 = await client.create_project(name="Project 2", folder_id=folder.id)

        await client.delete_folder(folder.id)

        # Projects should still exist and be accessible
        retrieved_project1 = await client.get_project(project1.id)
        retrieved_project2 = await client.get_project(project2.id)
        assert retrieved_project1 is not None
        assert retrieved_project2 is not None
        assert retrieved_project1.name == "Project 1"
        assert retrieved_project2.name == "Project 2"

    async def test_delete_nonexistent_folder(self, client: TickTickClient):
        """Test deleting a folder that doesn't exist."""
        from ticktick_sdk.exceptions import TickTickNotFoundError

        with pytest.raises(TickTickNotFoundError):
            await client.delete_folder("nonexistent_folder_id")

    async def test_delete_empty_folder(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test deleting an empty folder."""
        folder = await client.create_folder(name="Empty Folder")

        await client.delete_folder(folder.id)

        # Verify folder is no longer in the list
        folders = await client.get_all_folders()
        folder_ids = [f.id for f in folders]
        assert folder.id not in folder_ids


# =============================================================================
# Folder-Project Relationship Tests
# =============================================================================


class TestFolderProjectRelationships:
    """Tests for folder-project relationships."""

    async def test_add_project_to_folder(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test adding a project to a folder."""
        folder = await client.create_folder(name="Work")
        project = await client.create_project(name="Project", folder_id=folder.id)

        assert project.group_id == folder.id

    async def test_multiple_projects_in_folder(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test multiple projects in one folder."""
        folder = await client.create_folder(name="Work")

        project1 = await client.create_project(name="Project 1", folder_id=folder.id)
        project2 = await client.create_project(name="Project 2", folder_id=folder.id)
        project3 = await client.create_project(name="Project 3", folder_id=folder.id)

        assert project1.group_id == folder.id
        assert project2.group_id == folder.id
        assert project3.group_id == folder.id

    async def test_projects_in_different_folders(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test projects organized in different folders."""
        work = await client.create_folder(name="Work")
        personal = await client.create_folder(name="Personal")
        learning = await client.create_folder(name="Learning")

        work_project = await client.create_project(name="Work Project", folder_id=work.id)
        personal_project = await client.create_project(name="Personal Project", folder_id=personal.id)
        learning_project = await client.create_project(name="Learning Project", folder_id=learning.id)

        assert work_project.group_id == work.id
        assert personal_project.group_id == personal.id
        assert learning_project.group_id == learning.id

    async def test_mixed_grouped_and_ungrouped_projects(
        self,
        client: TickTickClient,
        mock_api: MockUnifiedAPI,
    ):
        """Test mix of grouped and ungrouped projects."""
        folder = await client.create_folder(name="Folder")

        grouped_project = await client.create_project(name="Grouped", folder_id=folder.id)
        ungrouped_project = await client.create_project(name="Ungrouped")

        assert grouped_project.group_id == folder.id
        assert ungrouped_project.group_id is None


# =============================================================================
# Folder Combination Tests
# =============================================================================


class TestFolderCombinations:
    """Tests for combinations of folder operations."""

    async def test_folder_lifecycle_with_projects(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test full folder lifecycle with projects."""
        # Create folder
        folder = await client.create_folder(name="Lifecycle Folder")

        # Add projects
        project1 = await client.create_project(name="Project 1", folder_id=folder.id)
        project2 = await client.create_project(name="Project 2", folder_id=folder.id)

        # Verify organization
        assert project1.group_id == folder.id
        assert project2.group_id == folder.id

        # Delete folder
        await client.delete_folder(folder.id)

        # Verify folder deleted but projects remain accessible
        folders = await client.get_all_folders()
        assert folder.id not in [f.id for f in folders]

        # Projects should still exist
        retrieved_p1 = await client.get_project(project1.id)
        retrieved_p2 = await client.get_project(project2.id)
        assert retrieved_p1 is not None
        assert retrieved_p2 is not None

    async def test_complex_folder_structure(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test complex folder structure with tasks."""
        # Create folder structure
        work = await client.create_folder(name="Work")
        personal = await client.create_folder(name="Personal")

        # Create projects in folders
        work_project1 = await client.create_project(name="Client A", folder_id=work.id)
        work_project2 = await client.create_project(name="Client B", folder_id=work.id)
        personal_project = await client.create_project(name="Home", folder_id=personal.id)
        inbox_tasks = await client.create_project(name="Quick Tasks")  # No folder

        # Add tasks
        task1 = await client.create_task(title="Client A Task", project_id=work_project1.id)
        task2 = await client.create_task(title="Client B Task", project_id=work_project2.id)
        task3 = await client.create_task(title="Home Task", project_id=personal_project.id)
        task4 = await client.create_task(title="Quick Task", project_id=inbox_tasks.id)

        # Verify folder structure via client API
        folders = await client.get_all_folders()
        folder_ids = [f.id for f in folders]
        assert work.id in folder_ids
        assert personal.id in folder_ids

        # Verify projects have correct group assignments
        assert work_project1.group_id == work.id
        assert work_project2.group_id == work.id
        assert personal_project.group_id == personal.id
        assert inbox_tasks.group_id is None

        # Verify tasks are accessible
        assert task1 is not None
        assert task2 is not None
        assert task3 is not None
        assert task4 is not None

    async def test_delete_folder_preserves_tasks(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test that deleting folder preserves tasks in its projects."""
        folder = await client.create_folder(name="Folder")
        project = await client.create_project(name="Project", folder_id=folder.id)

        task1 = await client.create_task(title="Task 1", project_id=project.id)
        task2 = await client.create_task(title="Task 2", project_id=project.id)

        await client.delete_folder(folder.id)

        # Tasks should still exist and be accessible via client API
        retrieved_task1 = await client.get_task(task1.id)
        retrieved_task2 = await client.get_task(task2.id)
        assert retrieved_task1 is not None
        assert retrieved_task2 is not None
        assert retrieved_task1.title == "Task 1"
        assert retrieved_task2.title == "Task 2"
