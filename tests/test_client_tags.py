"""
Comprehensive Tag Operation Tests for TickTick Client.

This module tests all tag-related functionality including:
- Create, Delete, Rename, Merge
- List all tags
- Tag-task relationships
- Tag hierarchy (parent/child)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.conftest import MockUnifiedAPI, TagFactory
    from ticktick_sdk.client import TickTickClient


pytestmark = [pytest.mark.tags, pytest.mark.unit]


# =============================================================================
# Tag Creation Tests
# =============================================================================


class TestTagCreation:
    """Tests for tag creation functionality."""

    async def test_create_tag_minimal(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test creating a tag with only name."""
        tag = await client.create_tag(name="SimpleTag")

        assert tag is not None
        assert tag.label == "SimpleTag"
        assert tag.name == "simpletag"  # Lowercase

    async def test_create_tag_with_color(self, client: TickTickClient):
        """Test creating a tag with color."""
        tag = await client.create_tag(name="ColoredTag", color="#F18181")

        assert tag.color == "#F18181"

    @pytest.mark.parametrize("color", [
        "#F18181",
        "#86BB6D",
        "#4CAFF6",
        "#FFD966",
        "#9C87E0",
        None,
    ])
    async def test_create_tag_various_colors(self, client: TickTickClient, color: str | None):
        """Test creating tags with various colors."""
        tag = await client.create_tag(name=f"Tag_{color}", color=color)

        if color:
            assert tag.color == color

    async def test_create_tag_with_parent(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test creating a nested tag."""
        parent = await client.create_tag(name="ParentTag")
        child = await client.create_tag(name="ChildTag", parent=parent.name)

        assert child.parent == parent.name

    async def test_create_multiple_tags(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test creating multiple tags."""
        tags = []
        for label in ["Work", "Personal", "Urgent", "Later"]:
            tag = await client.create_tag(name=label)
            tags.append(tag)

        assert len(tags) == 4

        # All names should be unique
        names = [t.name for t in tags]
        assert len(names) == len(set(names))

    async def test_create_tag_with_spaces(self, client: TickTickClient):
        """Test creating tag with spaces in name."""
        tag = await client.create_tag(name="High Priority")

        assert tag.label == "High Priority"
        # Name might be normalized (depends on implementation)

    async def test_create_tag_hierarchy(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test creating tag hierarchy."""
        work = await client.create_tag(name="Work")
        client_a = await client.create_tag(name="ClientA", parent=work.name)
        client_b = await client.create_tag(name="ClientB", parent=work.name)

        assert client_a.parent == work.name
        assert client_b.parent == work.name


# =============================================================================
# Tag Retrieval Tests
# =============================================================================


class TestTagRetrieval:
    """Tests for tag retrieval functionality."""

    async def test_get_all_tags(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test getting all tags."""
        tag1 = await client.create_tag(name="Tag1")
        tag2 = await client.create_tag(name="Tag2")
        tag3 = await client.create_tag(name="Tag3")

        tags = await client.get_all_tags()

        # Verify created tags are in the list
        tag_names = [t.name for t in tags]
        assert tag1.name in tag_names
        assert tag2.name in tag_names
        assert tag3.name in tag_names

    @pytest.mark.mock_only
    async def test_get_all_tags_empty(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test getting tags when none exist.

        Mock-only because live accounts may have existing tags.
        """
        tags = await client.get_all_tags()

        assert tags == []

    async def test_get_all_tags_includes_hierarchy(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test that getting tags includes parent-child relationships."""
        parent = await client.create_tag(name="Parent")
        child = await client.create_tag(name="Child", parent=parent.name)

        tags = await client.get_all_tags()

        child_tag = next(t for t in tags if t.name == child.name)
        assert child_tag.parent == parent.name


# =============================================================================
# Tag Deletion Tests
# =============================================================================


class TestTagDeletion:
    """Tests for tag deletion functionality."""

    async def test_delete_tag(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test deleting a tag."""
        tag = await client.create_tag(name="TagToDelete")
        tag_name = tag.name

        await client.delete_tag(tag_name)

        # Verify tag is gone via client API
        tags = await client.get_all_tags()
        tag_names = [t.name for t in tags]
        assert tag_name not in tag_names

    async def test_delete_tag_removes_from_tasks(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test that deleting a tag removes it from tasks."""
        tag = await client.create_tag(name="TagToRemove")
        task = await client.create_task(title="Task", tags=[tag.name])

        assert tag.name in task.tags

        await client.delete_tag(tag.name)

        # Task should no longer have the tag - verify via client API
        updated_task = await client.get_task(task.id)
        assert tag.name not in updated_task.tags

    async def test_delete_nonexistent_tag(self, client: TickTickClient):
        """Test deleting a tag that doesn't exist."""
        from ticktick_sdk.exceptions import TickTickNotFoundError

        with pytest.raises(TickTickNotFoundError):
            await client.delete_tag("nonexistent_tag")

    async def test_delete_tag_with_multiple_tasks(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test deleting a tag used by multiple tasks."""
        tag = await client.create_tag(name="SharedTag")

        task1 = await client.create_task(title="Task 1", tags=[tag.name])
        task2 = await client.create_task(title="Task 2", tags=[tag.name])
        task3 = await client.create_task(title="Task 3", tags=[tag.name])

        await client.delete_tag(tag.name)

        # All tasks should have tag removed - verify via client API
        updated_task1 = await client.get_task(task1.id)
        updated_task2 = await client.get_task(task2.id)
        updated_task3 = await client.get_task(task3.id)
        assert tag.name not in updated_task1.tags
        assert tag.name not in updated_task2.tags
        assert tag.name not in updated_task3.tags


# =============================================================================
# Tag Rename Tests
# =============================================================================


class TestTagRename:
    """Tests for tag rename functionality."""

    async def test_rename_tag(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test renaming a tag."""
        tag = await client.create_tag(name="OldName")

        await client.rename_tag("oldname", "NewName")

        # Verify via client API
        tags = await client.get_all_tags()
        tag_names = [t.name for t in tags]
        assert "oldname" not in tag_names
        assert "newname" in tag_names

    async def test_rename_tag_updates_tasks(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test that renaming a tag updates tasks."""
        tag = await client.create_tag(name="OldTag")
        task = await client.create_task(title="Task", tags=[tag.name])

        await client.rename_tag(tag.name, "NewTag")

        # Verify via client API
        updated_task = await client.get_task(task.id)
        assert "oldtag" not in [t.lower() for t in updated_task.tags]
        assert "newtag" in [t.lower() for t in updated_task.tags]

    async def test_rename_nonexistent_tag(self, client: TickTickClient):
        """Test renaming a tag that doesn't exist."""
        from ticktick_sdk.exceptions import TickTickNotFoundError

        with pytest.raises(TickTickNotFoundError):
            await client.rename_tag("nonexistent", "NewName")

    async def test_rename_tag_preserves_other_properties(
        self,
        client: TickTickClient,
        mock_api: MockUnifiedAPI,
    ):
        """Test that renaming preserves color and other properties."""
        tag = await client.create_tag(name="ColoredTag", color="#F18181")

        await client.rename_tag("coloredtag", "RenamedTag")

        # Verify via client API
        tags = await client.get_all_tags()
        renamed_tag = next(t for t in tags if t.name == "renamedtag")
        assert renamed_tag.color == "#F18181"


# =============================================================================
# Tag Merge Tests
# =============================================================================


class TestTagMerge:
    """Tests for tag merge functionality."""

    async def test_merge_tags(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test merging one tag into another."""
        source = await client.create_tag(name="SourceTag")
        target = await client.create_tag(name="TargetTag")

        await client.merge_tags(source.name, target.name)

        # Verify via client API
        tags = await client.get_all_tags()
        tag_names = [t.name for t in tags]
        # Source should be deleted
        assert source.name not in tag_names
        # Target should still exist
        assert target.name in tag_names

    async def test_merge_tags_moves_tasks(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test that merging moves tasks to target tag."""
        source = await client.create_tag(name="Source")
        target = await client.create_tag(name="Target")

        task1 = await client.create_task(title="Task 1", tags=[source.name])
        task2 = await client.create_task(title="Task 2", tags=[source.name])

        await client.merge_tags(source.name, target.name)

        # Tasks should now have target tag instead of source - verify via client API
        updated_task1 = await client.get_task(task1.id)
        updated_task2 = await client.get_task(task2.id)
        assert target.name in updated_task1.tags
        assert target.name in updated_task2.tags
        assert source.name not in updated_task1.tags
        assert source.name not in updated_task2.tags

    async def test_merge_nonexistent_source(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test merging nonexistent source tag."""
        from ticktick_sdk.exceptions import TickTickNotFoundError

        target = await client.create_tag(name="Target")

        with pytest.raises(TickTickNotFoundError):
            await client.merge_tags("nonexistent", target.name)

    async def test_merge_nonexistent_target(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test merging into nonexistent target tag."""
        from ticktick_sdk.exceptions import TickTickNotFoundError

        source = await client.create_tag(name="Source")

        with pytest.raises(TickTickNotFoundError):
            await client.merge_tags(source.name, "nonexistent")

    async def test_merge_tags_no_duplicates(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test that merging doesn't create duplicate tags on tasks."""
        source = await client.create_tag(name="Source")
        target = await client.create_tag(name="Target")

        # Task already has both tags
        task = await client.create_task(title="Task", tags=[source.name, target.name])

        await client.merge_tags(source.name, target.name)

        # Should only have target once - verify via client API
        updated_task = await client.get_task(task.id)
        target_count = sum(1 for t in updated_task.tags if t.lower() == target.name)
        assert target_count == 1


# =============================================================================
# Tag-Task Relationship Tests
# =============================================================================


class TestTagTaskRelationships:
    """Tests for tag-task relationships."""

    async def test_add_tag_to_task(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test adding a tag to a task."""
        tag = await client.create_tag(name="TestTag")
        task = await client.create_task(title="Task", tags=[tag.name])

        assert tag.name in task.tags

    async def test_multiple_tags_on_task(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test adding multiple tags to a task."""
        tag1 = await client.create_tag(name="Tag1")
        tag2 = await client.create_tag(name="Tag2")
        tag3 = await client.create_tag(name="Tag3")

        task = await client.create_task(title="Task", tags=[tag1.name, tag2.name, tag3.name])

        assert len(task.tags) == 3

    async def test_same_tag_on_multiple_tasks(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test using same tag on multiple tasks."""
        tag = await client.create_tag(name="SharedTag")

        task1 = await client.create_task(title="Task 1", tags=[tag.name])
        task2 = await client.create_task(title="Task 2", tags=[tag.name])
        task3 = await client.create_task(title="Task 3", tags=[tag.name])

        assert tag.name in task1.tags
        assert tag.name in task2.tags
        assert tag.name in task3.tags

    async def test_filter_tasks_by_tag(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test filtering tasks by tag."""
        work = await client.create_tag(name="Work")
        personal = await client.create_tag(name="Personal")

        task1 = await client.create_task(title="Work Task 1", tags=[work.name])
        task2 = await client.create_task(title="Work Task 2", tags=[work.name])
        await client.create_task(title="Personal Task", tags=[personal.name])
        await client.create_task(title="No Tag Task")

        work_tasks = await client.get_tasks_by_tag(work.name)

        # Verify our work tasks are in the results
        work_task_ids = [t.id for t in work_tasks]
        assert task1.id in work_task_ids
        assert task2.id in work_task_ids

    async def test_update_task_tags(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test updating task tags."""
        tag1 = await client.create_tag(name="Original")
        tag2 = await client.create_tag(name="Updated")

        task = await client.create_task(title="Task", tags=[tag1.name])
        task.tags = [tag2.name]
        updated = await client.update_task(task)

        assert tag1.name not in updated.tags
        assert tag2.name in updated.tags


# =============================================================================
# Tag Combination Tests
# =============================================================================


class TestTagCombinations:
    """Tests for combinations of tag operations."""

    async def test_tag_lifecycle(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test full tag lifecycle: create, use, rename, delete."""
        # Create
        tag = await client.create_tag(name="LifecycleTag", color="#4CAFF6")

        # Use on tasks
        task = await client.create_task(title="Task", tags=[tag.name])
        assert tag.name in task.tags

        # Rename
        await client.rename_tag(tag.name, "RenamedTag")

        # Verify task updated via client API
        updated_task = await client.get_task(task.id)
        assert "renamedtag" in [t.lower() for t in updated_task.tags]

        # Delete
        await client.delete_tag("renamedtag")

        # Verify removed from task via client API
        final_task = await client.get_task(task.id)
        assert len(final_task.tags) == 0

    async def test_complex_tag_hierarchy(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test complex tag hierarchy operations."""
        # Create hierarchy
        work = await client.create_tag(name="Work")
        projects = await client.create_tag(name="Projects", parent=work.name)
        client_a = await client.create_tag(name="ClientA", parent=projects.name)
        client_b = await client.create_tag(name="ClientB", parent=projects.name)

        # Create tasks with nested tags
        await client.create_task(title="Client A Task", tags=[client_a.name])
        await client.create_task(title="Client B Task", tags=[client_b.name])

        # Verify hierarchy via client API
        tags = await client.get_all_tags()
        tag_names = [t.name for t in tags]
        assert work.name in tag_names
        assert projects.name in tag_names
        assert client_a.name in tag_names
        assert client_b.name in tag_names

        client_a_tag = next(t for t in tags if t.name == client_a.name)
        assert client_a_tag.parent == projects.name

    async def test_merge_and_filter(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test merging tags and then filtering tasks."""
        urgent = await client.create_tag(name="Urgent")
        important = await client.create_tag(name="Important")

        task1 = await client.create_task(title="Task 1", tags=[urgent.name])
        task2 = await client.create_task(title="Task 2", tags=[urgent.name])
        task3 = await client.create_task(title="Task 3", tags=[important.name])

        # Before merge
        urgent_tasks = await client.get_tasks_by_tag(urgent.name)
        urgent_task_ids = [t.id for t in urgent_tasks]
        assert task1.id in urgent_task_ids
        assert task2.id in urgent_task_ids

        # Merge urgent into important
        await client.merge_tags(urgent.name, important.name)

        # After merge - all should be in important
        important_tasks = await client.get_tasks_by_tag(important.name)
        important_task_ids = [t.id for t in important_tasks]
        assert task1.id in important_task_ids
        assert task2.id in important_task_ids
        assert task3.id in important_task_ids

    async def test_bulk_tag_operations(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test bulk tag creation and usage."""
        # Create many tags
        tags = []
        for label in ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]:
            tag = await client.create_tag(name=f"Tag{label.upper()}")
            tags.append(tag)

        # Create tasks with various tag combinations
        task1 = await client.create_task(title="Task 1", tags=[tags[0].name, tags[1].name])
        await client.create_task(title="Task 2", tags=[tags[2].name, tags[3].name, tags[4].name])
        task3 = await client.create_task(title="Task 3", tags=[tags[0].name, tags[5].name, tags[9].name])

        # Verify via client API
        all_tags = await client.get_all_tags()
        all_tag_names = [t.name for t in all_tags]
        for tag in tags:
            assert tag.name in all_tag_names

        tag_a_tasks = await client.get_tasks_by_tag(tags[0].name)
        tag_a_task_ids = [t.id for t in tag_a_tasks]
        assert task1.id in tag_a_task_ids
        assert task3.id in tag_a_task_ids
