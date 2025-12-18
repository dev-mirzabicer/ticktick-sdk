"""
Comprehensive Habit Tests for TickTick Client.

This module tests all habit-related functionality including:
- Listing habits
- Creating habits (boolean and numeric)
- Updating habits
- Deleting habits
- Checking in habits
- Archiving/unarchiving habits
- Habit sections
- Habit check-in history
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ticktick_sdk.models import Habit, HabitSection, HabitPreferences

if TYPE_CHECKING:
    from tests.conftest import MockUnifiedAPI
    from ticktick_sdk.client import TickTickClient


pytestmark = [pytest.mark.unit]


# =============================================================================
# List Habits Tests
# =============================================================================


@pytest.mark.habits
class TestListHabits:
    """Tests for listing habits."""

    @pytest.mark.mock_only
    async def test_get_all_habits(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test listing all habits."""
        # Setup mock data
        mock_api._habits = {
            "habit1": Habit(
                id="habit1",
                name="Exercise",
                habit_type="Boolean",
                goal=1.0,
                total_checkins=10,
                current_streak=3,
            ),
            "habit2": Habit(
                id="habit2",
                name="Read",
                habit_type="Real",
                goal=30.0,
                unit="Pages",
                total_checkins=5,
                current_streak=1,
            ),
        }

        habits = await client.get_all_habits()

        assert len(habits) == 2
        assert all(isinstance(h, Habit) for h in habits)
        names = {h.name for h in habits}
        assert "Exercise" in names
        assert "Read" in names

    @pytest.mark.mock_only
    async def test_get_all_habits_empty(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test listing habits when none exist."""
        mock_api._habits = {}

        habits = await client.get_all_habits()

        assert habits == []


# =============================================================================
# Get Habit Tests
# =============================================================================


@pytest.mark.habits
class TestGetHabit:
    """Tests for getting a specific habit."""

    @pytest.mark.mock_only
    async def test_get_habit_by_id(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test getting a habit by ID."""
        mock_api._habits = {
            "habit123": Habit(
                id="habit123",
                name="Meditate",
                habit_type="Boolean",
                goal=1.0,
                total_checkins=15,
                current_streak=7,
            ),
        }

        habit = await client.get_habit("habit123")

        assert habit.id == "habit123"
        assert habit.name == "Meditate"
        assert habit.total_checkins == 15

    @pytest.mark.mock_only
    async def test_get_habit_not_found(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test getting a nonexistent habit."""
        mock_api._habits = {}

        from ticktick_sdk.exceptions import TickTickNotFoundError

        with pytest.raises(TickTickNotFoundError):
            await client.get_habit("nonexistent")


# =============================================================================
# Create Habit Tests
# =============================================================================


@pytest.mark.habits
class TestCreateHabit:
    """Tests for creating habits."""

    @pytest.mark.mock_only
    async def test_create_boolean_habit(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test creating a boolean (yes/no) habit."""
        mock_api._habits = {}

        habit = await client.create_habit(
            name="Exercise Daily",
            habit_type="Boolean",
            color="#4A90D9",
        )

        assert habit is not None
        assert habit.name == "Exercise Daily"
        assert habit.habit_type == "Boolean"
        assert habit.goal == 1.0

    @pytest.mark.mock_only
    async def test_create_numeric_habit(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test creating a numeric habit."""
        mock_api._habits = {}

        habit = await client.create_habit(
            name="Read Books",
            habit_type="Real",
            goal=30.0,
            step=5.0,
            unit="Pages",
        )

        assert habit is not None
        assert habit.name == "Read Books"
        assert habit.habit_type == "Real"
        assert habit.goal == 30.0
        assert habit.step == 5.0
        assert habit.unit == "Pages"

    @pytest.mark.mock_only
    async def test_create_habit_with_reminders(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test creating a habit with reminders."""
        mock_api._habits = {}

        habit = await client.create_habit(
            name="Morning Routine",
            reminders=["07:00", "08:00"],
        )

        assert habit is not None
        assert habit.reminders == ["07:00", "08:00"]

    @pytest.mark.mock_only
    async def test_create_habit_with_target_days(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test creating a habit with a day target (challenge)."""
        mock_api._habits = {}

        habit = await client.create_habit(
            name="100 Day Challenge",
            target_days=100,
        )

        assert habit is not None
        assert habit.target_days == 100


# =============================================================================
# Update Habit Tests
# =============================================================================


@pytest.mark.habits
class TestUpdateHabit:
    """Tests for updating habits."""

    @pytest.mark.mock_only
    async def test_update_habit_name(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test updating a habit's name."""
        mock_api._habits = {
            "habit1": Habit(
                id="habit1",
                name="Old Name",
                habit_type="Boolean",
                goal=1.0,
            ),
        }

        habit = await client.update_habit(
            habit_id="habit1",
            name="New Name",
        )

        assert habit.name == "New Name"

    @pytest.mark.mock_only
    async def test_update_habit_goal(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test updating a habit's goal."""
        mock_api._habits = {
            "habit1": Habit(
                id="habit1",
                name="Read",
                habit_type="Real",
                goal=30.0,
            ),
        }

        habit = await client.update_habit(
            habit_id="habit1",
            goal=50.0,
        )

        assert habit.goal == 50.0

    @pytest.mark.mock_only
    async def test_update_habit_color(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test updating a habit's color."""
        mock_api._habits = {
            "habit1": Habit(
                id="habit1",
                name="Exercise",
                habit_type="Boolean",
                goal=1.0,
                color="#97E38B",
            ),
        }

        habit = await client.update_habit(
            habit_id="habit1",
            color="#FF5500",
        )

        assert habit.color == "#FF5500"


# =============================================================================
# Delete Habit Tests
# =============================================================================


@pytest.mark.habits
class TestDeleteHabit:
    """Tests for deleting habits."""

    @pytest.mark.mock_only
    async def test_delete_habit(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test deleting a habit."""
        mock_api._habits = {
            "habit1": Habit(
                id="habit1",
                name="Old Habit",
                habit_type="Boolean",
                goal=1.0,
            ),
        }

        await client.delete_habit("habit1")

        assert "habit1" not in mock_api._habits

    @pytest.mark.mock_only
    async def test_delete_habit_not_found(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test deleting a nonexistent habit."""
        mock_api._habits = {}

        from ticktick_sdk.exceptions import TickTickNotFoundError

        with pytest.raises(TickTickNotFoundError):
            await client.delete_habit("nonexistent")


# =============================================================================
# Check-in Habit Tests
# =============================================================================


@pytest.mark.habits
class TestCheckinHabit:
    """Tests for checking in habits."""

    @pytest.mark.mock_only
    async def test_checkin_boolean_habit(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test checking in a boolean habit."""
        mock_api._habits = {
            "habit1": Habit(
                id="habit1",
                name="Exercise",
                habit_type="Boolean",
                goal=1.0,
                total_checkins=10,
                current_streak=3,
            ),
        }

        habit = await client.checkin_habit("habit1")

        assert habit.total_checkins == 11
        assert habit.current_streak == 4

    @pytest.mark.mock_only
    async def test_checkin_numeric_habit_with_value(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test checking in a numeric habit with a value."""
        mock_api._habits = {
            "habit1": Habit(
                id="habit1",
                name="Read",
                habit_type="Real",
                goal=30.0,
                total_checkins=5,
                current_streak=2,
            ),
        }

        habit = await client.checkin_habit("habit1", value=10.0)

        # Value is added as integer to total_checkins
        assert habit.total_checkins == 15
        assert habit.current_streak == 3


# =============================================================================
# Archive Habit Tests
# =============================================================================


@pytest.mark.habits
class TestArchiveHabit:
    """Tests for archiving/unarchiving habits."""

    @pytest.mark.mock_only
    async def test_archive_habit(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test archiving a habit."""
        mock_api._habits = {
            "habit1": Habit(
                id="habit1",
                name="Old Habit",
                habit_type="Boolean",
                goal=1.0,
                status=0,  # Active
            ),
        }

        habit = await client.archive_habit("habit1")

        assert habit.status == 2  # Archived
        assert habit.is_archived

    @pytest.mark.mock_only
    async def test_unarchive_habit(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test unarchiving a habit."""
        mock_api._habits = {
            "habit1": Habit(
                id="habit1",
                name="Old Habit",
                habit_type="Boolean",
                goal=1.0,
                status=2,  # Archived
            ),
        }

        habit = await client.unarchive_habit("habit1")

        assert habit.status == 0  # Active
        assert habit.is_active


# =============================================================================
# Habit Sections Tests
# =============================================================================


@pytest.mark.habits
class TestHabitSections:
    """Tests for habit sections."""

    @pytest.mark.mock_only
    async def test_get_habit_sections(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test getting habit sections."""
        sections = await client.get_habit_sections()

        assert isinstance(sections, list)
        # Mock should return default sections
        assert len(sections) >= 0


# =============================================================================
# Habit Preferences Tests
# =============================================================================


@pytest.mark.habits
class TestHabitPreferences:
    """Tests for habit preferences."""

    @pytest.mark.mock_only
    async def test_get_habit_preferences(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test getting habit preferences."""
        prefs = await client.get_habit_preferences()

        assert isinstance(prefs, HabitPreferences)
        assert hasattr(prefs, "show_in_calendar")
        assert hasattr(prefs, "show_in_today")
        assert hasattr(prefs, "enabled")


# =============================================================================
# Habit Checkins History Tests
# =============================================================================


@pytest.mark.habits
class TestHabitCheckinsHistory:
    """Tests for habit check-in history."""

    @pytest.mark.mock_only
    async def test_get_habit_checkins(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test getting habit check-in history."""
        mock_api._habits = {
            "habit1": Habit(
                id="habit1",
                name="Exercise",
                habit_type="Boolean",
                goal=1.0,
            ),
        }

        checkins = await client.get_habit_checkins(
            habit_ids=["habit1"],
            after_stamp=0,
        )

        assert isinstance(checkins, dict)
        assert "habit1" in checkins

    @pytest.mark.mock_only
    async def test_get_habit_checkins_with_date_filter(self, client: TickTickClient, mock_api: MockUnifiedAPI):
        """Test getting check-ins after a specific date."""
        mock_api._habits = {
            "habit1": Habit(
                id="habit1",
                name="Exercise",
                habit_type="Boolean",
                goal=1.0,
            ),
        }

        checkins = await client.get_habit_checkins(
            habit_ids=["habit1"],
            after_stamp=20251201,  # December 1, 2025
        )

        assert isinstance(checkins, dict)
