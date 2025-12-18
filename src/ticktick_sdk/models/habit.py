"""
Habit model for TickTick habits.

This module provides the Habit and HabitSection models for tracking
recurring habits with check-in functionality.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HabitSection(BaseModel):
    """
    Habit section (time of day grouping).

    TickTick organizes habits into sections like morning, afternoon, night.
    """

    id: str = Field(..., description="Section unique identifier")
    name: str = Field(..., description="Section name (_morning, _afternoon, _night)")
    sort_order: int = Field(default=0, description="Display order")
    created_time: datetime | None = Field(default=None, description="Creation timestamp")
    modified_time: datetime | None = Field(default=None, description="Last modification timestamp")
    etag: str | None = Field(default=None, description="Version tag for concurrency")

    @property
    def display_name(self) -> str:
        """Get human-readable section name."""
        name_map = {
            "_morning": "Morning",
            "_afternoon": "Afternoon",
            "_night": "Night",
        }
        return name_map.get(self.name, self.name.lstrip("_").title())

    @classmethod
    def from_v2(cls, data: dict[str, Any]) -> HabitSection:
        """Create a HabitSection from V2 API data."""
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            sort_order=data.get("sortOrder", 0),
            created_time=cls._parse_datetime(data.get("createdTime")),
            modified_time=cls._parse_datetime(data.get("modifiedTime")),
            etag=data.get("etag"),
        )

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime | None:
        """Parse ISO datetime string."""
        if not value:
            return None
        try:
            # Handle various TickTick datetime formats
            for fmt in [
                "%Y-%m-%dT%H:%M:%S.%f%z",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S.%f+0000",
                "%Y-%m-%dT%H:%M:%S+0000",
            ]:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
            return None
        except Exception:
            return None


class Habit(BaseModel):
    """
    Unified Habit model.

    Represents a recurring habit that can be checked in daily.
    Supports both boolean (yes/no) and numeric (count/measure) habits.
    """

    id: str = Field(..., description="Habit unique identifier")
    name: str = Field(..., description="Habit name/title")
    icon: str = Field(default="habit_daily_check_in", description="Icon resource name")
    color: str = Field(default="#97E38B", description="Hex color")
    sort_order: int = Field(default=0, description="Display order")
    status: int = Field(default=0, description="Status (0=active, 2=archived)")
    encouragement: str = Field(default="", description="Motivational message")
    total_checkins: int = Field(default=0, description="Total check-in count")
    created_time: datetime | None = Field(default=None, description="Creation timestamp")
    modified_time: datetime | None = Field(default=None, description="Last modification timestamp")
    archived_time: datetime | None = Field(default=None, description="Archival timestamp")

    # Habit type and goal
    habit_type: str = Field(default="Boolean", description="Type: Boolean or Real")
    goal: float = Field(default=1.0, description="Target goal value")
    step: float = Field(default=0.0, description="Increment step for numeric habits")
    unit: str = Field(default="Count", description="Unit of measurement")

    # Tracking
    etag: str | None = Field(default=None, description="Version tag for concurrency")
    repeat_rule: str | None = Field(default=None, description="RRULE recurrence pattern")
    reminders: list[str] = Field(default_factory=list, description="Reminder times (HH:MM)")
    record_enable: bool = Field(default=False, description="Enable value recording")

    # Section and targets
    section_id: str | None = Field(default=None, description="Time-of-day section ID")
    target_days: int = Field(default=0, description="Goal in days (0 = no target)")
    target_start_date: int | None = Field(default=None, description="Target start date (YYYYMMDD)")
    completed_cycles: int = Field(default=0, description="Completed cycles count")
    ex_dates: list[str] = Field(default_factory=list, description="Excluded dates")
    current_streak: int = Field(default=0, description="Current streak count")
    style: int = Field(default=1, description="Display style")

    @property
    def is_boolean(self) -> bool:
        """Check if this is a boolean (yes/no) habit."""
        return self.habit_type == "Boolean"

    @property
    def is_numeric(self) -> bool:
        """Check if this is a numeric (count/measure) habit."""
        return self.habit_type == "Real"

    @property
    def is_active(self) -> bool:
        """Check if habit is active (not archived)."""
        return self.status == 0

    @property
    def is_archived(self) -> bool:
        """Check if habit is archived."""
        return self.status == 2

    @classmethod
    def from_v2(cls, data: dict[str, Any]) -> Habit:
        """Create a Habit from V2 API data."""
        return cls(
            id=data.get("id") or "",
            name=data.get("name") or "",
            icon=data.get("iconRes") or "habit_daily_check_in",
            color=data.get("color") or "#97E38B",
            sort_order=data.get("sortOrder") or 0,
            status=data.get("status") or 0,
            encouragement=data.get("encouragement") or "",
            total_checkins=data.get("totalCheckIns") or 0,
            created_time=cls._parse_datetime(data.get("createdTime")),
            modified_time=cls._parse_datetime(data.get("modifiedTime")),
            archived_time=cls._parse_datetime(data.get("archivedTime")),
            habit_type=data.get("type") or "Boolean",
            goal=float(data.get("goal") or 1),
            step=float(data.get("step") or 0),
            unit=data.get("unit") or "Count",
            etag=data.get("etag"),
            repeat_rule=data.get("repeatRule"),
            reminders=data.get("reminders") or [],
            record_enable=data.get("recordEnable") or False,
            section_id=data.get("sectionId"),
            target_days=data.get("targetDays") or 0,
            target_start_date=data.get("targetStartDate"),
            completed_cycles=data.get("completedCycles") or 0,
            ex_dates=data.get("exDates") or [],
            current_streak=data.get("currentStreak") or 0,
            style=data.get("style") or 1,
        )

    def to_v2_dict(self, for_update: bool = False) -> dict[str, Any]:
        """
        Convert to V2 API dictionary format.

        Args:
            for_update: If True, include only fields needed for update

        Returns:
            Dictionary suitable for V2 API
        """
        data: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "iconRes": self.icon,
            "color": self.color,
            "sortOrder": self.sort_order,
            "status": self.status,
            "encouragement": self.encouragement,
            "totalCheckIns": self.total_checkins,
            "type": self.habit_type,
            "goal": self.goal,
            "step": self.step,
            "unit": self.unit,
            "reminders": self.reminders,
            "recordEnable": self.record_enable,
            "targetDays": self.target_days,
            "completedCycles": self.completed_cycles,
            "exDates": self.ex_dates,
            "currentStreak": self.current_streak,
            "style": self.style,
        }

        if self.etag:
            data["etag"] = self.etag
        if self.repeat_rule:
            data["repeatRule"] = self.repeat_rule
        if self.section_id:
            data["sectionId"] = self.section_id
        if self.target_start_date:
            data["targetStartDate"] = self.target_start_date

        # Add timestamps for creation
        if not for_update:
            now = datetime.now()
            data["createdTime"] = now.strftime("%Y-%m-%dT%H:%M:%S.000+0000")
            data["modifiedTime"] = now.strftime("%Y-%m-%dT%H:%M:%S.000+0000")
        else:
            data["modifiedTime"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.000+0000")

        return data

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime | None:
        """Parse ISO datetime string."""
        if not value:
            return None
        try:
            for fmt in [
                "%Y-%m-%dT%H:%M:%S.%f%z",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S.%f+0000",
                "%Y-%m-%dT%H:%M:%S+0000",
            ]:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
            return None
        except Exception:
            return None


class HabitCheckin(BaseModel):
    """
    Habit check-in record.

    Represents a single check-in for a habit on a specific date.
    """

    habit_id: str = Field(..., description="Habit ID")
    checkin_stamp: int = Field(..., description="Check-in date (YYYYMMDD)")
    checkin_time: datetime | None = Field(default=None, description="Check-in timestamp")
    value: float = Field(default=1.0, description="Check-in value")
    goal: float = Field(default=1.0, description="Goal at time of check-in")
    status: int = Field(default=2, description="Check-in status (2=completed)")

    @classmethod
    def from_v2(cls, data: dict[str, Any]) -> HabitCheckin:
        """Create a HabitCheckin from V2 API data."""
        return cls(
            habit_id=data.get("habitId", ""),
            checkin_stamp=data.get("checkinStamp", 0),
            checkin_time=Habit._parse_datetime(data.get("checkinTime")),
            value=float(data.get("value", 1)),
            goal=float(data.get("goal", 1)),
            status=data.get("status", 2),
        )


class HabitPreferences(BaseModel):
    """
    Habit preferences and settings.
    """

    show_in_calendar: bool = Field(default=True, description="Show habits in calendar")
    show_in_today: bool = Field(default=True, description="Show habits in today view")
    enabled: bool = Field(default=True, description="Habits feature enabled")
    default_section_order: int = Field(default=0, description="Default section order")

    @classmethod
    def from_v2(cls, data: dict[str, Any]) -> HabitPreferences:
        """Create HabitPreferences from V2 API data."""
        default_section = data.get("defaultSection", {})
        return cls(
            show_in_calendar=data.get("showInCalendar", True),
            show_in_today=data.get("showInToday", True),
            enabled=data.get("enabled", True),
            default_section_order=default_section.get("order", 0),
        )
