"""
TickTick Unified Data Models.

This package provides canonical Pydantic models that unify V1 and V2 API
data structures. These models are the single source of truth for the
TickTick SDK.

Models:
    - Task: Unified task model
    - Project: Unified project model
    - ProjectGroup: Project folder/group
    - Tag: Tag model
    - ChecklistItem: Subtask/checklist item
    - Column: Kanban column
    - User: User profile and status
    - FocusStats: Focus/Pomodoro statistics
    - Habit: Recurring habit model
    - HabitSection: Time-of-day habit grouping
    - HabitCheckin: Habit check-in record
    - HabitPreferences: Habit settings
"""

from ticktick_sdk.models.task import Task, ChecklistItem, TaskReminder
from ticktick_sdk.models.project import Project, ProjectGroup, Column, ProjectData
from ticktick_sdk.models.tag import Tag
from ticktick_sdk.models.user import User, UserStatus, UserStatistics
from ticktick_sdk.models.habit import Habit, HabitSection, HabitCheckin, HabitPreferences

__all__ = [
    "Task",
    "ChecklistItem",
    "TaskReminder",
    "Project",
    "ProjectGroup",
    "Column",
    "ProjectData",
    "Tag",
    "User",
    "UserStatus",
    "UserStatistics",
    "Habit",
    "HabitSection",
    "HabitCheckin",
    "HabitPreferences",
]
