"""
TickTick V2 API Client.

This module implements the client for TickTick's unofficial V2 API.
It provides methods for all reverse-engineered V2 endpoints.

Endpoints:
    Authentication:
        - POST /user/signon
        - POST /user/signon/totp (2FA)

    User:
        - GET /user/status
        - GET /user/profile
        - GET /user/preferences/settings
        - GET /statistics/general

    Sync:
        - GET /batch/check/0

    Tasks:
        - POST /batch/task (create/update/delete)
        - GET /task/{id}
        - POST /batch/taskProject (move)
        - POST /batch/taskParent (subtasks)
        - GET /project/all/closed
        - GET /project/all/completed
        - GET /project/all/trash/pagination

    Projects:
        - POST /batch/project

    Project Groups:
        - POST /batch/projectGroup

    Tags:
        - POST /batch/tag
        - PUT /tag/rename
        - DELETE /tag
        - PUT /tag/merge

    Focus/Pomodoro:
        - GET /pomodoros/statistics/heatmap/{from}/{to}
        - GET /pomodoros/statistics/dist/{from}/{to}

    Habits:
        - POST /habitCheckins/query
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any
from urllib.parse import quote

import httpx

from ticktick_sdk.api.base import BaseTickTickClient
from ticktick_sdk.api.v2.auth import SessionHandler, SessionToken
from ticktick_sdk.api.v2.types import (
    BatchHabitRequestV2,
    BatchResponseV2,
    BatchTaskParentResponseV2,
    BatchTaskRequestV2,
    BatchProjectRequestV2,
    BatchProjectGroupRequestV2,
    BatchTagRequestV2,
    FocusDistributionV2,
    FocusHeatmapV2,
    HabitCreateV2,
    HabitPreferencesV2,
    HabitSectionV2,
    HabitUpdateV2,
    HabitV2,
    ProjectCreateV2,
    ProjectGroupCreateV2,
    ProjectGroupUpdateV2,
    ProjectGroupV2,
    ProjectUpdateV2,
    ProjectV2,
    SyncStateV2,
    TagCreateV2,
    TagUpdateV2,
    TagV2,
    TaskCreateV2,
    TaskDeleteV2,
    TaskMoveV2,
    TaskParentV2,
    TaskUpdateV2,
    TaskV2,
    TrashResponseV2,
    UserPreferencesV2,
    UserProfileV2,
    UserStatisticsV2,
    UserStatusV2,
)
from ticktick_sdk.constants import (
    APIVersion,
    DEFAULT_TIMEOUT,
    TICKTICK_API_BASE_V2,
)
from ticktick_sdk.exceptions import TickTickAuthenticationError

logger = logging.getLogger(__name__)


class TickTickV2Client(BaseTickTickClient):
    """
    Client for TickTick V2 API.

    This client handles session-based authentication and provides methods
    for all V2 API endpoints.

    Usage:
        client = TickTickV2Client()

        # Authenticate
        await client.authenticate("user@example.com", "password")

        # Use the client
        async with client:
            state = await client.sync()
            tasks = state["syncTaskBean"]["update"]
    """

    def __init__(
        self,
        device_id: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(timeout=timeout)

        self._session_handler = SessionHandler(
            device_id=device_id,
            timeout=timeout,
        )

    # =========================================================================
    # Abstract Property Implementations
    # =========================================================================

    @property
    def api_version(self) -> APIVersion:
        """Return the API version."""
        return APIVersion.V2

    @property
    def base_url(self) -> str:
        """Return the base URL for V2 API."""
        return TICKTICK_API_BASE_V2

    @property
    def is_authenticated(self) -> bool:
        """Check if authenticated with a valid session."""
        return self._session_handler.is_authenticated

    def _get_x_device_header(self) -> str:
        """Get the x-device header JSON string.

        Uses minimal format (based on pyticktick):
        Only 3 fields: platform, version, id
        """
        return json.dumps({
            "platform": "web",
            "version": 6430,
            "id": self._session_handler.device_id,
        })

    # Simple User-Agent that works (based on pyticktick)
    V2_USER_AGENT = "Mozilla/5.0 (rv:145.0) Firefox/145.0"

    def _get_auth_headers(self) -> dict[str, str]:
        """Get V2 authentication headers.

        Uses minimal headers based on pyticktick's working approach.
        """
        headers: dict[str, str] = {}

        if self._session_handler.session is not None:
            session = self._session_handler.session

            # Override with simple User-Agent for V2 API
            headers["User-Agent"] = self.V2_USER_AGENT
            headers["X-Device"] = self._get_x_device_header()

            # Cookie is the primary auth mechanism for V2
            # The 't' cookie contains the session token
            if session.cookies:
                cookie_str = "; ".join(
                    f"{k}={v}" for k, v in session.cookies.items()
                )
                headers["Cookie"] = cookie_str

        return headers

    # =========================================================================
    # Authentication Methods
    # =========================================================================

    async def authenticate(
        self,
        username: str,
        password: str,
    ) -> SessionToken:
        """
        Authenticate with username and password.

        Args:
            username: TickTick account username/email
            password: TickTick account password

        Returns:
            SessionToken with authentication credentials
        """
        return await self._session_handler.authenticate(username, password)

    async def authenticate_2fa(
        self,
        auth_id: str,
        totp_code: str,
    ) -> SessionToken:
        """
        Complete 2FA authentication.

        Args:
            auth_id: Auth ID from initial sign-on
            totp_code: TOTP code from authenticator

        Returns:
            SessionToken with authentication credentials
        """
        return await self._session_handler.authenticate_2fa(auth_id, totp_code)

    def set_session(self, session: SessionToken) -> None:
        """Set an existing session directly."""
        self._session_handler.set_session(session)

    @property
    def session(self) -> SessionToken | None:
        """Get the current session."""
        return self._session_handler.session

    @property
    def inbox_id(self) -> str | None:
        """Get the inbox ID."""
        return self._session_handler.inbox_id

    # =========================================================================
    # Sync Endpoint
    # =========================================================================

    async def sync(self) -> SyncStateV2:
        """
        Get the complete account state.

        This returns all projects, tasks, tags, and settings.

        Returns:
            Complete sync state
        """
        response = await self._get_json("/batch/check/0")
        return response

    # =========================================================================
    # User Endpoints
    # =========================================================================

    async def get_user_status(self) -> UserStatusV2:
        """Get user subscription status."""
        response = await self._get_json("/user/status")
        return response

    async def get_user_profile(self) -> UserProfileV2:
        """Get user profile information."""
        response = await self._get_json("/user/profile")
        return response

    async def get_user_preferences(
        self,
        include_web: bool = True,
    ) -> UserPreferencesV2:
        """Get user preferences/settings."""
        params = {"includeWeb": str(include_web).lower()}
        response = await self._get_json(
            "/user/preferences/settings",
            params=params,
        )
        return response

    async def get_user_statistics(self) -> UserStatisticsV2:
        """Get user productivity statistics."""
        response = await self._get_json("/statistics/general")
        return response

    # =========================================================================
    # Task Endpoints
    # =========================================================================

    async def get_task(self, task_id: str) -> TaskV2:
        """
        Get a single task by ID.

        Args:
            task_id: Task identifier

        Returns:
            Task data
        """
        endpoint = f"/task/{task_id}"
        response = await self._get_json(endpoint)
        return response

    async def batch_tasks(
        self,
        add: list[TaskCreateV2] | None = None,
        update: list[TaskUpdateV2] | None = None,
        delete: list[TaskDeleteV2] | None = None,
    ) -> BatchResponseV2:
        """
        Batch create, update, and delete tasks.

        Args:
            add: Tasks to create
            update: Tasks to update
            delete: Tasks to delete

        Returns:
            Batch response with etags and errors
        """
        data: BatchTaskRequestV2 = {
            "add": add or [],
            "update": update or [],
            "delete": delete or [],
            "addAttachments": [],
            "updateAttachments": [],
            "deleteAttachments": [],
        }
        response = await self._post_json("/batch/task", json_data=data)
        return response

    async def create_task(
        self,
        title: str,
        project_id: str,
        *,
        content: str | None = None,
        desc: str | None = None,
        kind: str | None = None,
        priority: int | None = None,
        start_date: str | None = None,
        due_date: str | None = None,
        time_zone: str | None = None,
        is_all_day: bool | None = None,
        reminders: list[dict[str, Any]] | None = None,
        repeat_flag: str | None = None,
        tags: list[str] | None = None,
        items: list[dict[str, Any]] | None = None,
        sort_order: int | None = None,
        parent_id: str | None = None,
    ) -> BatchResponseV2:
        """
        Create a single task.

        Args:
            title: Task title
            project_id: Project ID
            content: Task content
            desc: Checklist description
            kind: Task kind (TEXT, NOTE, CHECKLIST)
            priority: Priority (0, 1, 3, 5)
            start_date: Start date
            due_date: Due date
            time_zone: Timezone
            is_all_day: All-day flag
            reminders: Reminder list
            repeat_flag: Recurrence rule
            tags: Tag list
            items: Subtask items
            sort_order: Sort order
            parent_id: Parent task ID (for subtasks)

        Returns:
            Batch response
        """
        task: TaskCreateV2 = {
            "title": title,
            "projectId": project_id,
        }

        if content is not None:
            task["content"] = content
        if desc is not None:
            task["desc"] = desc
        if kind is not None:
            task["kind"] = kind
        if priority is not None:
            task["priority"] = priority
        if start_date is not None:
            task["startDate"] = start_date
        if due_date is not None:
            task["dueDate"] = due_date
        if time_zone is not None:
            task["timeZone"] = time_zone
        if is_all_day is not None:
            task["isAllDay"] = is_all_day
        if reminders is not None:
            task["reminders"] = reminders  # type: ignore
        if repeat_flag is not None:
            task["repeatFlag"] = repeat_flag
        if tags is not None:
            task["tags"] = tags
        if items is not None:
            task["items"] = items  # type: ignore
        if sort_order is not None:
            task["sortOrder"] = sort_order
        if parent_id is not None:
            task["parentId"] = parent_id

        return await self.batch_tasks(add=[task])

    async def update_task(
        self,
        task_id: str,
        project_id: str,
        *,
        title: str | None = None,
        content: str | None = None,
        desc: str | None = None,
        kind: str | None = None,
        status: int | None = None,
        priority: int | None = None,
        start_date: str | None = None,
        due_date: str | None = None,
        time_zone: str | None = None,
        is_all_day: bool | None = None,
        reminders: list[dict[str, Any]] | None = None,
        repeat_flag: str | None = None,
        tags: list[str] | None = None,
        items: list[dict[str, Any]] | None = None,
        sort_order: int | None = None,
        completed_time: str | None = None,
    ) -> BatchResponseV2:
        """
        Update a single task.

        Args:
            task_id: Task ID
            project_id: Project ID
            title: New title
            content: New content
            desc: New description
            kind: New kind
            status: New status
            priority: New priority
            start_date: New start date
            due_date: New due date
            time_zone: New timezone
            is_all_day: All-day flag
            reminders: New reminders
            repeat_flag: New recurrence
            tags: New tags
            items: New subtasks
            sort_order: New sort order
            completed_time: Completion time

        Returns:
            Batch response
        """
        task: TaskUpdateV2 = {
            "id": task_id,
            "projectId": project_id,
        }

        if title is not None:
            task["title"] = title
        if content is not None:
            task["content"] = content
        if desc is not None:
            task["desc"] = desc
        if kind is not None:
            task["kind"] = kind
        if status is not None:
            task["status"] = status
        if priority is not None:
            task["priority"] = priority
        if start_date is not None:
            task["startDate"] = start_date
        if due_date is not None:
            task["dueDate"] = due_date
        if time_zone is not None:
            task["timeZone"] = time_zone
        if is_all_day is not None:
            task["isAllDay"] = is_all_day
        if reminders is not None:
            task["reminders"] = reminders  # type: ignore
        if repeat_flag is not None:
            task["repeatFlag"] = repeat_flag
        if tags is not None:
            task["tags"] = tags
        if items is not None:
            task["items"] = items  # type: ignore
        if sort_order is not None:
            task["sortOrder"] = sort_order
        if completed_time is not None:
            task["completedTime"] = completed_time

        return await self.batch_tasks(update=[task])

    async def delete_task(self, project_id: str, task_id: str) -> BatchResponseV2:
        """
        Delete a task.

        Args:
            project_id: Project ID
            task_id: Task ID

        Returns:
            Batch response
        """
        delete_item: TaskDeleteV2 = {
            "projectId": project_id,
            "taskId": task_id,
        }
        return await self.batch_tasks(delete=[delete_item])

    async def move_tasks(
        self,
        moves: list[TaskMoveV2],
    ) -> Any:
        """
        Move tasks between projects.

        Args:
            moves: List of move operations

        Returns:
            Response data
        """
        response = await self._post_json("/batch/taskProject", json_data=moves)
        return response

    async def move_task(
        self,
        task_id: str,
        from_project_id: str,
        to_project_id: str,
    ) -> Any:
        """
        Move a single task to a different project.

        Args:
            task_id: Task ID
            from_project_id: Source project ID
            to_project_id: Destination project ID

        Returns:
            Response data
        """
        move: TaskMoveV2 = {
            "taskId": task_id,
            "fromProjectId": from_project_id,
            "toProjectId": to_project_id,
        }
        return await self.move_tasks([move])

    async def set_task_parent(
        self,
        task_id: str,
        project_id: str,
        parent_id: str,
    ) -> BatchTaskParentResponseV2:
        """
        Make a task a subtask of another task.

        Args:
            task_id: Task to make a subtask
            project_id: Project ID
            parent_id: Parent task ID

        Returns:
            Batch parent response
        """
        data: list[TaskParentV2] = [{
            "taskId": task_id,
            "projectId": project_id,
            "parentId": parent_id,
        }]
        response = await self._post_json("/batch/taskParent", json_data=data)
        return response

    async def unset_task_parent(
        self,
        task_id: str,
        project_id: str,
        old_parent_id: str,
    ) -> BatchTaskParentResponseV2:
        """
        Remove a task from being a subtask.

        Args:
            task_id: Task to remove from parent
            project_id: Project ID
            old_parent_id: Current parent ID

        Returns:
            Batch parent response
        """
        data: list[TaskParentV2] = [{
            "taskId": task_id,
            "projectId": project_id,
            "oldParentId": old_parent_id,
        }]
        response = await self._post_json("/batch/taskParent", json_data=data)
        return response

    async def get_completed_tasks(
        self,
        from_date: datetime,
        to_date: datetime,
        limit: int = 100,
    ) -> list[TaskV2]:
        """
        Get completed tasks in a date range.

        Args:
            from_date: Start date
            to_date: End date
            limit: Maximum results

        Returns:
            List of completed tasks
        """
        # Format: YYYY-MM-DD HH:MM:SS (httpx handles URL encoding)
        from_str = from_date.strftime("%Y-%m-%d %H:%M:%S")
        to_str = to_date.strftime("%Y-%m-%d %H:%M:%S")

        params = {
            "from": from_str,
            "to": to_str,
            "status": "Completed",
            "limit": str(limit),
        }
        response = await self._get_json("/project/all/closed", params=params)
        return response

    async def get_abandoned_tasks(
        self,
        from_date: datetime,
        to_date: datetime,
        limit: int = 100,
    ) -> list[TaskV2]:
        """
        Get abandoned (won't do) tasks in a date range.

        Args:
            from_date: Start date
            to_date: End date
            limit: Maximum results

        Returns:
            List of abandoned tasks
        """
        # Format: YYYY-MM-DD HH:MM:SS (httpx handles URL encoding)
        from_str = from_date.strftime("%Y-%m-%d %H:%M:%S")
        to_str = to_date.strftime("%Y-%m-%d %H:%M:%S")

        params = {
            "from": from_str,
            "to": to_str,
            "status": "Abandoned",
            "limit": str(limit),
        }
        response = await self._get_json("/project/all/closed", params=params)
        return response

    async def get_deleted_tasks(
        self,
        start: int = 0,
        limit: int = 500,
    ) -> TrashResponseV2:
        """
        Get deleted tasks (trash).

        Args:
            start: Start offset
            limit: Maximum results

        Returns:
            Trash response with tasks
        """
        params = {
            "start": str(start),
            "limit": str(limit),
        }
        response = await self._get_json(
            "/project/all/trash/pagination",
            params=params,
        )
        return response

    # =========================================================================
    # Project Endpoints
    # =========================================================================

    async def batch_projects(
        self,
        add: list[ProjectCreateV2] | None = None,
        update: list[ProjectUpdateV2] | None = None,
        delete: list[str] | None = None,
    ) -> BatchResponseV2:
        """
        Batch create, update, and delete projects.

        Args:
            add: Projects to create
            update: Projects to update
            delete: Project IDs to delete

        Returns:
            Batch response
        """
        data: BatchProjectRequestV2 = {
            "add": add or [],
            "update": update or [],
            "delete": delete or [],
        }
        response = await self._post_json("/batch/project", json_data=data)
        return response

    async def create_project(
        self,
        name: str,
        *,
        color: str | None = None,
        kind: str | None = None,
        view_mode: str | None = None,
        group_id: str | None = None,
        sort_order: int | None = None,
    ) -> BatchResponseV2:
        """
        Create a single project.

        Args:
            name: Project name
            color: Hex color
            kind: Project kind (TASK, NOTE)
            view_mode: View mode (list, kanban, timeline)
            group_id: Parent folder ID
            sort_order: Sort order

        Returns:
            Batch response
        """
        project: ProjectCreateV2 = {"name": name}

        if color is not None:
            project["color"] = color
        if kind is not None:
            project["kind"] = kind
        if view_mode is not None:
            project["viewMode"] = view_mode
        if group_id is not None:
            project["groupId"] = group_id
        if sort_order is not None:
            project["sortOrder"] = sort_order

        return await self.batch_projects(add=[project])

    async def update_project(
        self,
        project_id: str,
        name: str,
        *,
        color: str | None = None,
        group_id: str | None = None,
    ) -> BatchResponseV2:
        """
        Update a project.

        Args:
            project_id: Project ID
            name: New name
            color: New color
            group_id: New folder ID (use "NONE" to ungroup)

        Returns:
            Batch response
        """
        project: ProjectUpdateV2 = {
            "id": project_id,
            "name": name,
        }

        if color is not None:
            project["color"] = color
        if group_id is not None:
            project["groupId"] = group_id

        return await self.batch_projects(update=[project])

    async def delete_project(self, project_id: str) -> BatchResponseV2:
        """
        Delete a project.

        Args:
            project_id: Project ID

        Returns:
            Batch response
        """
        return await self.batch_projects(delete=[project_id])

    # =========================================================================
    # Project Group Endpoints
    # =========================================================================

    async def batch_project_groups(
        self,
        add: list[ProjectGroupCreateV2] | None = None,
        update: list[ProjectGroupUpdateV2] | None = None,
        delete: list[str] | None = None,
    ) -> BatchResponseV2:
        """
        Batch create, update, and delete project groups.

        Args:
            add: Groups to create
            update: Groups to update
            delete: Group IDs to delete

        Returns:
            Batch response
        """
        data: BatchProjectGroupRequestV2 = {
            "add": add or [],
            "update": update or [],
            "delete": delete or [],
        }
        response = await self._post_json("/batch/projectGroup", json_data=data)
        return response

    async def create_project_group(self, name: str) -> BatchResponseV2:
        """
        Create a project group/folder.

        Args:
            name: Group name

        Returns:
            Batch response
        """
        group: ProjectGroupCreateV2 = {
            "name": name,
            "listType": "group",
        }
        return await self.batch_project_groups(add=[group])

    async def update_project_group(
        self,
        group_id: str,
        name: str,
    ) -> BatchResponseV2:
        """
        Update a project group.

        Args:
            group_id: Group ID
            name: New name

        Returns:
            Batch response
        """
        group: ProjectGroupUpdateV2 = {
            "id": group_id,
            "name": name,
            "listType": "group",
        }
        return await self.batch_project_groups(update=[group])

    async def delete_project_group(self, group_id: str) -> BatchResponseV2:
        """
        Delete a project group.

        Args:
            group_id: Group ID

        Returns:
            Batch response
        """
        return await self.batch_project_groups(delete=[group_id])

    # =========================================================================
    # Tag Endpoints
    # =========================================================================

    async def batch_tags(
        self,
        add: list[TagCreateV2] | None = None,
        update: list[TagUpdateV2] | None = None,
    ) -> BatchResponseV2:
        """
        Batch create and update tags.

        Note: Use rename_tag and delete_tag for those operations.

        Args:
            add: Tags to create
            update: Tags to update

        Returns:
            Batch response
        """
        data: BatchTagRequestV2 = {
            "add": add or [],
            "update": update or [],
        }
        response = await self._post_json("/batch/tag", json_data=data)
        return response

    async def create_tag(
        self,
        label: str,
        *,
        color: str | None = None,
        parent: str | None = None,
        sort_type: str | None = None,
        sort_order: int | None = None,
    ) -> BatchResponseV2:
        """
        Create a tag.

        Args:
            label: Tag display name
            color: Hex color
            parent: Parent tag name (for nesting)
            sort_type: Sort type
            sort_order: Sort order

        Returns:
            Batch response
        """
        tag: TagCreateV2 = {"label": label}

        # Auto-generate name from label
        tag["name"] = label.lower().replace(" ", "")

        if color is not None:
            tag["color"] = color
        if parent is not None:
            tag["parent"] = parent
        if sort_type is not None:
            tag["sortType"] = sort_type
        if sort_order is not None:
            tag["sortOrder"] = sort_order

        return await self.batch_tags(add=[tag])

    async def update_tag(
        self,
        name: str,
        label: str,
        *,
        color: str | None = None,
        parent: str | None = None,
        sort_type: str | None = None,
        sort_order: int | None = None,
    ) -> BatchResponseV2:
        """
        Update a tag.

        Args:
            name: Tag identifier (lowercase)
            label: Tag display name
            color: New color
            parent: New parent
            sort_type: New sort type
            sort_order: New sort order

        Returns:
            Batch response
        """
        tag: TagUpdateV2 = {
            "name": name,
            "label": label,
            "rawName": name,
        }

        if color is not None:
            tag["color"] = color
        if parent is not None:
            tag["parent"] = parent
        if sort_type is not None:
            tag["sortType"] = sort_type
        if sort_order is not None:
            tag["sortOrder"] = sort_order

        return await self.batch_tags(update=[tag])

    async def rename_tag(self, old_name: str, new_label: str) -> Any:
        """
        Rename a tag.

        Args:
            old_name: Current tag name
            new_label: New tag label

        Returns:
            Response data
        """
        data = {
            "name": old_name,
            "newName": new_label,
        }
        response = await self._put("/tag/rename", json_data=data)
        return response.json() if response.content else None

    async def delete_tag(self, name: str) -> None:
        """
        Delete a tag.

        Args:
            name: Tag name to delete
        """
        params = {"name": name}
        await self._delete("/tag", params=params)

    async def merge_tags(self, source_name: str, target_name: str) -> Any:
        """
        Merge one tag into another.

        Args:
            source_name: Tag to merge from (will be deleted)
            target_name: Tag to merge into

        Returns:
            Response data
        """
        data = {
            "name": source_name,
            "newName": target_name,
        }
        response = await self._put("/tag/merge", json_data=data)
        return response.json() if response.content else None

    # =========================================================================
    # Focus/Pomodoro Endpoints
    # =========================================================================

    async def get_focus_heatmap(
        self,
        start_date: date,
        end_date: date,
    ) -> list[FocusHeatmapV2]:
        """
        Get focus/pomodoro heatmap statistics.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            List of heatmap data points
        """
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        endpoint = f"/pomodoros/statistics/heatmap/{start_str}/{end_str}"
        response = await self._get_json(endpoint)
        return response

    async def get_focus_by_tag(
        self,
        start_date: date,
        end_date: date,
    ) -> FocusDistributionV2:
        """
        Get focus time distribution by tag.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Focus distribution data
        """
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        endpoint = f"/pomodoros/statistics/dist/{start_str}/{end_str}"
        response = await self._get_json(endpoint)
        return response

    # =========================================================================
    # Habit Endpoints
    # =========================================================================

    async def get_habits(self) -> list[HabitV2]:
        """
        Get all habits.

        Returns:
            List of habits
        """
        response = await self._get_json("/habits")
        return response

    async def get_habit_sections(self) -> list[HabitSectionV2]:
        """
        Get habit sections (time-of-day groupings).

        Sections are: _morning, _afternoon, _night

        Returns:
            List of habit sections
        """
        response = await self._get_json("/habitSections")
        return response

    async def get_habit_preferences(self) -> HabitPreferencesV2:
        """
        Get habit preferences/settings.

        Returns:
            Habit preferences (showInCalendar, showInToday, enabled, etc.)
        """
        params = {"platform": "web"}
        response = await self._get_json("/user/preferences/habit", params=params)
        return response

    async def batch_habits(
        self,
        add: list[HabitCreateV2] | None = None,
        update: list[HabitUpdateV2] | None = None,
        delete: list[str] | None = None,
    ) -> BatchResponseV2:
        """
        Batch create, update, and delete habits.

        Args:
            add: Habits to create
            update: Habits to update
            delete: Habit IDs to delete

        Returns:
            Batch response with etags and errors
        """
        data: BatchHabitRequestV2 = {
            "add": add or [],
            "update": update or [],
            "delete": delete or [],
        }
        response = await self._post_json("/habits/batch", json_data=data)
        return response

    async def create_habit(
        self,
        habit_id: str,
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
        target_start_date: int | None = None,
        encouragement: str = "",
        record_enable: bool = False,
        sort_order: int | None = None,
    ) -> BatchResponseV2:
        """
        Create a habit.

        Args:
            habit_id: Client-generated habit ID (24-char hex)
            name: Habit name
            habit_type: "Boolean" for yes/no, "Real" for numeric
            goal: Target goal value
            step: Increment step for numeric habits
            unit: Unit of measurement
            icon: Icon resource name
            color: Hex color
            section_id: Time-of-day section ID
            repeat_rule: RRULE recurrence pattern
            reminders: List of reminder times ("HH:MM")
            target_days: Goal in days (0 = no target)
            target_start_date: Target start date (YYYYMMDD)
            encouragement: Motivational message
            record_enable: Enable value recording
            sort_order: Display order

        Returns:
            Batch response
        """
        from datetime import datetime

        now = datetime.now()
        now_str = now.strftime("%Y-%m-%dT%H:%M:%S.000+0000")

        habit: HabitCreateV2 = {
            "id": habit_id,
            "name": name,
            "type": habit_type,
            "goal": goal,
            "step": step,
            "unit": unit,
            "iconRes": icon,
            "color": color,
            "status": 0,
            "totalCheckIns": 0,
            "currentStreak": 0,
            "completedCycles": 0,
            "createdTime": now_str,
            "modifiedTime": now_str,
            "encouragement": encouragement,
            "recordEnable": record_enable,
            "exDates": [],
            "style": 1,
            "etag": "",
        }

        if section_id:
            habit["sectionId"] = section_id
        if repeat_rule:
            habit["repeatRule"] = repeat_rule
        if reminders:
            habit["reminders"] = reminders
        if target_days:
            habit["targetDays"] = target_days
        if target_start_date:
            habit["targetStartDate"] = target_start_date
        if sort_order is not None:
            habit["sortOrder"] = sort_order

        return await self.batch_habits(add=[habit])

    async def update_habit(
        self,
        habit_id: str,
        *,
        name: str | None = None,
        habit_type: str | None = None,
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
        record_enable: bool | None = None,
        status: int | None = None,
        total_checkins: int | None = None,
        current_streak: int | None = None,
    ) -> BatchResponseV2:
        """
        Update a habit.

        Args:
            habit_id: Habit ID
            name: New name
            habit_type: New type
            goal: New goal
            step: New step
            unit: New unit
            icon: New icon
            color: New color
            section_id: New section ID
            repeat_rule: New repeat rule
            reminders: New reminders
            target_days: New target days
            encouragement: New encouragement
            record_enable: New record enable
            status: New status (0=active, 2=archived)
            total_checkins: New total check-ins
            current_streak: New current streak

        Returns:
            Batch response
        """
        from datetime import datetime

        habit: HabitUpdateV2 = {
            "id": habit_id,
            "modifiedTime": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.000+0000"),
        }

        if name is not None:
            habit["name"] = name
        if habit_type is not None:
            habit["type"] = habit_type
        if goal is not None:
            habit["goal"] = goal
        if step is not None:
            habit["step"] = step
        if unit is not None:
            habit["unit"] = unit
        if icon is not None:
            habit["iconRes"] = icon
        if color is not None:
            habit["color"] = color
        if section_id is not None:
            habit["sectionId"] = section_id
        if repeat_rule is not None:
            habit["repeatRule"] = repeat_rule
        if reminders is not None:
            habit["reminders"] = reminders
        if target_days is not None:
            habit["targetDays"] = target_days
        if encouragement is not None:
            habit["encouragement"] = encouragement
        if record_enable is not None:
            habit["recordEnable"] = record_enable
        if status is not None:
            habit["status"] = status
        if total_checkins is not None:
            habit["totalCheckIns"] = total_checkins
        if current_streak is not None:
            habit["currentStreak"] = current_streak

        return await self.batch_habits(update=[habit])

    async def delete_habit(self, habit_id: str) -> BatchResponseV2:
        """
        Delete a habit.

        Args:
            habit_id: Habit ID

        Returns:
            Batch response
        """
        return await self.batch_habits(delete=[habit_id])

    async def checkin_habit(
        self,
        habit_id: str,
        *,
        value: float = 1.0,
        current_total: int = 0,
        current_streak: int = 0,
    ) -> BatchResponseV2:
        """
        Check in a habit (complete it for today).

        This updates the habit's totalCheckIns and currentStreak.

        Args:
            habit_id: Habit ID
            value: Check-in value (1.0 for boolean habits)
            current_total: Current total check-ins (will be incremented)
            current_streak: Current streak (will be incremented)

        Returns:
            Batch response
        """
        return await self.update_habit(
            habit_id=habit_id,
            total_checkins=current_total + int(value),
            current_streak=current_streak + 1,
        )

    async def archive_habit(self, habit_id: str) -> BatchResponseV2:
        """
        Archive a habit.

        Args:
            habit_id: Habit ID

        Returns:
            Batch response
        """
        return await self.update_habit(habit_id=habit_id, status=2)

    async def unarchive_habit(self, habit_id: str) -> BatchResponseV2:
        """
        Unarchive a habit.

        Args:
            habit_id: Habit ID

        Returns:
            Batch response
        """
        return await self.update_habit(habit_id=habit_id, status=0)

    async def get_habit_checkins(
        self,
        habit_ids: list[str],
        after_stamp: int = 0,
    ) -> Any:
        """
        Query habit check-ins.

        Args:
            habit_ids: List of habit IDs to query
            after_stamp: Date stamp (YYYYMMDD) to get check-ins after

        Returns:
            Habit check-in data with structure:
            {
                "checkins": {
                    "habit_id": [
                        {"habitId": "...", "checkinStamp": 20251218, "value": 1.0, ...}
                    ]
                }
            }
        """
        data = {
            "habitIds": habit_ids,
            "afterStamp": after_stamp,
        }
        response = await self._post_json("/habitCheckins/query", json_data=data)
        return response

    # =========================================================================
    # Health Check
    # =========================================================================

    async def verify_authentication(self) -> bool:
        """
        Verify that authentication is working by syncing.

        Returns:
            True if authentication is valid

        Raises:
            TickTickAuthenticationError: If not authenticated
        """
        if not self.is_authenticated:
            raise TickTickAuthenticationError(
                "V2 API not authenticated - no session available"
            )

        try:
            # Try to sync as a health check
            await self.sync()
            return True
        except TickTickAuthenticationError:
            raise
        except Exception as e:
            logger.warning("V2 authentication verification failed: %s", e)
            return False
