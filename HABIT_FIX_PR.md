# Fix: Habit updates losing data due to partial API updates

## Problem

Habit check-ins and updates were disappearing after being recorded. For example, checking into a habit would initially show success, but the check-in would vanish shortly after.

## Root Cause

TickTick's `habits/batch` API requires **full object replacement**, not partial updates. When only some fields are sent in an update request, the omitted fields are reset to their default values (null/0).

This means:
- Checking in a habit would reset other fields like `name`, `color`, `goal`, etc.
- Archiving/unarchiving would lose check-in history
- Any habit update would potentially corrupt other habit data

## Solution

Modified all habit update operations to:
1. Fetch the complete original habit first
2. Merge the new values with all original values
3. Send the complete object back to the API

## Changes

### `src/ticktick_sdk/api/v2/client.py`

- Added missing parameters to `update_habit()`:
  - `sort_order`
  - `target_start_date`
  - `completed_cycles`
  - `ex_dates`
  - `style`
  - `etag`
  - `created_time`

- Added new method `full_update_habit(habit_data: dict)` for sending complete habit dictionaries directly to the batch API

### `src/ticktick_sdk/unified/api.py`

Updated the following methods to fetch original habit and send all fields:

- `update_habit()` - Now preserves all original fields when updating specific properties
- `checkin_habit()` - Uses `full_update_habit()` with complete `Habit` object converted via `to_v2_dict()`
- `archive_habit()` - Sends all fields with `status=2`
- `unarchive_habit()` - Sends all fields with `status=0`
- `batch_checkin_habits()` - Uses `full_update_habit()` for each habit update

## Example

Before (broken):
```python
# Only sends 3 fields - everything else gets reset!
await self._v2_client.update_habit(
    habit_id=habit_id,
    name=original_habit.name,
    total_checkins=calculated_total,
    current_streak=calculated_streak,
)
```

After (fixed):
```python
# Create complete habit object with all fields preserved
updated_habit = Habit(
    id=original_habit.id,
    name=original_habit.name,
    icon=original_habit.icon,
    color=original_habit.color,
    sort_order=original_habit.sort_order,
    status=original_habit.status,
    encouragement=original_habit.encouragement,
    total_checkins=calculated_total,
    created_time=original_habit.created_time,
    modified_time=original_habit.modified_time,
    archived_time=original_habit.archived_time,
    habit_type=original_habit.habit_type,
    goal=original_habit.goal,
    step=original_habit.step,
    unit=original_habit.unit,
    etag=original_habit.etag,
    repeat_rule=original_habit.repeat_rule,
    reminders=original_habit.reminders,
    record_enable=original_habit.record_enable,
    section_id=original_habit.section_id,
    target_days=original_habit.target_days,
    target_start_date=original_habit.target_start_date,
    completed_cycles=original_habit.completed_cycles,
    ex_dates=original_habit.ex_dates,
    current_streak=calculated_streak,
    style=original_habit.style,
)
habit_data = updated_habit.to_v2_dict(for_update=True)
await self._v2_client.full_update_habit(habit_data)
```

## Testing

1. Check in to a habit
2. Verify the check-in persists (refresh/re-fetch)
3. Check in again and verify total increments correctly
4. Archive and unarchive a habit, verify all data preserved
