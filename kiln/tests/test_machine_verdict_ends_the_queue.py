"""The machine's verdict ends the job — in the queue, not only in the ledger.

Klipper prints end in the same ``print_stats``-driven IDLE whether they
finished or the firmware raised a fault (``2026-09-14``, Snapmaker U1: a
Z-probe fault — ``No trigger on probe after full movement`` — cancelled the
retraction tower 52 s in).  ``PrinterState.last_job_result`` is the only word
that tells those endings apart, and the scheduler read it only for the
outcome ROW: the queue row was stamped ``COMPLETED`` first, a
``job.completed`` event was published, and the dispatch loop — seeing the
printer idle and the queue empty — sent the next job 1.5 s later.  The user
was never told the print had failed; the machine had already retracted.

The adapter layer already reports the machine's verdict (the
``auto_record_hook`` ledger logged ``('printing' → 'cancelled')`` at the very
same moment, and the outcome row said ``cancelled, observed``).  So the two
records disagreed: the ledger told the truth, the queue lied.  These tests
pin the queue's honesty.
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from kiln.events import EventBus, EventType
from kiln.printers.base import (
    JobProgress,
    JobResult,
    PrinterCapabilities,
    PrinterState,
    PrinterStatus,
    PrintResult,
)
from kiln.queue import JobStatus, PrintQueue
from kiln.registry import PrinterRegistry
from kiln.scheduler import JobScheduler


def make_mock_adapter(
    name: str = "printer-1",
    state: PrinterStatus = PrinterStatus.IDLE,
    last_job_result: JobResult | None = None,
) -> MagicMock:
    adapter = MagicMock()
    type(adapter).name = PropertyMock(return_value=name)
    type(adapter).capabilities = PropertyMock(return_value=PrinterCapabilities())
    adapter.get_state.return_value = PrinterState(
        connected=True,
        state=state,
        last_job_result=last_job_result,
    )
    adapter.get_job.return_value = JobProgress(file_name=None, completion=None)
    adapter.start_print.return_value = PrintResult(success=True, message="OK")
    return adapter


@pytest.fixture()
def queue():
    return PrintQueue()


@pytest.fixture()
def registry():
    return PrinterRegistry()


@pytest.fixture()
def event_bus():
    return EventBus()


@pytest.fixture()
def scheduler(queue, registry, event_bus):
    return JobScheduler(queue, registry, event_bus, poll_interval=0.1, max_retries=0)


@pytest.fixture(autouse=True)
def _reset_emergency_state(monkeypatch):
    monkeypatch.setenv("KILN_EMERGENCY_PERSIST", "0")
    import kiln.emergency as _emergency_mod

    _emergency_mod._coordinator = None
    yield
    _emergency_mod._coordinator = None


def _watch_one_print(queue, registry, scheduler, adapter, file_name="tower.gcode"):
    """Dispatch a job and get it seen printing — the watch the scheduler owes
    every job it starts."""
    job_id = queue.submit(file_name=file_name)
    scheduler.tick()  # dispatch (printer is IDLE)
    adapter.get_state.return_value = PrinterState(
        connected=True, state=PrinterStatus.PRINTING, last_job_result=None,
    )
    scheduler.tick()  # seen printing
    return job_id


class TestMachineCancelEndsTheJobCancelled:
    """A firmware fault cancels the print; the machine says so.  Every record
    must carry that verdict — the outcome row, the queue row, and the
    dispatch loop's idea of what just happened."""

    def test_the_u1_incident_replica(
        self, queue, registry, event_bus, scheduler,
    ):
        """2026-09-14, replayed at the unit scale: tower dispatched, seen
        printing, probe fault, machine cancel.  The queue row may never say
        completed, and the next queued job must NOT be dispatched."""
        adapter = make_mock_adapter(name="u1")
        registry.register("u1", adapter)
        tower = _watch_one_print(queue, registry, scheduler, adapter,
                                 file_name="calib/retraction.gcode")
        cornering = queue.submit(file_name="calib/cornering.gcode")

        # 16:20:16 — klippy raise exception: No trigger on probe.  Moonraker
        # folds the fault into print_stats.state=cancelled; the printer
        # reads idle either way.
        adapter.get_state.return_value = PrinterState(
            connected=True, state=PrinterStatus.IDLE,
            last_job_result=JobResult.CANCELLED,
        )
        result = scheduler.tick()

        assert queue.get_job(tower).status == JobStatus.CANCELLED
        # The dispatch loop must not read the machine's abort as a green
        # light: the printer just told us the last print did not finish.
        dispatched_ids = [d["job_id"] for d in result["dispatched"]]
        assert cornering not in dispatched_ids
        assert cornering not in scheduler.active_jobs
        # And the tick ledger says what ended — not what was hoped.
        assert tower not in result["completed"]
        assert {"job_id": tower, "printer_name": "u1"} in result["cancelled"]

    def test_machine_cancelled_gets_job_cancelled_event(
        self, queue, registry, event_bus, scheduler,
    ):
        """The user-facing notification rides the event bus.  A fault-cancel
        that arrives as ``job.completed`` tells nobody anything."""
        adapter = make_mock_adapter(name="u1")
        registry.register("u1", adapter)
        job_id = _watch_one_print(queue, registry, scheduler, adapter)

        adapter.get_state.return_value = PrinterState(
            connected=True, state=PrinterStatus.IDLE,
            last_job_result=JobResult.CANCELLED,
        )
        scheduler.tick()

        completed = event_bus.recent_events(EventType.JOB_COMPLETED)
        assert completed == [], (
            "a machine-cancelled print must never publish job.completed"
        )
        cancelled = event_bus.recent_events(EventType.JOB_CANCELLED)
        assert len(cancelled) == 1
        assert cancelled[0].data["job_id"] == job_id
        assert cancelled[0].data["printer_name"] == "u1"

    def test_machine_cancelled_queue_row_is_not_completed(
        self, queue, registry, event_bus, scheduler,
    ):
        """The direct lie, pinned on its own: the queue row said COMPLETED
        for a print the machine cancelled 52 s in."""
        adapter = make_mock_adapter(name="u1")
        registry.register("u1", adapter)
        job_id = _watch_one_print(queue, registry, scheduler, adapter)

        adapter.get_state.return_value = PrinterState(
            connected=True, state=PrinterStatus.IDLE,
            last_job_result=JobResult.CANCELLED,
        )
        scheduler.tick()

        row = queue.get_job(job_id)
        assert row.status == JobStatus.CANCELLED
        assert row.status != JobStatus.COMPLETED
        assert job_id not in scheduler.active_jobs


class TestMachineFailedEndsTheJobFailed:
    """An ending the adapter reports as FAILED (a protocol that names it) is
    a failure verdict — not a completion."""

    def test_machine_failed_gets_failed_queue_row_and_event(
        self, queue, registry, event_bus, scheduler,
    ):
        adapter = make_mock_adapter(name="printer-1")
        registry.register("printer-1", adapter)
        job_id = _watch_one_print(queue, registry, scheduler, adapter)

        adapter.get_state.return_value = PrinterState(
            connected=True, state=PrinterStatus.IDLE,
            last_job_result=JobResult.FAILED,
        )
        result = scheduler.tick()

        assert queue.get_job(job_id).status == JobStatus.FAILED
        failed_ids = [f["job_id"] for f in result["failed"]]
        assert job_id in failed_ids
        failed_events = event_bus.recent_events(EventType.JOB_FAILED)
        assert len(failed_events) == 1
        assert failed_events[0].data["job_id"] == job_id
        completed_events = event_bus.recent_events(EventType.JOB_COMPLETED)
        assert completed_events == []


class TestNoNamedEndingStillCompletes:
    """Firmware that never names an ending (OctoPrint's state flags, RRF's
    object model, an untouched printer) still ends a watched job as
    completed — that inference is the existing, deliberate contract, and
    these tests are what keep the machine-verdict branches from eating it."""

    def test_watched_idle_without_verdict_still_completes(
        self, queue, registry, event_bus, scheduler,
    ):
        adapter = make_mock_adapter(name="printer-1")
        registry.register("printer-1", adapter)
        job_id = _watch_one_print(queue, registry, scheduler, adapter)

        adapter.get_state.return_value = PrinterState(
            connected=True, state=PrinterStatus.IDLE, last_job_result=None,
        )
        result = scheduler.tick()

        assert queue.get_job(job_id).status == JobStatus.COMPLETED
        assert job_id in result["completed"]
        assert result["cancelled"] == []
        assert len(event_bus.recent_events(EventType.JOB_COMPLETED)) == 1

    def test_named_completion_still_completes(
        self, queue, registry, event_bus, scheduler,
    ):
        adapter = make_mock_adapter(name="printer-1")
        registry.register("printer-1", adapter)
        job_id = _watch_one_print(queue, registry, scheduler, adapter)

        adapter.get_state.return_value = PrinterState(
            connected=True, state=PrinterStatus.IDLE,
            last_job_result=JobResult.COMPLETED,
        )
        result = scheduler.tick()

        assert queue.get_job(job_id).status == JobStatus.COMPLETED
        assert job_id in result["completed"]
        assert len(event_bus.recent_events(EventType.JOB_COMPLETED)) == 1


class TestQueueCancelledJobStillRecordsCancelled:
    """The queue's own cancel (user abort from a queue surface) keeps its
    existing path: terminal CANCELLED row, cancelled outcome, no re-print —
    the machine-verdict rewrite must not disturb it."""

    def test_queue_cancel_unchanged(
        self, queue, registry, event_bus, scheduler,
    ):
        adapter = make_mock_adapter(name="printer-1")
        registry.register("printer-1", adapter)
        job_id = _watch_one_print(queue, registry, scheduler, adapter)

        queue.cancel(job_id)
        adapter.get_state.return_value = PrinterState(
            connected=True, state=PrinterStatus.IDLE,
            last_job_result=JobResult.CANCELLED,
        )
        result = scheduler.tick()

        assert queue.get_job(job_id).status == JobStatus.CANCELLED
        assert job_id not in scheduler.active_jobs
        # PrintQueue.cancel() itself publishes nothing — the JOB_CANCELLED
        # event belongs to the surface that asked for the cancel (the MCP
        # queue tool publishes it there).  The scheduler's idle read adds
        # no second one, and adds no job.completed either.
        assert event_bus.recent_events(EventType.JOB_CANCELLED) == []
        assert event_bus.recent_events(EventType.JOB_COMPLETED) == []
        assert job_id not in result["completed"]
        # It is still a real ending: the tick ledger files it under the
        # machine-confirmed cancel, not under completed.
        assert {"job_id": job_id, "printer_name": "printer-1"} in result["cancelled"]


class TestDispatchAfterMachineCancel:
    """The harm was never just the row: the scheduler read its own stamp as
    a green light and dispatched the next file at the machine that had just
    aborted one — the incident's advance was 1.5 s after job.completed.  The
    fix pauses ONE tick: the same poll that learns of a machine-named
    non-finish does not hand out the next file; the next poll does."""

    def test_next_job_not_dispatched_in_the_same_breath(
        self, queue, registry, event_bus, scheduler,
    ):
        adapter = make_mock_adapter(name="u1")
        registry.register("u1", adapter)
        tower = _watch_one_print(queue, registry, scheduler, adapter,
                                 file_name="calib/retraction.gcode")
        cornering = queue.submit(file_name="calib/cornering.gcode")

        adapter.get_state.return_value = PrinterState(
            connected=True, state=PrinterStatus.IDLE,
            last_job_result=JobResult.CANCELLED,
        )
        result = scheduler.tick()

        # The aborted print is closed honestly...
        assert queue.get_job(tower).status == JobStatus.CANCELLED
        assert {"job_id": tower, "printer_name": "u1"} in result["cancelled"]
        # ...and the same tick does NOT read the abort as a green light.
        dispatched_ids = [d["job_id"] for d in result["dispatched"]]
        assert cornering not in dispatched_ids
        assert len(event_bus.recent_events(EventType.JOB_COMPLETED)) == 0

        # One poll later the printer is free again and normal dispatch
        # resumes — the pause costs silence, not the job.
        result2 = scheduler.tick()
        dispatched2 = [d["job_id"] for d in result2["dispatched"]]
        assert cornering in dispatched2
        started = event_bus.recent_events(EventType.JOB_STARTED)
        assert any(e.data["job_id"] == cornering for e in started)

    def test_machine_failed_also_pauses_one_tick(
        self, queue, registry, event_bus, scheduler,
    ):
        adapter = make_mock_adapter(name="u1")
        registry.register("u1", adapter)
        first = _watch_one_print(queue, registry, scheduler, adapter,
                                 file_name="first.gcode")
        second = queue.submit(file_name="second.gcode")

        adapter.get_state.return_value = PrinterState(
            connected=True, state=PrinterStatus.IDLE,
            last_job_result=JobResult.FAILED,
        )
        result1 = scheduler.tick()
        assert queue.get_job(first).status == JobStatus.FAILED
        assert second not in [d["job_id"] for d in result1["dispatched"]]

        result2 = scheduler.tick()
        assert second in [d["job_id"] for d in result2["dispatched"]]


class TestTickLedgerShape:
    """The tick summary is the interface other surfaces read.  A
    machine-cancel belongs under a key that says so."""

    def test_cancelled_key_present_and_typed(
        self, queue, registry, event_bus, scheduler,
    ):
        adapter = make_mock_adapter(name="printer-1")
        registry.register("printer-1", adapter)
        _watch_one_print(queue, registry, scheduler, adapter)

        adapter.get_state.return_value = PrinterState(
            connected=True, state=PrinterStatus.IDLE,
            last_job_result=JobResult.CANCELLED,
        )
        result = scheduler.tick()

        assert "cancelled" in result
        assert result["cancelled"] == [
            {"job_id": result["cancelled"][0]["job_id"], "printer_name": "printer-1"},
        ]

    def test_clean_tick_has_empty_cancelled(
        self, queue, registry, event_bus, scheduler,
    ):
        adapter = make_mock_adapter(name="printer-1")
        registry.register("printer-1", adapter)
        job_id = _watch_one_print(queue, registry, scheduler, adapter)
        adapter.get_state.return_value = PrinterState(
            connected=True, state=PrinterStatus.IDLE, last_job_result=None,
        )
        result = scheduler.tick()
        assert result["cancelled"] == []
        assert job_id in result["completed"]