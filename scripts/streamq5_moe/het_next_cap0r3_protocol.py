#!/usr/bin/env python3
"""CAP0 Win32 protocol plus the exact mock used by the static preflight.

Importing this module performs no OS or device call.  Win32 DLLs are opened only
when ``Win32Primitives`` is instantiated by an authorized physical run.
"""
from __future__ import annotations

import ctypes as C
import os
import uuid
from pathlib import Path

WORKERS = ("intel", "nvidia")
REPETITIONS = 3
WAIT_MS = 30_000


class MockEvent:
    def __init__(self, manual: bool, initial: bool, name: str):
        self.manual, self.state, self.name = manual, initial, name


class MockPrimitives:
    def __init__(self):
        self.calls, self.qpc_value = [], 0

    def event(self, manual, initial, name):
        self.calls.append(["CreateEventW", name, bool(manual), bool(initial)])
        return MockEvent(bool(manual), bool(initial), name)

    def set(self, event):
        event.state = True; self.calls.append(["SetEvent", event.name]); return True

    def reset(self, event):
        event.state = False; self.calls.append(["ResetEvent", event.name]); return True

    def wait_one(self, event, timeout=WAIT_MS):
        self.calls.append(["WaitForSingleObject", event.name, timeout])
        value = event.state
        if value and not event.manual: event.state = False
        return value

    def wait_all(self, events, timeout=WAIT_MS):
        self.calls.append(["WaitForMultipleObjects", [e.name for e in events], timeout, True])
        return all(e.state for e in events)

    def exchange(self, cell, value):
        old = cell[0]; cell[0] = int(value)
        self.calls.append(["InterlockedExchange64", old, int(value)]); return old

    def read(self, cell):
        value = cell[0]; self.calls.append(["InterlockedCompareExchange64", value]); return value

    def memory_barrier(self): self.calls.append(["MemoryBarrier"])
    def lock_exclusive(self, lock): self.calls.append(["AcquireSRWLockExclusive", lock])
    def unlock_exclusive(self, lock): self.calls.append(["ReleaseSRWLockExclusive", lock])
    def new_lock(self): return "cap0_srwlock"

    def qpc(self):
        self.qpc_value += 100
        self.calls.append(["QueryPerformanceCounter", self.qpc_value]); return self.qpc_value

    def close(self, event): self.calls.append(["CloseHandle", event.name])


class Win32Primitives:
    def __init__(self, ledger=None):
        from ctypes import wintypes
        self.w = wintypes; self.k = C.WinDLL("kernel32", use_last_error=True); self.calls = []; self.ledger = ledger; self.event_rows = {}
        k = self.k
        k.CreateEventW.argtypes = [C.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]; k.CreateEventW.restype = wintypes.HANDLE
        k.SetEvent.argtypes = [wintypes.HANDLE]; k.SetEvent.restype = wintypes.BOOL
        k.ResetEvent.argtypes = [wintypes.HANDLE]; k.ResetEvent.restype = wintypes.BOOL
        k.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]; k.WaitForSingleObject.restype = wintypes.DWORD
        k.WaitForMultipleObjects.argtypes = [wintypes.DWORD, C.POINTER(wintypes.HANDLE), wintypes.BOOL, wintypes.DWORD]; k.WaitForMultipleObjects.restype = wintypes.DWORD
        k.CloseHandle.argtypes = [wintypes.HANDLE]; k.CloseHandle.restype = wintypes.BOOL
        k.QueryPerformanceCounter.argtypes = [C.POINTER(C.c_longlong)]; k.QueryPerformanceCounter.restype = wintypes.BOOL
        k.InterlockedExchange64.argtypes = [C.POINTER(C.c_longlong), C.c_longlong]; k.InterlockedExchange64.restype = C.c_longlong
        k.InterlockedCompareExchange64.argtypes = [C.POINTER(C.c_longlong), C.c_longlong, C.c_longlong]; k.InterlockedCompareExchange64.restype = C.c_longlong
        k.InitializeSRWLock.argtypes = [C.c_void_p]; k.InitializeSRWLock.restype = None
        k.AcquireSRWLockExclusive.argtypes = [C.c_void_p]; k.AcquireSRWLockExclusive.restype = None
        k.ReleaseSRWLockExclusive.argtypes = [C.c_void_p]; k.ReleaseSRWLockExclusive.restype = None

    def _ok(self, value, name):
        if not value: raise OSError(C.get_last_error(), name)

    def event(self, manual, initial, name):
        h = self.k.CreateEventW(None, bool(manual), bool(initial), name); self._ok(h, "CreateEventW")
        self.calls.append(["CreateEventW", name, bool(manual), bool(initial)])
        if self.ledger is not None: self.event_rows[int(h)] = self.ledger.create("win32_event", threading.current_thread().name, {"name": name, "handle": int(h), "manual_reset": bool(manual)})
        return h

    def set(self, event): self._ok(self.k.SetEvent(event), "SetEvent"); self.calls.append(["SetEvent", int(event)]); return True
    def reset(self, event): self._ok(self.k.ResetEvent(event), "ResetEvent"); self.calls.append(["ResetEvent", int(event)]); return True

    def wait_one(self, event, timeout=WAIT_MS):
        result = int(self.k.WaitForSingleObject(event, timeout)); self.calls.append(["WaitForSingleObject", int(event), timeout, result]); return result == 0

    def wait_all(self, events, timeout=WAIT_MS):
        handles = (self.w.HANDLE * len(events))(*events)
        result = int(self.k.WaitForMultipleObjects(len(events), handles, True, timeout))
        self.calls.append(["WaitForMultipleObjects", [int(e) for e in events], timeout, True, result]); return result == 0

    def exchange(self, cell, value):
        old = int(self.k.InterlockedExchange64(C.byref(cell), int(value))); self.calls.append(["InterlockedExchange64", old, int(value)]); return old

    def read(self, cell):
        value = int(self.k.InterlockedCompareExchange64(C.byref(cell), 0, 0)); self.calls.append(["InterlockedCompareExchange64", value]); return value

    def memory_barrier(self):
        # A locked RMW is a full Windows memory barrier and is independently logged.
        scratch = C.c_longlong(0); self.k.InterlockedCompareExchange64(C.byref(scratch), 0, 0); self.calls.append(["MemoryBarrier"])

    def new_lock(self):
        lock = C.c_void_p(); self.k.InitializeSRWLock(C.byref(lock))
        if self.ledger is not None: self.srw_row = self.ledger.create("win32_srwlock", threading.current_thread().name, {"address": C.addressof(lock), "destroy_api": "none_by_contract"})
        return lock

    def lock_exclusive(self, lock): self.k.AcquireSRWLockExclusive(C.byref(lock)); self.calls.append(["AcquireSRWLockExclusive"])
    def unlock_exclusive(self, lock): self.k.ReleaseSRWLockExclusive(C.byref(lock)); self.calls.append(["ReleaseSRWLockExclusive"])

    def qpc(self):
        value = C.c_longlong(); self._ok(self.k.QueryPerformanceCounter(C.byref(value)), "QueryPerformanceCounter")
        self.calls.append(["QueryPerformanceCounter", int(value.value)]); return int(value.value)

    def close(self, event):
        def release(): self._ok(self.k.CloseHandle(event), "CloseHandle"); self.calls.append(["CloseHandle", int(event)]); return 0
        if self.ledger is not None and int(event) in self.event_rows: self.ledger.release(self.event_rows.pop(int(event)), release)
        else: release()


class CachelineCellStorage(C.Structure):
    _fields_ = [("value", C.c_longlong), ("padding", C.c_ubyte * 120)]
    _align_ = 128


class Cell:
    def __init__(self, primitives, value=0):
        self.p = primitives
        self.storage = [int(value)] if isinstance(primitives, MockPrimitives) else CachelineCellStorage(int(value))
        self.value = self.storage if isinstance(primitives, MockPrimitives) else self.storage.value
    def read(self):
        if isinstance(self.p, MockPrimitives): return self.p.read(self.value)
        return self.p.read(C.c_longlong.from_address(C.addressof(self.storage)))
    def set(self, value):
        if isinstance(self.p, MockPrimitives): return self.p.exchange(self.value, value)
        return self.p.exchange(C.c_longlong.from_address(C.addressof(self.storage)), value)


class Channel:
    def __init__(self, p, name):
        self.name = name
        self.command = p.event(False, False, f"cap0_{name}_command")
        self.initialized = p.event(True, False, f"cap0_{name}_initialized")
        self.ready = p.event(True, False, f"cap0_{name}_ready")
        self.done = p.event(True, False, f"cap0_{name}_done")
        self.stop = p.event(True, False, f"cap0_{name}_stop")
        self.last, self.ack, self.descriptor, self.output = Cell(p), Cell(p), {}, None


class DualDeviceProtocol:
    """The same epoch functions are used by the mock preflight and physical runner."""
    def __init__(self, p):
        self.p, self.epoch, self.log = p, 0, []
        self.lock = p.new_lock(); self.start = p.event(True, False, "cap0_start")
        self.channels = {name: Channel(p, name) for name in WORKERS}

    def worker_initialized(self, name, init_row):
        self.channels[name].output = {"initialization": init_row}
        self.p.memory_barrier(); self.p.set(self.channels[name].initialized)
        self.log.append({"op": "initialized", "worker": name})

    def wait_initialized(self):
        if not self.p.wait_all([self.channels[n].initialized for n in WORKERS]): raise TimeoutError("initialization")
        self.p.memory_barrier()
        rows = {n: self.channels[n].output["initialization"] for n in WORKERS}
        self.log.append({"op": "initialization_collected", "workers": list(WORKERS)}); return rows

    def publish(self, repetition):
        if repetition != self.epoch + 1 or repetition not in (1, 2, 3): raise RuntimeError("epoch_not_strict")
        if any(c.ack.read() != c.last.read() for c in self.channels.values()): raise RuntimeError("stale_ack")
        self.p.reset(self.start); self.p.lock_exclusive(self.lock)
        try:
            self.epoch = repetition
            for name in WORKERS:
                c = self.channels[name]; self.p.reset(c.ready); self.p.reset(c.done)
                c.output = None; c.descriptor = {"epoch": repetition, "repetition": repetition, "active": list(WORKERS)}
                c.last.set(repetition); self.p.set(c.command)
        finally: self.p.unlock_exclusive(self.lock)
        self.log.append({"op": "publish", "epoch": repetition, "active": list(WORKERS)})

    def worker_descriptor(self, name):
        c = self.channels[name]
        if not self.p.wait_one(c.command): raise TimeoutError("command")
        self.p.lock_exclusive(self.lock)
        try: descriptor = dict(c.descriptor)
        finally: self.p.unlock_exclusive(self.lock)
        if descriptor.get("epoch") <= c.ack.read() or c.last.read() != descriptor.get("epoch"): raise RuntimeError("worker_epoch")
        self.p.set(c.ready); self.log.append({"op": "ready", "worker": name, "epoch": descriptor["epoch"]}); return descriptor

    def coordinator_release(self, repetition):
        if not self.p.wait_all([self.channels[n].ready for n in WORKERS]): raise TimeoutError("ready")
        if repetition != self.epoch: raise RuntimeError("release_epoch")
        qpc = self.p.qpc(); self.p.set(self.start)
        self.log.append({"op": "release", "epoch": repetition, "qpc": qpc}); return qpc

    def worker_start(self, name, repetition):
        if not self.p.wait_one(self.start): raise TimeoutError("start")
        if self.channels[name].last.read() != repetition: raise RuntimeError("start_epoch")

    def worker_finish(self, name, repetition, output):
        c = self.channels[name]
        if c.last.read() != repetition: raise RuntimeError("finish_epoch")
        c.output = output; self.p.memory_barrier(); c.ack.set(repetition); self.p.set(c.done)
        self.log.append({"op": "ack_done", "worker": name, "epoch": repetition})

    def collect(self, repetition):
        if not self.p.wait_all([self.channels[n].done for n in WORKERS]): raise TimeoutError("done")
        if any(c.ack.read() != repetition or c.last.read() != repetition for c in self.channels.values()): raise RuntimeError("collect_ack")
        self.p.memory_barrier(); outputs = {n: self.channels[n].output for n in WORKERS}
        if any(v is None for v in outputs.values()): raise RuntimeError("premature_read")
        qpc = self.p.qpc(); self.p.reset(self.start)
        for c in self.channels.values(): self.p.reset(c.ready); self.p.reset(c.done)
        self.log.append({"op": "collect_reset", "epoch": repetition, "qpc": qpc}); return outputs, qpc

    def request_stop(self):
        for c in self.channels.values(): self.p.set(c.stop); self.p.set(c.command)

    def close(self):
        errors = []
        for c in self.channels.values():
            for e in (c.command, c.initialized, c.ready, c.done, c.stop):
                try:self.p.close(e)
                except BaseException as exc:errors.append(f'event:{c.name}:{type(exc).__name__}:{exc}')
        try:self.p.close(self.start)
        except BaseException as exc:errors.append(f'start:{type(exc).__name__}:{exc}')
        if getattr(self.p, "ledger", None) is not None and hasattr(self.p, "srw_row"):
            try:self.p.ledger.release(self.p.srw_row, lambda: 0)
            except BaseException as exc:errors.append(f'srw:{type(exc).__name__}:{exc}')
        if errors:raise RuntimeError(';'.join(errors))


def atomic_create_json(path: Path, payload: bytes):
    """Create-new/fsync/rename, shared by production and failure simulation."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".inprogress")
    with temp.open("xb") as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    if path.exists(): raise FileExistsError(path)
    if os.name == "nt":
        move = C.WinDLL("kernel32", use_last_error=True).MoveFileExW; move.argtypes = [C.c_wchar_p, C.c_wchar_p, C.c_ulong]; move.restype = C.c_int
        if not move(str(temp), str(path), 0x8): raise OSError(C.get_last_error(), "MoveFileExW_WRITE_THROUGH")
        with path.open("r+b") as handle: handle.flush(); os.fsync(handle.fileno())
    else:
        os.rename(temp, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    return path


def simulate_protocol():
    p = MockPrimitives(); protocol = DualDeviceProtocol(p); outputs = []
    for name in WORKERS: protocol.worker_initialized(name, {"worker": name})
    initialized = protocol.wait_initialized()
    for epoch in range(1, REPETITIONS + 1):
        protocol.publish(epoch); descriptors = {n: protocol.worker_descriptor(n) for n in WORKERS}
        t0 = protocol.coordinator_release(epoch)
        for name in WORKERS:
            protocol.worker_start(name, epoch); protocol.worker_finish(name, epoch, {"epoch": epoch, "worker": name})
        row, t1 = protocol.collect(epoch); outputs.append({"epoch": epoch, "descriptors": descriptors, "outputs": row, "t0": t0, "t1": t1})
    negatives = {}
    protocol.channels["intel"].ack.set(0); protocol.epoch = 2
    try: protocol.publish(3); negatives["stale_ack"] = False
    except RuntimeError: negatives["stale_ack"] = True
    try: protocol.worker_finish("intel", 99, {}); negatives["wrong_epoch"] = False
    except RuntimeError: negatives["wrong_epoch"] = True
    negatives["timeout"] = not p.wait_one(MockEvent(True, False, "timeout"), 1)
    for c in protocol.channels.values(): c.done.state = True; c.ack.value[0] = 3; c.last.value[0] = 3
    try:
        protocol.channels["intel"].output = None; protocol.collect(3); negatives["premature_read"] = False
    except (RuntimeError, TimeoutError): negatives["premature_read"] = True
    protocol.close()
    return {"pass": len(initialized) == 2 and len(outputs) == 3 and all(negatives.values()), "initialized": initialized, "outputs": outputs, "negative": negatives, "calls": p.calls, "log": protocol.log}


def simulate_atomic_failure(directory: Path):
    directory = Path(directory); target = directory / "result.json"
    atomic_create_json(target, b'{"ok":true}\n')
    overwrite_rejected = False
    try: atomic_create_json(target, b'{"ok":false}\n')
    except FileExistsError: overwrite_rejected = True
    partial = directory / "failure.json.partial.inprogress"; partial.write_bytes(b"partial")
    quarantine = directory / "failed_attempts"; quarantine.mkdir(exist_ok=True)
    moved = quarantine / partial.name; os.rename(partial, moved)
    return {"overwrite_rejected": overwrite_rejected, "quarantined": moved.exists(), "result_preserved": target.read_bytes() == b'{"ok":true}\n'}


def simulate_release_failure():
    attempts = []
    for name in ("third", "second", "first"):
        try:
            attempts.append(name)
            if name == "second": raise RuntimeError("injected_release_failure")
        except RuntimeError:
            pass
    return {"all_attempted": attempts == ["third", "second", "first"], "failure_observed": "second" in attempts}


def recover_transaction(directory: Path):
    directory = Path(directory); bad = directory / "failed_attempts"; moved = []
    if not directory.exists(): return moved
    result, commit = directory / "cap0r3_result.json", directory / "cap0r3_commit.json"
    valid = False
    if result.exists() and commit.exists():
        try:
            marker = __import__('json').loads(commit.read_text())
            valid = marker.get('result', {}).get('bytes') == result.stat().st_size and marker['result'].get('sha256') == __import__('hashlib').sha256(result.read_bytes()).hexdigest()
        except Exception: valid = False
    if valid: return [{"valid_commit_preserved": True}]
    bad.mkdir(exist_ok=True)
    for item in list(directory.glob('*.inprogress')) + [x for x in (result, commit, directory/'cap0r3_failure.json') if x.exists()]:
        target = bad / (uuid.uuid4().hex + '_' + item.name); os.rename(item, target); moved.append({"from": item.name, "to": target.name})
    return moved

