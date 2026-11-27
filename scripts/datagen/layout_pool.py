"""File-lock-backed dynamic layout_id pool for persistent Isaac Sim datagen workers.

Multiple persistent collect.py processes on the same seed folder share one pool file (e.g.
`<output>/.claim_pool.json`) protected by `fcntl.flock` on `<pool>.lock`. Each worker claims one
layout_id at a time via ClaimingSeedManager.get_seeds(), eliminating static --chunk straggler tails.
Crashed/SIGKILLed workers' in-flight claims are automatically reclaimed once their PID exits.
"""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time

BROKEN_TRIES = 2


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class LayoutPool:
    """Flock-protected dynamic layout_id pool shared by workers writing to `output_dir`."""

    def __init__(self, path, output_dir, *, stop_file=None, broken_tries=BROKEN_TRIES, pid_alive=_pid_alive):
        self.path = Path(path)
        self.lock_path = Path(f"{self.path}.lock")
        self.output_dir = Path(output_dir)
        self.stop_file = Path(stop_file) if stop_file else None
        self.broken_tries = int(broken_tries)
        self._pid_alive = pid_alive

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock_fp:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)

    def _read_unlocked(self):
        if not self.path.exists():
            return dict(candidates=[], in_flight={}, skipped=[], broken={})
        data = json.loads(self.path.read_text())
        return dict(
            candidates=[int(x) for x in data.get("candidates", [])],
            in_flight={str(k): v for k, v in (data.get("in_flight") or {}).items()},
            skipped=[int(x) for x in data.get("skipped", [])],
            broken={str(k): int(v) for k, v in (data.get("broken") or {}).items()},
        )

    def _write_unlocked(self, state):
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(self.path)

    def _is_done_on_disk(self, layout_id: int) -> bool:
        return (self.output_dir / f"episode_{int(layout_id):06d}.json").exists()

    def _reap_dead_unlocked(self, state) -> bool:
        changed = False
        for key, meta in list(state["in_flight"].items()):
            pid = int((meta or {}).get("pid", 0))
            if not self._pid_alive(pid) or self._is_done_on_disk(int(key)):
                del state["in_flight"][key]
                changed = True
        return changed

    def ensure_initialized(self, candidates, *, skipped=(), broken=None):
        """Create or merge candidates/skipped/broken into the pool without disturbing live claims."""
        candidates = [int(x) for x in candidates]
        skipped_set = {int(x) for x in skipped}
        broken_map = {str(k): int(v) for k, v in (broken or {}).items()}
        with self._locked():
            state = self._read_unlocked()
            self._reap_dead_unlocked(state)
            seen = set(state["candidates"])
            for lid in candidates:
                if lid not in seen:
                    state["candidates"].append(lid)
                    seen.add(lid)
            merged_skipped = set(state["skipped"]) | skipped_set
            for k, count in broken_map.items():
                state["broken"][k] = max(state["broken"].get(k, 0), count)
                if state["broken"][k] >= self.broken_tries:
                    merged_skipped.add(int(k))
            state["skipped"] = sorted(merged_skipped)
            self._write_unlocked(state)
            return state

    def claim(self, *, worker="", pid=None, count=1) -> list[int]:
        """Claim up to `count` available layout_ids, or [] when none are left or stop_file exists."""
        if count <= 0:
            return []
        if self.stop_file and self.stop_file.exists():
            return []
        pid = os.getpid() if pid is None else int(pid)
        with self._locked():
            if self.stop_file and self.stop_file.exists():
                return []
            state = self._read_unlocked()
            changed = self._reap_dead_unlocked(state)
            skipped = set(state["skipped"])
            claimed = []
            now = time.time()
            for lid in state["candidates"]:
                if len(claimed) >= count:
                    break
                key = str(lid)
                if lid in skipped or key in state["in_flight"] or self._is_done_on_disk(lid):
                    continue
                state["in_flight"][key] = dict(pid=pid, worker=str(worker), claimed_at=now)
                claimed.append(lid)
                changed = True
            if changed:
                self._write_unlocked(state)
            return claimed

    def finish(self, layout_id: int, *, status="done"):
        """Release an in-flight layout_id and record unstable/broken outcomes."""
        lid = int(layout_id)
        key = str(lid)
        with self._locked():
            state = self._read_unlocked()
            state["in_flight"].pop(key, None)
            if status == "unstable":
                if lid not in state["skipped"]:
                    state["skipped"].append(lid)
                    state["skipped"].sort()
            elif status == "broken":
                count = state["broken"].get(key, 0) + 1
                state["broken"][key] = count
                if count >= self.broken_tries and lid not in state["skipped"]:
                    state["skipped"].append(lid)
                    state["skipped"].sort()
            elif status not in ("done", "release"):
                raise ValueError(f"Unknown layout finish status: {status!r}")
            self._write_unlocked(state)

    def has_pending(self) -> bool:
        """True if at least one candidate is not done, not skipped, and not held by a live worker."""
        with self._locked():
            state = self._read_unlocked()
            if self._reap_dead_unlocked(state):
                self._write_unlocked(state)
            skipped = set(state["skipped"])
            for lid in state["candidates"]:
                if lid not in skipped and str(lid) not in state["in_flight"] and not self._is_done_on_disk(lid):
                    return True
            return False

    def snapshot(self) -> dict:
        with self._locked():
            state = self._read_unlocked()
            if self._reap_dead_unlocked(state):
                self._write_unlocked(state)
            return state


class ClaimingSeedManager:
    """SeedManager wrapper that dynamically claims layout_ids from a shared LayoutPool."""

    def __init__(self, inner, pool: LayoutPool, *, worker=""):
        self._inner = inner
        self.pool = pool
        self.worker = worker or f"pid-{os.getpid()}"
        self.idx = 0
        self.ed_idx = max(1, len(getattr(inner, "seed_list", ())))
        self._current_batch_seeds: list[int] | None = None

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def get_seeds(self, max_count: int | None = None) -> list[int] | None:
        if self.idx >= self.ed_idx:
            return None
        num_envs = int(getattr(self._inner, "num_envs", 1))
        batch_size = min(num_envs, max(0, int(max_count))) if max_count is not None else num_envs
        if batch_size <= 0:
            return None
        claimed = self.pool.claim(worker=self.worker, count=batch_size)
        if not claimed:
            return None
        existing = list(self._current_batch_seeds or ())
        self._current_batch_seeds = existing + list(claimed)
        return list(claimed)

    def eval_step(self):
        for lid in self._current_batch_seeds or ():
            self.pool.finish(lid, status="done")
        self._current_batch_seeds = None

    def mark_unstable(self):
        for lid in self._current_batch_seeds or ():
            self.pool.finish(lid, status="unstable")
        self._current_batch_seeds = None

    def mark_broken(self, seeds):
        broken = {int(s) for s in seeds}
        for lid in broken:
            self.pool.finish(lid, status="broken")
        if self._current_batch_seeds:
            self._current_batch_seeds = [s for s in self._current_batch_seeds if int(s) not in broken]

    def release_current(self):
        for lid in self._current_batch_seeds or ():
            self.pool.finish(lid, status="release")
        self._current_batch_seeds = None
