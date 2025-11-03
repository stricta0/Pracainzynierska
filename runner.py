#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prosty, odporny runner:
- Czyta jobs.txt (jedna linia = jedno polecenie).
- Oznacza linie: [RUNNING] / [DONE] / [FAIL n].
- Uruchamia do --workers zadań równolegle.
- Wznawia po przerwaniu (RUNNING -> do kolejki).
- Cross-platform, bez dodatkowych pakietów.
"""

from __future__ import annotations
import argparse
import datetime as dt
import os
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

STATE_RUNNING = "[RUNNING]"
STATE_DONE    = "[DONE]"
STATE_FAIL    = "[FAIL"  # np. "[FAIL 1]"

class FileLock:
    """Prosty lock plikowy (działa na Win/Linux)."""
    def __init__(self, path: Path, retry_ms: int = 100, timeout_s: Optional[int] = None):
        self.path = Path(path)
        self.retry_ms = retry_ms
        self.timeout_s = timeout_s
        self._fd = None
    def acquire(self):
        start = time.time()
        while True:
            try:
                self._fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(self._fd, str(os.getpid()).encode("utf-8"))
                return
            except FileExistsError:
                if self.timeout_s is not None and (time.time() - start) > self.timeout_s:
                    raise TimeoutError(f"Timeout acquiring lock: {self.path}")
                time.sleep(self.retry_ms / 1000.0)
    def release(self):
        if self._fd is not None:
            try: os.close(self._fd)
            except Exception: pass
        try:
            if self.path.exists(): self.path.unlink()
        except Exception: pass
    def __enter__(self): self.acquire(); return self
    def __exit__(self, *_): self.release()

def _strip_state_prefix(line: str) -> Tuple[str, Optional[str]]:
    line = line.rstrip("\n")
    if line.startswith(STATE_RUNNING):
        return line[len(STATE_RUNNING):].lstrip(), STATE_RUNNING
    if line.startswith(STATE_DONE):
        return line[len(STATE_DONE):].lstrip(), STATE_DONE
    if line.startswith(STATE_FAIL):
        end = line.find("]")
        state = line[:end+1] if end != -1 else None
        rest = line[end+1:].lstrip() if end != -1 else line
        return rest, state
    return line, None

def read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()

def write_lines(path: Path, lines: List[str]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)

def is_comment_or_empty(line: str) -> bool:
    raw, _ = _strip_state_prefix(line)
    s = raw.strip()
    return (s == "") or s.startswith("#")

def pending_indices(lines: List[str]) -> List[int]:
    idxs = []
    for i, ln in enumerate(lines):
        if is_comment_or_empty(ln): continue
        _, st = _strip_state_prefix(ln)
        if (st is None) or (st == STATE_RUNNING):
            idxs.append(i)
    return idxs

def mark_line(lines: List[str], i: int, new_state: str) -> None:
    raw, _ = _strip_state_prefix(lines[i])
    lines[i] = f"{new_state} {raw}"

def bump_fail(lines: List[str], i: int) -> None:
    raw, old = _strip_state_prefix(lines[i])
    cnt = 0
    if old and old.startswith(STATE_FAIL):
        try: cnt = int(old[6:-1].strip())
        except Exception: cnt = 0
    lines[i] = f"[FAIL {cnt+1}] {raw}"

def split_cmd(cmd: str) -> List[str]:
    return shlex.split(cmd, posix=(os.name != "nt"))

def short_log_name(tokens: List[str], i: int) -> str:
    base = Path(tokens[0]).stem if tokens else "cmd"
    return f"{i:04d}_{base}"[:80]

def run_one(i: int, jobs: Path, lock: Path, logs_dir: Path, retry: int) -> Tuple[int, bool, str]:
    # Pobierz i oznacz RUNNING
    with FileLock(lock):
        lines = read_lines(jobs)
        raw_cmd, _ = _strip_state_prefix(lines[i])
        mark_line(lines, i, STATE_RUNNING)
        write_lines(jobs, lines)

    tokens = split_cmd(raw_cmd)
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{stamp}_{short_log_name(tokens, i)}.log"

    attempts, rc = 0, None
    with open(log_file, "w", encoding="utf-8") as lf:
        while attempts <= retry:
            attempts += 1
            lf.write(f"=== RUN #{attempts} | line={i} | cmd: {raw_cmd}\n"); lf.flush()
            try:
                proc = subprocess.Popen(
                    tokens,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                )
                for line in iter(proc.stdout.readline, ""):
                    lf.write(line)
                proc.stdout.close()
                rc = proc.wait()
                lf.write(f"\n=== RETURN CODE: {rc}\n"); lf.flush()
                if rc == 0:
                    break
            except Exception as e:
                lf.write(f"\n=== EXCEPTION: {e}\n"); lf.flush()
                rc = -1

    # Zaktualizuj status
    with FileLock(lock):
        lines = read_lines(jobs)
        if rc == 0:
            mark_line(lines, i, STATE_DONE)
        else:
            bump_fail(lines, i)
        write_lines(jobs, lines)

    return i, (rc == 0), str(log_file)

def main():
    ap = argparse.ArgumentParser(description="Prosty runner eksperymentów (równoległy, wznawialny).")
    ap.add_argument("--jobs", required=True, type=Path, help="Plik z zadaniami (jedna linia = jedna komenda).")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Ile zadań równolegle.")
    ap.add_argument("--retry", type=int, default=0, help="Ile razy ponowić nieudane zadanie.")
    ap.add_argument("--logs_dir", type=Path, default=Path("logs"), help="Katalog na logi.")
    args = ap.parse_args()

    jobs_path: Path = args.jobs
    if not jobs_path.exists():
        print(f"[ERR] Brak pliku: {jobs_path}"); return
    lock_path: Path = jobs_path.with_suffix(jobs_path.suffix + ".lock")

    with FileLock(lock_path):
        lines = read_lines(jobs_path)
    idxs = pending_indices(lines)

    total = len([ln for ln in lines if not is_comment_or_empty(ln)])
    print(f"[INFO] Wszystkich zadań: {total} | teraz uruchomię: {len(idxs)}")

    if not idxs:
        print("[INFO] Nic do zrobienia (wszystko DONE albo FAIL)."); return

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = [ex.submit(run_one, i, jobs_path, lock_path, args.logs_dir, int(args.retry)) for i in idxs]
        for f in as_completed(futs):
            i, success, logp = f.result()
            if success:
                ok += 1; print(f"[OK]   line {i} | log: {logp}")
            else:
                fail += 1; print(f"[FAIL] line {i} | log: {logp}")

    print(f"[SUM] OK={ok} | FAIL={fail} | LOGS={args.logs_dir.resolve()}")

if __name__ == "__main__":
    main()
