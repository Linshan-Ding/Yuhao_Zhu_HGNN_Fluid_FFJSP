"""Atomic storage, checksums and safe single-writer ownership."""
from contextlib import contextmanager
from pathlib import Path
import csv
import hashlib
import json
import os
import shutil
import socket
import time
import uuid
import numpy as np


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


TEXT_SUFFIXES = ('.csv', '.json', '.jsonl', '.md', '.yaml', '.yml', '.py', '.txt', '.tex', '.bib')


def normalized_bytes(path):
    """File bytes with CRLF folded to LF for text files, so identities do not depend on the platform
    or on git's end-of-line conversion. Binary files (.pt, .gz, .zip, .pdf, ...) are returned as is."""
    data = Path(path).read_bytes()
    return data.replace(b'\r\n', b'\n') if Path(path).suffix.lower() in TEXT_SUFFIXES else data


def digest(path):
    return hashlib.sha256(normalized_bytes(path)).hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=plain).encode()).hexdigest()


def state_hash(state, prefix=b''):
    """Identity of a network state dict: parameter names, dtypes, shapes and bytes, after an optional prefix."""
    h = hashlib.sha256(prefix)
    for name, value in sorted(state.items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str((a.dtype, a.shape)).encode()); h.update(a.tobytes())
    return h.hexdigest()


def atomic_json(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + '.' + uuid.uuid4().hex + '.tmp')
    with t.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=plain); f.flush(); os.fsync(f.fileno())
    os.replace(t, p)


def atomic_torch_save(value, path):
    import torch
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + '.' + uuid.uuid4().hex + '.tmp')
    with t.open('wb') as f:
        torch.save(value, f); f.flush(); os.fsync(f.fileno())
    os.replace(t, p)


def write_csv(path, rows, fields=None):
    rows = list(rows); p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    t = p.with_name(p.name + '.tmp')
    with t.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n'); w.writeheader(); w.writerows(rows)
    os.replace(t, p)


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as f: return list(csv.DictReader(f))


def disk_check(path, required_bytes=0, reserve_gb=5.):
    p = Path(path).resolve()
    while not p.exists(): p = p.parent
    free = shutil.disk_usage(p).free
    if free < required_bytes + reserve_gb * 2**30:
        raise OSError(f'Insufficient disk space: free={free}, requested={required_bytes}, reserve_gb={reserve_gb}')
    return free


def _pid_alive(pid):
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle: return ctypes.get_last_error() == 5
        status = wintypes.DWORD()
        ok=kernel.GetExitCodeProcess(handle, ctypes.byref(status))
        kernel.CloseHandle(handle)
        if not ok:return True
        return status.value == 259
    try: os.kill(pid, 0); return True
    except ProcessLookupError: return False
    except PermissionError: return True


@contextmanager
def run_lock(directory):
    p = Path(directory); p.mkdir(parents=True, exist_ok=True); lock = p / '.lock'
    record = dict(pid=os.getpid(), host=socket.gethostname(), token=uuid.uuid4().hex, started=time.time())
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, 'w', encoding='utf-8') as f: json.dump(record, f)
            break
        except FileExistsError:
            old = json.loads(lock.read_text(encoding='utf-8'))
            if old['host'] != record['host'] or _pid_alive(old['pid']):
                raise RuntimeError(f'Run is owned by another process: {old}')
            os.replace(lock, p / f'abandoned_lock_{old["token"]}.json')
    else: raise RuntimeError('Cannot acquire run lock')
    try: yield
    finally:
        if lock.exists() and json.loads(lock.read_text())['token'] == record['token']: lock.unlink()
