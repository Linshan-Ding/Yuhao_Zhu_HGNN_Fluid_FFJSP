"""Explicit process-local engineering accounting, including spawned workers."""
_counter = None


def install(counter):
    global _counter
    _counter = counter


def current():
    return _counter


def charge():
    if _counter is not None:
        _counter.charge()
