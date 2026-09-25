"""Process-local engineering interaction cap, including tests and evaluation."""
_counter = None


def install(counter):
    global _counter
    _counter = counter


def charge():
    if _counter is not None:
        _counter.charge()
