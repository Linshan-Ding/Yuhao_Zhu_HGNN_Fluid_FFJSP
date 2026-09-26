"""Tests persist simulator calls without a cumulative engineering limit."""
from pathlib import Path
import time

def pytest_configure(config):
    import torch
    torch.set_num_threads(1)
    if config.option.basetemp is None:
        parent=Path(__file__).resolve().parents[1]/'.pytest_tmp';parent.mkdir(exist_ok=True)
        config.option.basetemp=str(parent/str(time.time_ns()))

def pytest_sessionstart(session):
    from environment.accounting import install
    from result.engineering import InteractionCounter
    session.counter=InteractionCounter(label='pytest');install(session.counter)

def pytest_sessionfinish(session,exitstatus):
    from environment.accounting import install
    install(None);session.counter.close()
