"""Anchor every entry point at the repository root and apply the configured per-process thread count."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
os.chdir(ROOT)
import yaml  # noqa: E402
THREADS=str(yaml.safe_load((ROOT/'configs/experiment.yaml').read_text(encoding='utf-8'))['runtime']['threads'])
os.environ['OMP_NUM_THREADS']=THREADS
os.environ['MKL_NUM_THREADS']=THREADS
