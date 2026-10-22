"""Zero-argument pre-registered pilot entry (docs/preregistration-v2.md); optional positive integer concurrency."""
import _bootstrap
from scripts.pipeline import entry

if __name__ == "__main__":
    entry('all', study='pilot')
