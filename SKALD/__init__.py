# SKALD package exports
#
# Avoid importing `core` at module import time so `python -m SKALD.core`
# does not trip runpy's "found in sys.modules before execution" warning.
def run_pipeline(*args, **kwargs):
    from .core import run_pipeline as _run_pipeline
    return _run_pipeline(*args, **kwargs)
