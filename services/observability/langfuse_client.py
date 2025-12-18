import os

# Try to import observe, else provide dummy
try:
    from langfuse.decorators import observe as _observe
except ImportError:
    try:
        # Fallback for older/newer versions?
        # Or maybe it is just 'from langfuse import observe'? 
        # Actually it's likely just not installed in the env properly or I need 'langfuse' package
        from langfuse import observe as _observe
    except ImportError:
        _observe = None

import logging

# Suppress langfuse auth errors if no key
if not os.getenv("LANGFUSE_PUBLIC_KEY"):
    logging.getLogger("langfuse").setLevel(logging.CRITICAL)

def observe(*args, **kwargs):
    if _observe:
        return _observe(*args, **kwargs)
    # Dummy decorator
    def decorator(func):
        return func
    return decorator

# helper to flush traces if needed (usually handled by SDK background thread)
def flush():
    # langfuse auto-flushes on exit, but we can force it
    pass
