# gesture_arm.vision
# Lazy imports: cvzone is only required at runtime, not at import time.
# This allows the package to be imported in CI without hardware dependencies.

def __getattr__(name):
    if name in ("HandTracker", "HandState", "TrackerOutput"):
        from .tracker import HandTracker, HandState, TrackerOutput
        return locals()[name]
    raise AttributeError(f"module 'gesture_arm.vision' has no attribute {name!r}")
