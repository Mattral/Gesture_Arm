# gesture_arm.hardware
# Lazy imports: pyfirmata is only required at runtime, not at import time.

def __getattr__(name):
    if name in ("ArmController", "BaseController", "connect", "board_session"):
        from .arduino import ArmController, BaseController, connect, board_session
        return locals()[name]
    raise AttributeError(f"module 'gesture_arm.hardware' has no attribute {name!r}")
