"""Cooperative cancellation for in-flight try-on jobs."""


class GenerationCancelled(RuntimeError):
    """Raised when the user stops an in-flight try-on."""
