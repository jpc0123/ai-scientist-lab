"""M2 Instrumentation Contract — record execution facts only.

Does not control Planner/Gate/Reviewer. M3 adds model/human events.
"""

from scientist_lab.instrumentation.appender import EventAppender

__all__ = ["EventAppender"]
