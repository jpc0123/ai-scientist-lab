"""Freeze Reviewer role surface.

Implementation lives in scientist_lab.core.reviewer (deterministic service).
This module is the Agent-role import path. Not a second Reviewer. No Planner.
"""

from scientist_lab.core.reviewer import Reviewer, ReviewPacket, ReviewRefused

__all__ = ["Reviewer", "ReviewPacket", "ReviewRefused"]
