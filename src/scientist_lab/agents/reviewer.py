"""Freeze Reviewer role surface.

Implementation lives in scientist_lab.core.reviewer (deterministic service).
This module is the Agent-role import path. Not a second Reviewer. No Planner.
v2.5-C optional LLM backend defaults to rules; Gateway is not a fifth Agent.
"""

from scientist_lab.core.reviewer import Reviewer, ReviewPacket, ReviewRefused

__all__ = ["Reviewer", "ReviewPacket", "ReviewRefused"]
