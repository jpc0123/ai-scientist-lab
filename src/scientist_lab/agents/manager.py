"""Freeze Manager role surface.

Implementation lives in scientist_lab.core.manager (state-machine dispatch).
This module is the Agent-role import path. Not a fourth scientific brain.
Does not use legacy research_loop / search state machines.
"""

from scientist_lab.core.manager import Manager, ManagerStep

__all__ = ["Manager", "ManagerStep"]
