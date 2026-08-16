"""Freeze Planner role surface.

Implementation lives in scientist_lab.core.planner (rules-first service).
This module is the Agent-role import path. Not a second Planner kind.
Legacy MockPlanner is isolated in scientist_lab.agents.legacy_planner.
"""

from scientist_lab.core.planner import (
    PlanPacket,
    Planner,
    PlanRefused,
    propose_and_gate_next,
)

__all__ = ["Planner", "PlanPacket", "PlanRefused", "propose_and_gate_next"]
