"""COLA framework package.

This package is organized so each major training component can be developed,
tested, and replaced independently.
"""

from cola_framework.envs.mpe_wrapper import MPEWrapper
from cola_framework.interfaces.environment import MultiAgentEnvironment

__all__ = ["MPEWrapper", "MultiAgentEnvironment"]
