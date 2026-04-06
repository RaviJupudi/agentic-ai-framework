"""
Agentic AI Framework
────────────────────
A production-ready framework for building multi-agent AI systems
with parallel execution, subagent delegation, and intelligent orchestration.

Quick start:
    from agentic import Orchestrator, Agent
    from agentic.tools import WebSearch

    agents = [
        Agent("Researcher", "Search and gather information", tools=[WebSearch()]),
        Agent("Writer", "Write clear, engaging content"),
    ]
    orchestrator = Orchestrator(agents=agents, parallel=True)
    result = await orchestrator.run("Your goal here")
    print(result.output)
"""

from .orchestrator.orchestrator import Orchestrator, OrchestratorResult
from .agents.base_agent import Agent, AgentResult
from .agents.subagent import SubAgent
from . import tools

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"

__all__ = [
    "Orchestrator",
    "OrchestratorResult",
    "Agent",
    "AgentResult",
    "SubAgent",
    "tools",
]
