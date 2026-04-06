"""
subagent.py
───────────
SubAgents are micro-specialists that can be spawned by regular agents.

While a regular Agent handles a broad subtask (e.g., "Research AI trends"),
a SubAgent handles an extremely narrow, precise task (e.g., "Verify that
this specific statistic is correctly cited").

SubAgents are useful for:
  - Quality assurance (fact-checking, grammar, formatting)
  - Data validation (verify figures, check logic)
  - Specialization within specialization
  - Breaking very long tasks into smaller chunks

A SubAgent works exactly like a regular Agent — it's just scoped to be
spawned by a specific parent agent rather than the orchestrator directly.
"""

from .base_agent import Agent


class SubAgent(Agent):
    """
    A specialized micro-agent designed to be spawned by a parent agent.

    SubAgents are identical to Agents in behavior but carry metadata about
    which parent agent is allowed to use them. This is enforced at the
    orchestrator level to maintain clean delegation hierarchies.

    Usage:
        # Define a subagent for fact-checking
        fact_checker = SubAgent(
            name="FactChecker",
            role="Verify that every factual claim is accurate and properly cited",
            parent_agent="Researcher",  # Only Researcher can spawn this
        )

        # Attach to the parent agent
        researcher = Agent(
            name="Researcher",
            role="Find and summarize information",
            tools=[WebSearch()],
            subagents=[fact_checker]   # SubAgents attached here
        )
    """

    def __init__(self, name: str, role: str, parent_agent: str, **kwargs):
        """
        Args:
            name        : Unique name for this subagent
            role        : What this subagent specializes in (be very specific)
            parent_agent: Name of the Agent that can spawn this subagent
            **kwargs    : All other Agent arguments (tools, model, etc.)
        """
        super().__init__(name=name, role=role, **kwargs)
        self.parent_agent = parent_agent

    def __repr__(self) -> str:
        return f"SubAgent(name={self.name!r}, parent={self.parent_agent!r})"
