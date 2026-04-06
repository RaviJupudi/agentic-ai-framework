"""
result_synthesizer.py
─────────────────────
Merges outputs from multiple agents into one coherent final response.

After all agents complete, you have N separate text outputs — one per agent.
The ResultSynthesizer sends all of these to Claude with the original goal
and asks it to produce a unified, polished final answer.

This is important because:
  - Individual agents may have overlapping information
  - Outputs need to be structured coherently for the end user
  - Contradictions between agents need to be resolved
  - The final response should read as one document, not N fragments
"""

import logging
from anthropic import AsyncAnthropic
from ..agents.base_agent import AgentResult

logger = logging.getLogger(__name__)


class ResultSynthesizer:
    """
    Merges multiple agent outputs into a single, coherent final answer.

    Called by the Orchestrator after all agents have completed. It sends
    all agent outputs plus the original goal to Claude, which produces
    a unified response.
    """

    SYNTHESIS_PROMPT = """You are a result synthesizer for a multi-agent AI system.

Multiple specialized agents have worked in parallel to achieve the following goal:

GOAL: {goal}

Here are the outputs from each agent:

{agent_outputs}

Your task:
1. Synthesize all agent outputs into ONE coherent, well-structured response
2. Eliminate redundancy — don't repeat the same information twice
3. Resolve any contradictions by using the most reliable/recent information
4. Structure the output logically for the end user
5. Preserve all important insights from every agent
6. Write in a clear, professional tone

Produce the final synthesized response now:"""

    def __init__(self, client: AsyncAnthropic, model: str):
        self._client = client
        self.model = model

    async def merge(self, goal: str, agent_outputs: list[AgentResult]) -> str:
        """
        Synthesize all agent outputs into one final response.

        Args:
            goal          : The original user goal
            agent_outputs : List of AgentResult from each agent

        Returns:
            A single, coherent string combining all agent insights
        """
        # Filter to successful outputs only
        successful = [r for r in agent_outputs if r.success and r.output.strip()]

        if not successful:
            failed = [r.agent_name for r in agent_outputs if not r.success]
            return f"All agents failed to complete their tasks. Failed agents: {', '.join(failed)}"

        # If only one agent succeeded, return its output directly
        if len(successful) == 1:
            return successful[0].output

        # Format agent outputs for the synthesis prompt
        formatted_outputs = "\n\n".join(
            f"── {result.agent_name} ──\n{result.output}"
            for result in successful
        )

        prompt = self.SYNTHESIS_PROMPT.format(
            goal=goal,
            agent_outputs=formatted_outputs
        )

        logger.debug(f"[ResultSynthesizer] Synthesizing {len(successful)} agent outputs...")

        response = await self._client.messages.create(
            model=self.model,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text.strip()
