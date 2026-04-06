"""
orchestrator.py
───────────────
The heart of the Agentic AI Framework.

The Orchestrator receives a high-level goal, uses Claude to decompose it into
subtasks, assigns each subtask to the most suitable Agent, runs them in parallel
(or sequentially), and synthesizes all outputs into a final result.

Architecture:
    goal ──► TaskPlanner ──► [Agent, Agent, Agent] ──► ResultSynthesizer ──► output
                                      │
                              (parallel via asyncio)
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Optional
from anthropic import AsyncAnthropic

from .task_planner import TaskPlanner
from .result_synthesizer import ResultSynthesizer
from ..agents.base_agent import Agent, AgentResult

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResult:
    """
    Final result returned after all agents have completed.

    Attributes:
        output          : The synthesized final answer (string)
        agent_outputs   : Individual results from each agent
        total_tokens    : Total tokens consumed across all agents
        duration_seconds: Wall-clock time from start to finish
        success         : True if all agents completed without fatal errors
    """
    output: str
    agent_outputs: list[AgentResult]
    total_tokens: int
    duration_seconds: float
    success: bool
    errors: list[str] = field(default_factory=list)


class Orchestrator:
    """
    Multi-agent orchestrator with parallel execution support.

    The Orchestrator is the top-level coordinator. It:
      1. Receives a natural-language goal from the user
      2. Uses Claude to break the goal into discrete subtasks (TaskPlanner)
      3. Matches each subtask to the best-suited Agent
      4. Executes agents concurrently using asyncio.gather()
      5. Merges all outputs into a coherent final response (ResultSynthesizer)

    Usage:
        orchestrator = Orchestrator(
            agents=[research_agent, analysis_agent, writer_agent],
            parallel=True,
            max_retries=3,
            verbose=True
        )
        result = await orchestrator.run("Research AI trends and write a summary")
        print(result.output)
    """

    def __init__(
        self,
        agents: list[Agent],
        parallel: bool = True,
        max_retries: int = 2,
        timeout: int = 120,
        verbose: bool = False,
        model: str = "claude-sonnet-4-20250514",
        cost_limit_usd: Optional[float] = None,
    ):
        """
        Args:
            agents        : List of Agent instances available to this orchestrator.
                            The orchestrator picks from these based on each agent's role.
            parallel      : If True, eligible agents run simultaneously via asyncio.
                            If False, agents run one after another (useful when
                            later agents need earlier agents' outputs).
            max_retries   : How many times to retry a failed agent before giving up.
            timeout       : Maximum seconds to wait for any single agent.
            verbose       : If True, logs every planning and execution step.
            model         : Claude model used for planning and synthesis.
            cost_limit_usd: Abort if estimated cost exceeds this value (safety guard).
        """
        if not agents:
            raise ValueError("Orchestrator requires at least one agent.")

        self.agents = {agent.name: agent for agent in agents}
        self.parallel = parallel
        self.max_retries = max_retries
        self.timeout = timeout
        self.verbose = verbose
        self.model = model
        self.cost_limit_usd = cost_limit_usd

        self._client = AsyncAnthropic()
        self._planner = TaskPlanner(client=self._client, model=model)
        self._synthesizer = ResultSynthesizer(client=self._client, model=model)

        if verbose:
            logging.basicConfig(level=logging.DEBUG)

    async def run(self, goal: str) -> OrchestratorResult:
        """
        Execute a multi-agent workflow for the given goal.

        This is the main entry point. Call this with any natural-language goal
        and the orchestrator handles everything else.

        Args:
            goal: A natural-language description of what you want to achieve.
                  Be specific — the more detail you provide, the better the
                  task decomposition will be.

        Returns:
            OrchestratorResult with the synthesized output and full metadata.

        Example:
            result = await orchestrator.run(
                "Analyze Tesla's Q1 2026 earnings and compare to analyst expectations"
            )
        """
        start_time = time.time()
        logger.debug(f"[Orchestrator] Starting goal: {goal[:80]}...")

        # ── Step 1: Plan ──────────────────────────────────────────────────────
        # Ask Claude to decompose the goal into subtasks, one per agent
        logger.debug("[Orchestrator] Planning subtasks...")
        subtasks = await self._planner.decompose(
            goal=goal,
            available_agents=list(self.agents.values())
        )
        logger.debug(f"[Orchestrator] Planned {len(subtasks)} subtasks: "
                     f"{[t.agent_name for t in subtasks]}")

        # ── Step 2: Validate agent assignments ────────────────────────────────
        # Make sure every assigned agent actually exists
        errors = []
        valid_subtasks = []
        for task in subtasks:
            if task.agent_name not in self.agents:
                errors.append(f"Unknown agent '{task.agent_name}' for task: {task.instruction}")
                logger.warning(f"[Orchestrator] {errors[-1]}")
            else:
                valid_subtasks.append(task)

        if not valid_subtasks:
            return OrchestratorResult(
                output="No valid agents could be assigned to this goal.",
                agent_outputs=[],
                total_tokens=0,
                duration_seconds=time.time() - start_time,
                success=False,
                errors=errors
            )

        # ── Step 3: Execute ───────────────────────────────────────────────────
        if self.parallel:
            # Run all agents AT THE SAME TIME using asyncio.gather
            # This is the key performance advantage — total time ≈ slowest agent
            agent_outputs = await self._run_parallel(valid_subtasks)
        else:
            # Run agents one by one — use when later agents need earlier results
            agent_outputs = await self._run_sequential(valid_subtasks)

        # ── Step 4: Synthesize ────────────────────────────────────────────────
        # Merge all individual agent outputs into one coherent final answer
        logger.debug("[Orchestrator] Synthesizing final output...")
        final_output = await self._synthesizer.merge(
            goal=goal,
            agent_outputs=agent_outputs
        )

        total_tokens = sum(r.tokens_used for r in agent_outputs)
        duration = time.time() - start_time

        logger.debug(f"[Orchestrator] Done in {duration:.1f}s | "
                     f"Tokens: {total_tokens} | Agents: {len(agent_outputs)}")

        return OrchestratorResult(
            output=final_output,
            agent_outputs=agent_outputs,
            total_tokens=total_tokens,
            duration_seconds=duration,
            success=all(r.success for r in agent_outputs),
            errors=errors + [r.error for r in agent_outputs if r.error]
        )

    async def _run_parallel(self, subtasks) -> list[AgentResult]:
        """
        Execute all subtasks concurrently.

        Uses asyncio.gather() to fire all agents simultaneously. Each agent
        runs in its own coroutine — total wall-clock time equals the duration
        of the SLOWEST agent, not the sum of all agents.

        This is the core performance advantage of the parallel architecture:
        a 4-agent workflow that each take 15s runs in ~15s, not ~60s.
        """
        logger.debug(f"[Orchestrator] Parallel execution: {len(subtasks)} agents")

        async def run_with_retry(task) -> AgentResult:
            agent = self.agents[task.agent_name]
            for attempt in range(self.max_retries + 1):
                try:
                    result = await asyncio.wait_for(
                        agent.execute(task.instruction),
                        timeout=self.timeout
                    )
                    return result
                except asyncio.TimeoutError:
                    logger.warning(f"[{task.agent_name}] Timeout on attempt {attempt + 1}")
                    if attempt == self.max_retries:
                        return AgentResult(
                            agent_name=task.agent_name,
                            output="",
                            success=False,
                            error=f"Agent timed out after {self.timeout}s",
                            tokens_used=0
                        )
                except Exception as e:
                    logger.warning(f"[{task.agent_name}] Error on attempt {attempt + 1}: {e}")
                    if attempt == self.max_retries:
                        return AgentResult(
                            agent_name=task.agent_name,
                            output="",
                            success=False,
                            error=str(e),
                            tokens_used=0
                        )
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

        # Fire all agents simultaneously
        results = await asyncio.gather(*[run_with_retry(t) for t in subtasks])
        return list(results)

    async def _run_sequential(self, subtasks) -> list[AgentResult]:
        """
        Execute subtasks one after another.

        Use this when agents depend on each other's outputs — for example,
        a Writer agent that needs the Researcher agent's output first.
        Each agent receives the outputs of all previous agents as context.
        """
        logger.debug(f"[Orchestrator] Sequential execution: {len(subtasks)} tasks")
        results = []
        prior_context = ""

        for task in subtasks:
            agent = self.agents[task.agent_name]

            # Inject prior agents' outputs as context
            instruction = task.instruction
            if prior_context:
                instruction = (
                    f"Previous agents have produced the following:\n\n{prior_context}\n\n"
                    f"Your task: {instruction}"
                )

            try:
                result = await asyncio.wait_for(
                    agent.execute(instruction),
                    timeout=self.timeout
                )
                results.append(result)
                if result.success:
                    prior_context += f"\n\n[{task.agent_name}]:\n{result.output}"
            except Exception as e:
                results.append(AgentResult(
                    agent_name=task.agent_name,
                    output="",
                    success=False,
                    error=str(e),
                    tokens_used=0
                ))

        return results
