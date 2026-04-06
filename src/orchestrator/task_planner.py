"""
task_planner.py
───────────────
Breaks a high-level user goal into discrete subtasks, one per agent.

The TaskPlanner asks Claude to analyze the goal and the available agents,
then produces a structured list of assignments: which agent should do what,
and in what order (for sequential mode).

This is what makes the Orchestrator "intelligent" — it doesn't use hardcoded
routing rules. It uses Claude itself to figure out the best delegation strategy.
"""

import json
import logging
from dataclasses import dataclass
from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)


@dataclass
class SubTask:
    """
    A single unit of work assigned to a specific agent.

    Attributes:
        agent_name  : Must match an Agent's name in the Orchestrator's registry
        instruction : The specific instruction for that agent
        priority    : Lower number = runs first in sequential mode
        depends_on  : List of agent names whose output this task needs
    """
    agent_name: str
    instruction: str
    priority: int = 0
    depends_on: list[str] = None

    def __post_init__(self):
        self.depends_on = self.depends_on or []


class TaskPlanner:
    """
    Uses Claude to decompose a goal into per-agent subtasks.

    The planner receives:
    - The user's goal (natural language)
    - A list of available agents with their names and roles

    It returns a list of SubTask objects, each mapped to a specific agent.

    This approach is more flexible than hardcoded routing because it can
    handle novel goals that the framework designers never anticipated.
    """

    PLANNING_PROMPT = """You are a task planning system for a multi-agent AI framework.

Your job: Given a goal and a list of available agents, decompose the goal into
specific subtasks — one per agent — that together achieve the goal.

Available agents:
{agent_descriptions}

Goal: {goal}

Rules:
1. Assign at most ONE subtask per agent
2. Only use agents from the list above — use their EXACT names
3. Write specific, actionable instructions for each agent
4. If agents depend on each other's output, note it in depends_on
5. Assign priority: 0 = first, 1 = second, etc. (for sequential execution)

Respond ONLY with valid JSON in this exact format:
{{
  "subtasks": [
    {{
      "agent_name": "ExactAgentName",
      "instruction": "Specific instruction for this agent",
      "priority": 0,
      "depends_on": []
    }}
  ]
}}

No explanation. No markdown. Pure JSON only."""

    def __init__(self, client: AsyncAnthropic, model: str):
        self._client = client
        self.model = model

    async def decompose(self, goal: str, available_agents) -> list[SubTask]:
        """
        Decompose a goal into subtasks using Claude.

        Args:
            goal            : The user's high-level goal
            available_agents: List of Agent instances

        Returns:
            List of SubTask objects, sorted by priority
        """
        # Build agent descriptions for the prompt
        agent_descriptions = "\n".join(
            f"- {agent.name}: {agent.role}"
            for agent in available_agents
        )

        prompt = self.PLANNING_PROMPT.format(
            agent_descriptions=agent_descriptions,
            goal=goal
        )

        logger.debug("[TaskPlanner] Requesting task decomposition from Claude...")

        response = await self._client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.content[0].text.strip()
        logger.debug(f"[TaskPlanner] Raw plan: {raw[:200]}...")

        # Parse and validate the JSON response
        subtasks = self._parse_plan(raw, available_agents)

        # Sort by priority for sequential execution
        subtasks.sort(key=lambda t: t.priority)

        logger.debug(f"[TaskPlanner] Created {len(subtasks)} subtasks")
        return subtasks

    def _parse_plan(self, raw_json: str, available_agents) -> list[SubTask]:
        """
        Parse Claude's JSON response into SubTask objects.

        Falls back gracefully if the JSON is malformed — assigns all agents
        their role as the instruction rather than crashing.
        """
        valid_names = {agent.name for agent in available_agents}

        try:
            # Strip any accidental markdown code fences
            clean = raw_json.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean)

            subtasks = []
            for i, item in enumerate(data.get("subtasks", [])):
                name = item.get("agent_name", "")
                if name not in valid_names:
                    logger.warning(f"[TaskPlanner] Unknown agent '{name}' — skipping")
                    continue

                subtasks.append(SubTask(
                    agent_name=name,
                    instruction=item.get("instruction", "Complete your assigned task."),
                    priority=item.get("priority", i),
                    depends_on=item.get("depends_on", [])
                ))

            if subtasks:
                return subtasks

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"[TaskPlanner] JSON parse failed: {e}. Using fallback.")

        # Fallback: assign each agent their own role as the instruction
        return [
            SubTask(
                agent_name=agent.name,
                instruction=f"{agent.role}. Goal context: {agent.role}",
                priority=i
            )
            for i, agent in enumerate(available_agents)
        ]
