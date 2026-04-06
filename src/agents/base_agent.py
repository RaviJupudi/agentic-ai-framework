"""
base_agent.py
─────────────
Defines the Agent and AgentResult classes.

Every agent in the framework inherits from Agent. An agent wraps a Claude API
call with a specific role, optional tools, and retry logic. Agents are the
workers — the Orchestrator is the manager.

Design philosophy:
  - Each agent should do ONE thing well (single responsibility)
  - Agents are stateless between calls (safe to run in parallel)
  - Tools extend what agents can DO (web search, code execution, etc.)
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional
from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """
    The output of a single agent execution.

    Attributes:
        agent_name  : Name of the agent that produced this result
        output      : The text output from the agent
        success     : True if the agent completed without errors
        error       : Error message if success=False, else None
        tokens_used : Number of tokens consumed (input + output)
        duration_s  : Wall-clock seconds this agent took
    """
    agent_name: str
    output: str
    success: bool
    tokens_used: int
    error: Optional[str] = None
    duration_s: float = 0.0
    metadata: dict = field(default_factory=dict)


class Agent:
    """
    A single AI worker powered by Claude.

    An Agent has a name, a role (natural language description of what it does),
    optional tools, and optional customization of the underlying model.

    The Orchestrator reads each agent's `role` to decide which agent should
    handle which subtask — so write clear, specific role descriptions.

    Usage:
        agent = Agent(
            name="Researcher",
            role="Search the web and gather accurate, cited information on any topic",
            tools=[WebSearch()],
        )

        result = await agent.execute("Find the top 5 AI papers from 2025")
        print(result.output)
    """

    def __init__(
        self,
        name: str,
        role: str,
        tools: list = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 2000,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
    ):
        """
        Args:
            name        : Short unique identifier. Used by the orchestrator to
                          assign tasks. Keep it descriptive: "Researcher", not "A1".
            role        : Natural-language description of what this agent does.
                          The orchestrator uses this to match tasks to agents.
                          Be specific: "Analyze financial data and produce
                          structured summaries" beats "Do analysis".
            tools       : List of Tool instances this agent can use.
                          Tools let agents interact with the real world —
                          web search, code execution, file I/O, APIs.
            model       : Claude model to use. Defaults to claude-sonnet-4-20250514.
                          Use a faster/cheaper model for simple agents,
                          stronger model for complex reasoning.
            max_tokens  : Maximum tokens in the agent's response.
            system_prompt: Override the default system prompt. If None, a
                           standard prompt is generated from the role.
            temperature : Controls randomness. Lower = more deterministic.
                          Use 0.1–0.3 for factual agents, 0.7–1.0 for creative.
        """
        if not name.strip():
            raise ValueError("Agent name cannot be empty.")
        if not role.strip():
            raise ValueError("Agent role cannot be empty.")

        self.name = name
        self.role = role
        self.tools = tools or []
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Build the system prompt
        self.system_prompt = system_prompt or self._build_system_prompt()

        self._client = AsyncAnthropic()

    def _build_system_prompt(self) -> str:
        """
        Auto-generate a system prompt from the agent's role.

        If you provide a custom system_prompt in __init__, this is skipped.
        """
        tool_desc = ""
        if self.tools:
            tool_names = [t.name for t in self.tools]
            tool_desc = f"\n\nYou have access to the following tools: {', '.join(tool_names)}. Use them when needed."

        return (
            f"You are {self.name}, a specialized AI agent.\n\n"
            f"Your role: {self.role}\n\n"
            f"Instructions:\n"
            f"- Focus exclusively on your role. Do not attempt tasks outside your expertise.\n"
            f"- Be thorough but concise. Quality over quantity.\n"
            f"- If you cannot complete a task, explain clearly why.\n"
            f"- Always structure your output clearly so other agents can build on it."
            f"{tool_desc}"
        )

    async def execute(self, instruction: str) -> AgentResult:
        """
        Execute a single instruction and return the result.

        This is called by the Orchestrator for each subtask. It wraps the
        Claude API call with timing, error handling, and tool support.

        Args:
            instruction: The specific task for this agent to complete.
                         The Orchestrator generates this from the user's goal.

        Returns:
            AgentResult with the agent's output and execution metadata.
        """
        start = time.time()
        logger.debug(f"[{self.name}] Executing: {instruction[:60]}...")

        # Build the API request
        request_params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.system_prompt,
            "messages": [{"role": "user", "content": instruction}],
        }

        # Attach tools if this agent has any
        if self.tools:
            request_params["tools"] = [t.to_anthropic_format() for t in self.tools]

        try:
            response = await self._client.messages.create(**request_params)

            # Handle tool use responses (agent may call tools before final answer)
            output = await self._process_response(response, instruction)

            tokens = response.usage.input_tokens + response.usage.output_tokens
            duration = time.time() - start

            logger.debug(f"[{self.name}] Done in {duration:.1f}s | Tokens: {tokens}")

            return AgentResult(
                agent_name=self.name,
                output=output,
                success=True,
                tokens_used=tokens,
                duration_s=duration,
            )

        except Exception as e:
            duration = time.time() - start
            logger.error(f"[{self.name}] Failed: {e}")
            return AgentResult(
                agent_name=self.name,
                output="",
                success=False,
                error=str(e),
                tokens_used=0,
                duration_s=duration,
            )

    async def _process_response(self, response, original_instruction: str) -> str:
        """
        Process Claude's response, handling tool calls if present.

        If Claude decides to use a tool (e.g., web search), this method:
        1. Extracts the tool call parameters
        2. Runs the tool
        3. Sends the tool result back to Claude
        4. Returns Claude's final text response

        This loop continues until Claude produces a pure text response
        (stop_reason == "end_turn").
        """
        messages = [{"role": "user", "content": original_instruction}]

        while response.stop_reason == "tool_use":
            # Extract text content and tool use blocks
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]

            # Add assistant's response (including tool calls) to history
            messages.append({"role": "assistant", "content": response.content})

            # Execute each tool call
            tool_results = []
            for tool_use in tool_uses:
                tool = self._find_tool(tool_use.name)
                if tool:
                    logger.debug(f"[{self.name}] Using tool: {tool_use.name}")
                    try:
                        tool_output = await tool.run(**tool_use.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": str(tool_output),
                        })
                    except Exception as e:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": f"Tool error: {e}",
                            "is_error": True,
                        })
                else:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": f"Tool '{tool_use.name}' not found.",
                        "is_error": True,
                    })

            # Send tool results back to Claude
            messages.append({"role": "user", "content": tool_results})

            response = await self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                messages=messages,
                tools=[t.to_anthropic_format() for t in self.tools] if self.tools else [],
            )

        # Extract final text response
        text_content = " ".join(
            block.text for block in response.content if hasattr(block, "text")
        )
        return text_content.strip()

    def _find_tool(self, name: str):
        """Find a tool by name from this agent's tool list."""
        return next((t for t in self.tools if t.name == name), None)

    def __repr__(self) -> str:
        tools_str = f", tools=[{', '.join(t.name for t in self.tools)}]" if self.tools else ""
        return f"Agent(name={self.name!r}, model={self.model!r}{tools_str})"
