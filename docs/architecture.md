# Architecture Deep Dive

This document explains the internal architecture of the Agentic AI Framework — how the pieces fit together and why they were designed this way.

## Overview

```
User Goal (string)
       │
       ▼
  Orchestrator
       │
       ├──► TaskPlanner  ──► Subtask list (one per agent)
       │
       ├──► asyncio.gather()  ──► [Agent A, Agent B, Agent C] (parallel)
       │              │
       │         Each Agent:
       │           1. Calls Claude API
       │           2. Handles tool_use loops
       │           3. Returns AgentResult
       │
       └──► ResultSynthesizer ──► Final merged output
```

## Component Responsibilities

### Orchestrator (`src/orchestrator/orchestrator.py`)
The top-level coordinator. Owns the full lifecycle of a workflow:
- Receives the user goal
- Delegates planning to TaskPlanner
- Routes subtasks to agents
- Controls parallel vs sequential execution
- Handles retries and timeouts
- Delegates synthesis to ResultSynthesizer

**Key design decision:** The Orchestrator itself never calls Claude directly for business logic — it only coordinates. This keeps it testable and replaceable.

### TaskPlanner (`src/orchestrator/task_planner.py`)
Uses Claude to decompose goals into subtasks. Outputs a list of `SubTask` objects, each specifying:
- Which agent should handle it
- What specific instruction to give
- Priority order (for sequential mode)
- Dependencies on other agents

**Key design decision:** Using Claude for planning (instead of hardcoded routing) means the framework can handle novel goals it was never explicitly programmed for.

### Agent (`src/agents/base_agent.py`)
A single worker that wraps a Claude API call. Each agent:
- Has a name and role (used by TaskPlanner for assignment)
- Optionally has tools (web search, code runner, etc.)
- Handles tool_use loops automatically
- Returns a structured `AgentResult`

**Key design decision:** Agents are stateless between calls — they hold no memory of previous executions. This makes them safe to run in parallel without race conditions.

### ResultSynthesizer (`src/orchestrator/result_synthesizer.py`)
Merges N agent outputs into one coherent response. Asks Claude to:
- Eliminate redundancy
- Resolve contradictions
- Structure information logically
- Write in a consistent voice

**Key design decision:** Synthesis is a separate step rather than asking agents to synthesize each other's outputs. This creates cleaner separation of concerns.

### Tools (`src/utils/tools.py`)
Real-world capabilities attached to agents. All tools implement `BaseTool` with:
- `name` — identifier Claude uses to call the tool
- `description` — tells Claude when/how to use it
- `input_schema` — JSON Schema for parameters
- `run()` — the actual implementation

## Parallel Execution Model

```python
# Pseudocode of what asyncio.gather() does internally
results = await asyncio.gather(
    agent_a.execute(task_a),   # Starts immediately
    agent_b.execute(task_b),   # Starts immediately
    agent_c.execute(task_c),   # Starts immediately
)
# All three run concurrently — total time = max(a, b, c), not a+b+c
```

This is the core performance advantage. In a 4-agent workflow where each agent takes 15 seconds:
- Sequential: 60 seconds total
- Parallel: ~15 seconds total (4× speedup)

## Error Handling Strategy

The framework uses a "fail gracefully" approach:

1. **Agent timeout** → Return an error AgentResult, continue with other agents
2. **Agent exception** → Retry with exponential backoff, then return error
3. **Planning failure** → Fallback to assigning each agent its own role
4. **Synthesis with partial results** → Synthesize whatever succeeded

This means a single failing agent doesn't break the entire workflow.

## Extending the Framework

### Custom Agent
```python
class DomainExpertAgent(Agent):
    def __init__(self):
        super().__init__(
            name="DomainExpert",
            role="Your specific expertise here",
            system_prompt="Custom instructions...",
        )
```

### Custom Tool
```python
class DatabaseTool(BaseTool):
    name = "database_query"
    description = "Query a PostgreSQL database"
    input_schema = {
        "type": "object",
        "properties": {"sql": {"type": "string"}},
        "required": ["sql"]
    }
    async def run(self, sql: str) -> str:
        # Your DB implementation
        ...
```

### Custom Orchestration Strategy
```python
class PriorityOrchestrator(Orchestrator):
    async def _run_parallel(self, subtasks):
        # Custom logic: run high-priority agents first
        high = [t for t in subtasks if t.priority == 0]
        rest = [t for t in subtasks if t.priority > 0]
        results = await super()._run_parallel(high)
        # Inject high-priority results into remaining tasks
        ...
```
