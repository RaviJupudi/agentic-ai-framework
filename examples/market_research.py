"""
market_research.py
──────────────────
Example: Multi-agent competitive analysis pipeline.

This example shows how to build a real-world workflow using parallel agents:
  1. Researcher  — Finds data on both companies using web search
  2. Analyst     — Compares strengths, weaknesses, and market position
  3. Strategist  — Recommends actionable next steps
  4. Writer      — Packages everything into an executive summary

All 4 agents run in parallel (where possible), then results are synthesized.

Run this:
    export ANTHROPIC_API_KEY="your-key"
    python examples/market_research.py
"""

import asyncio
import sys
import os

# Add parent directory to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import Orchestrator, Agent
from src.utils.tools import WebSearch


async def competitive_analysis(company_a: str, company_b: str) -> str:
    """
    Run a competitive analysis between two companies using parallel agents.

    Args:
        company_a: First company name
        company_b: Second company name (the competitor)

    Returns:
        A comprehensive competitive analysis report as a string
    """

    print(f"\n🔍 Starting competitive analysis: {company_a} vs {company_b}")
    print("─" * 50)

    # Define our specialized agents
    # Each has a clear, specific role that the orchestrator uses for task assignment
    agents = [
        Agent(
            name="Researcher",
            role=(
                "Search the web to gather recent, factual information about companies, "
                "including financials, products, market share, and recent news."
            ),
            tools=[WebSearch()],
            max_tokens=2000,
        ),
        Agent(
            name="Analyst",
            role=(
                "Analyze business data to identify strengths, weaknesses, opportunities, "
                "and threats (SWOT). Compare companies objectively using data."
            ),
            max_tokens=2000,
        ),
        Agent(
            name="Strategist",
            role=(
                "Identify strategic opportunities and risks from competitive analysis. "
                "Recommend specific, actionable next steps for business decision-making."
            ),
            max_tokens=1500,
        ),
        Agent(
            name="Writer",
            role=(
                "Transform analysis and strategy into a clear, professional executive summary "
                "that is easy to read and actionable for senior leadership."
            ),
            max_tokens=2000,
        ),
    ]

    # Create the orchestrator with parallel execution enabled
    orchestrator = Orchestrator(
        agents=agents,
        parallel=True,       # All agents fire simultaneously
        max_retries=2,       # Retry failed agents up to 2 times
        timeout=90,          # 90 second timeout per agent
        verbose=True,        # Log progress to console
    )

    # Run the full workflow
    result = await orchestrator.run(
        goal=(
            f"Produce a comprehensive competitive analysis of {company_a} versus {company_b}. "
            f"Cover: market position, key products/services, recent performance, "
            f"strengths and weaknesses of each, and strategic recommendations."
        )
    )

    # Print execution summary
    print(f"\n✅ Completed in {result.duration_seconds:.1f}s")
    print(f"📊 Total tokens used: {result.total_tokens:,}")
    print(f"🤖 Agents completed: {len(result.agent_outputs)}")

    if result.errors:
        print(f"⚠️  Errors encountered: {result.errors}")

    print("\n" + "─" * 50)
    print("📄 FINAL REPORT")
    print("─" * 50)

    return result.output


# ─── Individual agent outputs (for debugging) ───────────────────────────────

async def show_agent_outputs(company_a: str, company_b: str):
    """
    Same workflow but prints each agent's individual output.
    Useful for debugging or when you need the raw agent data.
    """
    agents = [
        Agent("Researcher", "Gather factual data on companies", tools=[WebSearch()]),
        Agent("Analyst", "Perform SWOT and competitive comparison"),
    ]

    orchestrator = Orchestrator(agents=agents, parallel=True, verbose=False)

    result = await orchestrator.run(
        f"Research and analyze {company_a} vs {company_b} in the AI industry"
    )

    print("\n🔬 Individual Agent Outputs:")
    for agent_result in result.agent_outputs:
        status = "✅" if agent_result.success else "❌"
        print(f"\n{status} [{agent_result.agent_name}] ({agent_result.duration_s:.1f}s)")
        print(agent_result.output[:500] + "..." if len(agent_result.output) > 500 else agent_result.output)


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # You can change these to any two companies
    COMPANY_A = "OpenAI"
    COMPANY_B = "Anthropic"

    async def main():
        report = await competitive_analysis(COMPANY_A, COMPANY_B)
        print(report)

        # Optionally show individual agent outputs
        # await show_agent_outputs(COMPANY_A, COMPANY_B)

    asyncio.run(main())
