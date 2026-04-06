"""
tools.py
────────
Built-in tools that agents can use to interact with the world.

Tools extend what agents can DO beyond just reasoning. Without tools, agents
can only produce text. With tools, they can search the web, run code,
read files, call APIs, and more.

How tools work:
  1. You attach tools to an Agent at creation time
  2. When the agent decides it needs a tool, Claude generates a tool_use block
  3. The Agent's _process_response() method detects this and calls tool.run()
  4. The result is sent back to Claude, which continues its reasoning
  5. This loop repeats until Claude gives a final text answer

Adding a custom tool:
  class MyTool(BaseTool):
      name = "my_tool"
      description = "What this tool does and when to use it"
      input_schema = {
          "type": "object",
          "properties": {
              "param": {"type": "string", "description": "What this param is"}
          },
          "required": ["param"]
      }
      async def run(self, param: str) -> str:
          return your_implementation(param)
"""

import asyncio
import subprocess
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """
    Abstract base class for all tools.

    Subclass this to create custom tools. You must define:
      - name        : Short identifier (snake_case)
      - description : Tells Claude WHEN and HOW to use this tool
      - input_schema: JSON Schema for the tool's parameters
      - run()       : The actual implementation
    """

    name: str = ""
    description: str = ""
    input_schema: dict = {"type": "object", "properties": {}, "required": []}

    @abstractmethod
    async def run(self, **kwargs) -> Any:
        """Execute the tool and return the result."""
        ...

    def to_anthropic_format(self) -> dict:
        """
        Convert this tool to the format expected by the Anthropic API.

        This is called automatically by Agent.execute() — you don't need
        to call this yourself.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class WebSearch(BaseTool):
    """
    Lets agents search the web for current information.

    Uses the Anthropic built-in web search tool. Agents with this tool can
    find recent news, documentation, prices, research papers, and more.

    Usage:
        agent = Agent("Researcher", "...", tools=[WebSearch()])
    """

    name = "web_search"
    description = (
        "Search the internet for current information. Use this when you need "
        "recent news, facts you're unsure about, current prices, documentation, "
        "or any information that may have changed recently."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Be specific for better results."
            }
        },
        "required": ["query"]
    }

    def to_anthropic_format(self) -> dict:
        """WebSearch uses the special Anthropic built-in format."""
        return {"type": "web_search_20250305", "name": "web_search"}

    async def run(self, query: str) -> str:
        # Handled natively by the Anthropic API — this method is not called directly
        return f"Search results for: {query}"


class CodeRunner(BaseTool):
    """
    Lets agents write and execute Python code safely.

    The agent writes Python code, this tool runs it in a subprocess with
    a timeout and returns stdout/stderr. Useful for data analysis, calculations,
    file processing, and anything requiring computation.

    ⚠️  Security note: In production, run this inside a sandbox (Docker, etc.)
    """

    name = "code_runner"
    description = (
        "Write and execute Python code. Use this for calculations, data processing, "
        "file manipulation, generating charts, or any task requiring computation. "
        "Return results via print() statements."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Valid Python code to execute. Use print() for output."
            },
            "timeout": {
                "type": "integer",
                "description": "Max seconds to run (default: 30)",
                "default": 30
            }
        },
        "required": ["code"]
    }

    async def run(self, code: str, timeout: int = 30) -> str:
        """
        Execute Python code in a subprocess and return the output.

        Runs in a separate process to isolate execution from the main app.
        Captures both stdout and stderr.
        """
        logger.debug(f"[CodeRunner] Executing {len(code)} chars of Python")

        try:
            # Run in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["python3", "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
            )

            output = result.stdout.strip()
            errors = result.stderr.strip()

            if result.returncode == 0:
                return output if output else "(Code executed successfully, no output)"
            else:
                return f"Error (exit code {result.returncode}):\n{errors}"

        except subprocess.TimeoutExpired:
            return f"Execution timed out after {timeout} seconds."
        except Exception as e:
            return f"Execution error: {e}"


class FileReader(BaseTool):
    """
    Lets agents read files from the local filesystem.

    Useful for processing documents, CSVs, configs, or any file-based data.
    Reads text files by default; binary files return a size indicator.
    """

    name = "file_reader"
    description = (
        "Read the contents of a local file. Use this when you need to process "
        "documents, CSV data, configuration files, or any file on disk."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file"
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum characters to read (default: 10000)",
                "default": 10000
            }
        },
        "required": ["path"]
    }

    async def run(self, path: str, max_chars: int = 10000) -> str:
        """Read a file and return its contents (truncated if needed)."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(max_chars)
            truncated = " [truncated]" if len(content) == max_chars else ""
            return content + truncated
        except FileNotFoundError:
            return f"File not found: {path}"
        except PermissionError:
            return f"Permission denied: {path}"
        except Exception as e:
            return f"Error reading file: {e}"


class APICall(BaseTool):
    """
    Lets agents call any REST API endpoint.

    Supports GET and POST with optional headers and body.
    Useful for fetching live data, triggering webhooks, or integrating
    with any external service.
    """

    name = "api_call"
    description = (
        "Make an HTTP request to any REST API. Use this to fetch live data, "
        "submit forms, trigger webhooks, or integrate with external services."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The full URL to call (including https://)"
            },
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "DELETE"],
                "description": "HTTP method",
                "default": "GET"
            },
            "headers": {
                "type": "object",
                "description": "Optional HTTP headers as key-value pairs"
            },
            "body": {
                "type": "object",
                "description": "Optional JSON body for POST/PUT requests"
            }
        },
        "required": ["url"]
    }

    async def run(
        self,
        url: str,
        method: str = "GET",
        headers: dict = None,
        body: dict = None
    ) -> str:
        """Make an HTTP request and return the response as a string."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers or {},
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    text = await resp.text()
                    return f"Status: {resp.status}\n{text[:5000]}"
        except ImportError:
            return "aiohttp not installed. Run: pip install aiohttp"
        except Exception as e:
            return f"API call failed: {e}"
