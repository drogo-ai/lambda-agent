"""
Sub-Agent Module
================
Provides a lightweight, disposable agent that the main Lambda agent can spawn
to perform focused tasks in parallel.  Each sub-agent gets its own Gemini chat
session, a restricted set of tools, and a tight iteration budget.

The main agent uses the ``dispatch_subagent`` tool function to fire off work.

Rate-limit handling:
- A global semaphore throttles how many sub-agents can hit the API at once.
- Each API call is wrapped in exponential-backoff retry logic for 429 errors.
- Concurrency is tuneable via env vars SUBAGENT_MAX_WORKERS and
  SUBAGENT_MAX_CONCURRENT_API.
"""

import os
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, Future

from rich.text import Text
from rich.panel import Panel
from rich import box

from . import config

try:
    from google import genai
    from google.genai import types
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Shared console for sub-agent output
# ---------------------------------------------------------------------------
try:
    from .spinner import console
except ImportError:
    from rich.console import Console

    console = Console()

# ---------------------------------------------------------------------------
# Sub-agent tool set (lazy-loaded to avoid circular imports with tools.py)
# ---------------------------------------------------------------------------

# Default tools — the main agent can override per-task
_DEFAULT_TOOL_NAMES = [
    "read_file",
    "search_repo",
    "run_command",
    "write_file",
    "list_directory",
    "get_git_status",
]


def _get_tool_set() -> dict:
    """Lazily import tool functions from tools.py to avoid circular imports."""
    from .tools import (
        read_file,
        search_repo,
        run_command,
        write_file,
        list_directory,
        get_git_status,
    )

    return {
        "read_file": read_file,
        "search_repo": search_repo,
        "run_command": run_command,
        "write_file": write_file,
        "list_directory": list_directory,
        "get_git_status": get_git_status,
    }


# ---------------------------------------------------------------------------
# Thread-safe counter for sub-agent IDs
# ---------------------------------------------------------------------------
_id_lock = threading.Lock()
_next_id = 1


def _get_next_id() -> int:
    global _next_id
    with _id_lock:
        current = _next_id
        _next_id += 1
        return current


# ---------------------------------------------------------------------------
# Rate-limit helpers
# ---------------------------------------------------------------------------

# Max sub-agents that can make an API call at the same time.
# Lower this if you're on a free-tier key with tight RPM limits.
_MAX_CONCURRENT_API = int(os.getenv("SUBAGENT_MAX_CONCURRENT_API", "2"))
_api_semaphore = threading.Semaphore(_MAX_CONCURRENT_API)

# Retry settings for 429 / ResourceExhausted errors
_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 2.0  # seconds — first retry waits ~2s
_RETRY_MAX_DELAY = 60.0  # cap so we don't wait forever


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if *exc* looks like a Gemini rate-limit / quota error."""
    msg = str(exc).lower()
    return any(kw in msg for kw in ("429", "resource_exhausted", "rate limit", "quota"))


def _send_with_retry(chat_session, message, agent_id: int):
    """Send a message through the chat session with semaphore + exp backoff."""
    for attempt in range(1, _MAX_RETRIES + 1):
        with _api_semaphore:
            try:
                return chat_session.send_message(message)
            except Exception as exc:
                if _is_rate_limit_error(exc) and attempt < _MAX_RETRIES:
                    delay = min(
                        _RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 1),
                        _RETRY_MAX_DELAY,
                    )
                    console.print(
                        f"  [dim yellow]⏳ sub-agent #{agent_id}: rate-limited, "
                        f"retry {attempt}/{_MAX_RETRIES} in {delay:.1f}s[/dim yellow]"
                    )
                    time.sleep(delay)
                else:
                    raise
    # Should not reach here, but just in case:
    raise RuntimeError(f"sub-agent #{agent_id}: exhausted {_MAX_RETRIES} retries")


# ---------------------------------------------------------------------------
# SubAgent class
# ---------------------------------------------------------------------------

_SUBAGENT_SYSTEM_INSTRUCTION = """\
You are a Lambda sub-agent — a focused worker spawned by the main Lambda agent \
to complete a specific task.

RULES:
1. Complete the assigned task efficiently.  You have a maximum of {max_iter} tool \
calls before you must produce a final answer.
2. Your final answer MUST be a concise summary of your findings or actions — \
no more than a few sentences.  The main agent will read this summary.
3. You are fully capable of reading, writing, and editing files. Do so if the task demands it, but otherwise avoid unnecessary modifications.
4. Do NOT ask the user questions — you cannot interact with the user.
5. If you hit an error, briefly report what went wrong in your summary.
"""

MAX_SUBAGENT_ITERATIONS = 5
RESULT_MAX_CHARS = 500


class SubAgent:
    """A disposable, lightweight agent that runs a short task and returns a summary."""

    def __init__(
        self,
        task: str,
        context: str = "",
        tool_names: list[str] | None = None,
        model: str | None = None,
    ):
        self.id = _get_next_id()
        self.task = task
        self.context = context
        self.model = model or "gemini-2.0-flash-lite"

        # Resolve tool set (lazy-loaded to avoid circular imports)
        all_tools = _get_tool_set()
        names = tool_names if tool_names else _DEFAULT_TOOL_NAMES
        self.tool_executors: dict = {}
        self.tool_functions: list = []
        for name in names:
            fn = all_tools.get(name)
            if fn:
                self.tool_executors[name] = fn
                self.tool_functions.append(fn)

        # Build Gemini session
        self.client = genai.Client(api_key=config.API_KEY)
        sys_instr = _SUBAGENT_SYSTEM_INSTRUCTION.format(
            max_iter=MAX_SUBAGENT_ITERATIONS
        )

        chat_config = types.GenerateContentConfig(
            system_instruction=sys_instr,
            tools=self.tool_functions if self.tool_functions else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )
        self.chat_session = self.client.chats.create(
            model=self.model, config=chat_config
        )

    def run(self) -> str:
        """Execute the sub-agent task and return a concise result string."""
        # Build the initial prompt
        parts = []
        if self.context:
            parts.append(f"--- CONTEXT ---\n{self.context}\n--- END CONTEXT ---\n\n")
        parts.append(f"Task: {self.task}")
        prompt = "".join(parts)

        try:
            response = _send_with_retry(self.chat_session, prompt, self.id)
        except Exception as e:
            return f"[sub-agent {self.id}] Error on initial message: {e}"

        iterations = 0
        while True:
            iterations += 1
            if iterations > MAX_SUBAGENT_ITERATIONS:
                return self._clip(
                    f"[sub-agent {self.id}] Hit iteration limit. "
                    f"Last response: {getattr(response, 'text', '(none)')}"
                )

            try:
                tool_calls = response.function_calls if response.function_calls else []

                if tool_calls:
                    tool_responses = []
                    for fc in tool_calls:
                        fn_name = fc.name
                        args = fc.args
                        if hasattr(args, "items"):
                            args = {k: v for k, v in args.items()}
                        elif not isinstance(args, dict):
                            args = dict(args) if args else {}

                        if fn_name in self.tool_executors:
                            result = self.tool_executors[fn_name](**args)
                        else:
                            result = (
                                f"Error: Tool '{fn_name}' not available to sub-agent."
                            )

                        tool_responses.append(
                            types.Part.from_function_response(
                                name=fn_name,
                                response={"result": str(result)},
                            )
                        )

                    response = _send_with_retry(
                        self.chat_session, tool_responses, self.id
                    )
                    continue
                else:
                    # Final text response
                    return self._clip(response.text or "(no output)")
            except Exception as e:
                return f"[sub-agent {self.id}] Error during tool loop: {e}"

    def _clip(self, text: str) -> str:
        """Truncate result to RESULT_MAX_CHARS."""
        if len(text) <= RESULT_MAX_CHARS:
            return text
        return text[:RESULT_MAX_CHARS] + f"\n...[TRUNCATED — {len(text)} chars total]"


# ---------------------------------------------------------------------------
# Tool function exposed to the main agent
# ---------------------------------------------------------------------------

# Thread pool for running sub-agents concurrently
_MAX_WORKERS = int(os.getenv("SUBAGENT_MAX_WORKERS", "4"))
_executor = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="subagent")


def dispatch_subagent(task: str, context: str = "", tools: str = "") -> str:
    """Spawns a lightweight sub-agent to perform a focused task and returns its result.

    Use this to delegate independent, parallelizable work such as reading and
    analyzing files, searching the repo, or running investigative commands.
    Multiple dispatch_subagent calls in the *same turn* run in parallel.

    Args:
        task: A clear, specific description of what the sub-agent should do.
              Must be self-contained — the sub-agent has no access to your
              chat history.
        context: Optional context string to give the sub-agent (e.g. file
                 contents, prior findings).  Keep this minimal.
        tools: Optional comma-separated list of tool names the sub-agent can
               use.  Defaults to 'read_file,search_repo,run_command,write_file'.
    """
    # Parse tool list
    tool_names = None
    if tools.strip():
        tool_names = [t.strip() for t in tools.split(",") if t.strip()]

    agent = SubAgent(task=task, context=context, tool_names=tool_names)
    agent_id = agent.id

    # Show dispatch in the terminal
    console.print()
    dispatch_label = Text.assemble(
        (" ⚡ SUB-AGENT ", "bold black on green"),
        (f"  #{agent_id}", "bold green"),
        ("  →  ", "dim"),
        (task[:80] + ("…" if len(task) > 80 else ""), "green"),
    )
    console.print(dispatch_label)

    # Run the sub-agent in a thread (blocks until done — Gemini processes
    # all tool_calls in a batch, so parallel calls happen naturally)
    future: Future = _executor.submit(agent.run)
    result = future.result(timeout=120)  # 2 min hard timeout

    # Show completion
    status_label = Text.assemble(
        (" ✓ SUB-AGENT ", "bold black on green"),
        (f"  #{agent_id} done", "bold green"),
    )
    console.print(status_label)
    console.print(
        Panel(
            result,
            border_style="green",
            box=box.SIMPLE,
            padding=(0, 2),
        )
    )

    return result


# ---------------------------------------------------------------------------
# Registration dicts (imported by tools.py)
# ---------------------------------------------------------------------------

SUBAGENT_EXECUTORS = {
    "dispatch_subagent": dispatch_subagent,
}

SUBAGENT_FUNCTIONS = [
    dispatch_subagent,
]
