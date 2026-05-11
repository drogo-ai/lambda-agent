import subprocess
import os

from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich import box
from rich.console import Console

from .scratchpad import SCRATCHPAD_EXECUTORS, SCRATCHPAD_FUNCTIONS
from .todo import TODO_EXECUTORS, TODO_FUNCTIONS
from .subagent import SUBAGENT_EXECUTORS, SUBAGENT_FUNCTIONS

# Use the same console as the rest of the app if available; else create one
try:
    from .spinner import console
except ImportError:
    console = Console()


def read_file(path: str) -> str:
    """Reads the contents of a file.

    Args:
        path: The absolute or relative path to the file.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file {path}: {str(e)}"


def write_file(path: str, content: str) -> str:
    """Writes content to a specific file path.

    Args:
        path: The path to the file to write.
        content: The text content to write to the file.
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing to file {path}: {str(e)}"


def run_command(command: str) -> str:
    """Executes a shell command on the host system.

    Args:
        command: The shell command to execute.
    """
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )
        output = result.stdout + result.stderr
        return output if output else "Command executed successfully with no output."
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except Exception as e:
        return f"Error executing command: {str(e)}"


# ---------------------------------------------------------------------------
# Dedicated directory listing tool
# ---------------------------------------------------------------------------

# Directories to always skip when walking the tree
_IGNORE_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "env",
    ".ruff_cache",
    "dist",
    "build",
    ".next",
    ".cache",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "*.egg-info",
}


def _should_skip_dir(name: str) -> bool:
    """Return True if a directory name matches any ignore pattern."""
    if name in _IGNORE_DIRS:
        return True
    # Handle glob-style suffix patterns like *.egg-info
    for pat in _IGNORE_DIRS:
        if pat.startswith("*") and name.endswith(pat[1:]):
            return True
    return False


def list_directory(path: str = ".", max_depth: int = 3, git_aware: bool = True) -> str:
    """Lists directory contents as a tree structure with smart filtering.

    Returns a compact, indented tree of files and directories.  By default it
    respects .gitignore (via `git ls-files`) so the model never wastes tokens
    on build artefacts or vendored deps.

    Args:
        path: Root directory to list (defaults to current directory '.').
        max_depth: How many levels deep to recurse (defaults to 3).
        git_aware: If True, use git ls-files to respect .gitignore (defaults to True).
    """
    abs_path = os.path.abspath(path)
    if not os.path.isdir(abs_path):
        return f"Error: '{path}' is not a directory."

    # ---- Fast path: use git ls-files when inside a repo ----
    if git_aware:
        try:
            tracked = subprocess.check_output(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard", path],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if tracked:
                lines = tracked.splitlines()
                # Enforce max_depth for git-aware output as well.
                normalized_root = os.path.normpath(path)
                filtered: list[str] = []
                for p in lines:
                    rel = (
                        os.path.relpath(p, normalized_root)
                        if normalized_root not in (".", "")
                        else p
                    )
                    depth = rel.count(os.sep) + 1
                    if depth <= max_depth:
                        filtered.append(p)
                lines = filtered
                if len(lines) > 300:
                    return (
                        "\n".join(lines[:300])
                        + f"\n\n... and {len(lines) - 300} more files."
                    )
                return "\n".join(lines)
        except Exception:
            pass  # Not a git repo or git not available — fall through to manual walk

    # ---- Fallback: manual os.scandir walk ----
    output_lines: list[str] = []

    def _walk(dir_path: str, prefix: str, depth: int):
        if depth > max_depth:
            return
        try:
            entries = sorted(
                os.scandir(dir_path), key=lambda e: (not e.is_dir(), e.name.lower())
            )
        except PermissionError:
            return

        visible_dirs = [
            e
            for e in entries
            if e.is_dir()
            and not _should_skip_dir(e.name)
            and not e.name.startswith(".")
        ]
        files = [e for e in entries if e.is_file()]
        combined = visible_dirs + files

        for i, entry in enumerate(combined):
            is_last = i == len(combined) - 1
            connector = "└── " if is_last else "├── "
            suffix = "/" if entry.is_dir() else ""
            output_lines.append(f"{prefix}{connector}{entry.name}{suffix}")
            if entry.is_dir():
                extension = "    " if is_last else "│   "
                _walk(entry.path, prefix + extension, depth + 1)

    _walk(abs_path, "", 1)
    result = "\n".join(output_lines)
    if len(result) > 4000:
        result = result[:4000] + "\n...[TRUNCATED]"
    return result or "Empty directory."


# ---------------------------------------------------------------------------
# Dedicated git status tool
# ---------------------------------------------------------------------------


def get_git_status(include_diff: bool = False) -> str:
    """Returns a comprehensive git status summary in a single call.

    Bundles branch name, porcelain status, and recent commits so the agent
    does not need multiple run_command calls.

    Args:
        include_diff: If True, also include a condensed diff stat of staged and unstaged changes (defaults to False).
    """
    parts: list[str] = []

    # Branch
    try:
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        parts.append(f"Branch: {branch}")
    except Exception:
        return "Not a git repository."

    # Status (porcelain for compact, stable output)
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v2", "--branch"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        parts.append(f"Status:\n{status}" if status else "Status: Clean working tree")
    except Exception as e:
        parts.append(f"Status error: {e}")

    # Recent commits
    try:
        log = subprocess.check_output(
            ["git", "log", "-n", "5", "--oneline", "--no-decorate"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if log:
            parts.append(f"Recent commits:\n{log}")
    except Exception:
        pass

    # Optional diff stat
    if include_diff:
        try:
            diff = subprocess.check_output(
                ["git", "diff", "--stat", "--no-color"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if diff:
                parts.append(f"Unstaged changes:\n{diff}")
        except Exception:
            pass
        try:
            staged = subprocess.check_output(
                ["git", "diff", "--cached", "--stat", "--no-color"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if staged:
                parts.append(f"Staged changes:\n{staged}")
        except Exception:
            pass

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Workspace summary (used at session start)
# ---------------------------------------------------------------------------


def get_workspace_summary() -> str:
    """Gathers git context, project structure, and documentation to help the agent understand the whole project."""
    summary_parts = []

    # 1. Git context — reuse the dedicated tool
    git_info = get_git_status()
    summary_parts.append(f"### Git Context\n{git_info}")

    # 2. Project structure — reuse the dedicated tool (depth 2 to keep it compact)
    tree = list_directory(".", max_depth=2)
    summary_parts.append(f"### Project Structure\n{tree}")

    # 3. Read important docs
    docs_to_check = [
        "README.md",
        "README",
        ".cursorrules",
        ".agentrules",
        ".agent/scratchpad.md",
        ".agent/todo.md",
        "pyproject.toml",
        "package.json",
    ]
    for doc in docs_to_check:
        if os.path.exists(doc) and os.path.isfile(doc):
            try:
                with open(doc, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Truncate to save tokens if massive
                    if len(content) > 3000:
                        content = content[:3000] + "\n...[TRUNCATED]"
                    summary_parts.append(f"### Document: {doc}\n```\n{content}\n```")
            except Exception:
                pass

    return "\n\n".join(summary_parts)


def search_repo(query: str, path: str = ".") -> str:
    """Searches for a specific string query across all text files in the repository.

    Args:
        query: The substring to search for.
        path: The directory path to search within (defaults to current directory '.').
    """
    try:
        # Use grep for faster searching.
        # -r: recursive, -n: line numbers, -I: ignore binary files
        # -F: fixed strings (prevents regex injection if query has special chars)
        command = [
            "grep",
            "-rnIF",
            "--exclude-dir=.git",
            "--exclude-dir=.venv",
            "--exclude-dir=venv",
            "--exclude-dir=env",
            "--exclude-dir=__pycache__",
            "--exclude-dir=node_modules",
            "--exclude-dir=.ruff_cache",
            "--",
            query,
            path,
        ]

        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode == 0:
            # grep output is already in the format we want (file:line: content)
            # but we strip it to clean it up lightly
            results = result.stdout.strip().split("\n")
            if not results or not results[0]:
                return f"No matches found for '{query}'"

            if len(results) > 100:
                return (
                    "\n".join(results[:100])
                    + f"\n\n... and {len(results) - 100} more matches."
                )
            return "\n".join(results)
        elif result.returncode == 1:
            return f"No matches found for '{query}'"
        else:
            return f"Error searching repository: {result.stderr.strip()}"
    except FileNotFoundError:
        return "Error: 'grep' is not installed or available in PATH."
    except Exception as e:
        return f"Error executing search: {str(e)}"


def ask_user(question: str) -> str:
    """Asks the user a clarifying question and returns their answer.

    Args:
        question: The question to ask the user.
    """
    try:
        console.print()
        console.print(
            Panel(
                Text(question, style="bold white"),
                border_style="yellow",
                box=box.ROUNDED,
                title=Text(" 🤔 Lambda asks ", style="bold black on bright_yellow"),
                title_align="left",
                padding=(0, 2),
            )
        )
        answer = Prompt.ask(
            "[bold bright_yellow]  Your answer[/bold bright_yellow]",
            console=console,
        )
        return answer
    except Exception as e:
        return f"Error asking user: {str(e)}"


def finish_task(message: str) -> str:
    """Explicitly mark a task as fully complete and return the final message to the user.

    Call this tool when you have completed all steps in your todo list and are ready to stop.
    This will immediately exit your execution loop.

    Args:
        message: The final message summarizing what was accomplished to present to the user.
    """
    return message


# A dictionary mapping tool names to Python functions for dynamic execution
TOOL_EXECUTORS = {
    "read_file": read_file,
    "write_file": write_file,
    "run_command": run_command,
    "list_directory": list_directory,
    "get_git_status": get_git_status,
    "search_repo": search_repo,
    "ask_user": ask_user,
    "finish_task": finish_task,
    **SCRATCHPAD_EXECUTORS,
    **TODO_EXECUTORS,
    **SUBAGENT_EXECUTORS,
}

# The list of raw Python functions for the Gemini SDK to auto-generate schemas
TOOL_FUNCTIONS = [
    read_file,
    write_file,
    run_command,
    list_directory,
    get_git_status,
    search_repo,
    ask_user,
    finish_task,
    *SCRATCHPAD_FUNCTIONS,
    *TODO_FUNCTIONS,
    *SUBAGENT_FUNCTIONS,
]
