from .agent import Agent, TokenUsage
from . import config
from .spinner import console
import os
import getpass
from pathlib import Path

from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich import box
from rich.align import Align
from rich.table import Table


BANNER = r"""
██╗      █████╗ ███╗   ███╗██████╗ ██████╗  █████╗
██║     ██╔══██╗████╗ ████║██╔══██╗██╔══██╗██╔══██╗
██║     ███████║██╔████╔██║██████╔╝██║  ██║███████║
██║     ██╔══██║██║╚██╔╝██║██╔══██╗██║  ██║██╔══██║
███████╗██║  ██║██║ ╚═╝ ██║██████╔╝██████╔╝██║  ██║
╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═════╝ ╚═════╝ ╚═╝  ╚═╝
"""


SLASH_COMMANDS = {
    "/models": "List available models and switch between them",
    "/config": "Update API key and save to config",
    "/help": "Show available slash commands",
}


def print_banner():
    banner_text = Text(BANNER, style="bold cyan", justify="center")
    subtitle = Text(
        "  Minimal AI Coding Agent  ·  Type '/help' for commands  ",
        style="dim white",
        justify="center",
    )

    panel = Panel(
        Align.center(Text.assemble(banner_text, "\n", subtitle)),
        border_style="cyan",
        box=box.DOUBLE_EDGE,
        padding=(0, 2),
    )
    console.print(panel)


def print_user_message(text: str):
    label = Text(" YOU ", style="bold black on bright_yellow")
    content = Text(f"  {text}", style="bright_white")
    console.print()
    console.print(Text.assemble(label, content))


def print_lambda_message(text: str):
    console.print()
    label = Text(" LAMBDA ", style="bold black on cyan")
    console.print(label)
    console.print(
        Panel(
            Markdown(text),
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 2),
        )
    )


def print_token_stats(turn: TokenUsage, session: TokenUsage):
    """Render a compact token usage line under the Lambda response."""
    console.print(
        Text.assemble(
            ("  ▶ tokens  ", "dim"),
            ("this turn: ", "dim"),
            (f"↑{turn.prompt:,}", "dim cyan"),
            (" in  ", "dim"),
            (f"↓{turn.completion:,}", "dim cyan"),
            (" out     ", "dim"),
            ("session total: ", "dim"),
            (f"{session.total:,}", "bold cyan"),
            (" tokens", "dim"),
        )
    )


def handle_models_command(agent: Agent):
    """Display available models and let the user pick one."""
    table = Table(
        title="Available Models",
        title_style="bold cyan",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(0, 2),
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Model", style="white")
    table.add_column("Status", justify="center")

    models = config.AVAILABLE_MODELS
    for i, model in enumerate(models, 1):
        is_active = model == agent.model_name
        status = "[bold green]● active[/bold green]" if is_active else "[dim]—[/dim]"
        name_style = "bold cyan" if is_active else "white"
        table.add_row(str(i), f"[{name_style}]{model}[/{name_style}]", status)

    console.print()
    console.print(table)
    console.print()

    choice = Prompt.ask(
        "[bold bright_yellow]  Select model #[/bold bright_yellow] (or Enter to cancel)",
        default="",
        console=console,
    )

    if not choice.strip():
        console.print("  [dim]No change.[/dim]")
        return

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            selected = models[idx]
            if selected == agent.model_name:
                console.print(f"  [dim]Already using[/dim] [cyan]{selected}[/cyan]")
            else:
                msg = agent.switch_model(selected)
                console.print(f"  {msg}")
        else:
            console.print("  [red]Invalid selection.[/red]")
    except ValueError:
        console.print("  [red]Please enter a number.[/red]")


def handle_help_command():
    """Show available slash commands."""
    table = Table(
        title="Slash Commands",
        title_style="bold cyan",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(0, 1),
    )
    table.add_column("Command", style="bold bright_yellow", min_width=12)
    table.add_column("Description", style="white")

    for cmd, desc in SLASH_COMMANDS.items():
        table.add_row(cmd, desc)

    # Also list the built-in exit commands
    table.add_row("exit / quit", "End the session")

    console.print()
    console.print(table)


def handle_config_command(agent: Agent):
    """Let the user update their API key mid-session."""
    console.print()
    console.print(
        Panel(
            Text.assemble(
                ("Current API key: ", "dim"),
                (f"{config.API_KEY[:8]}...{config.API_KEY[-4:]}", "cyan"),
            ),
            border_style="cyan",
            box=box.ROUNDED,
            title="[bold cyan]⚙ Configuration[/bold cyan]",
            title_align="left",
        )
    )
    console.print()

    new_key = getpass.getpass("  Enter new API key (or press Enter to keep current): ")

    if not new_key.strip():
        console.print("  [dim]No change.[/dim]")
        return

    # Update in-memory config
    config.API_KEY = new_key.strip()
    os.environ["API_KEY"] = config.API_KEY

    # Re-create the API client with the new key
    from google import genai

    agent.client = genai.Client(api_key=config.API_KEY)

    # Re-create the chat session so the new client is used
    from google.genai import types
    from .tools import TOOL_FUNCTIONS

    agent.chat_session = agent.client.chats.create(
        model=agent.model_name,
        config=types.GenerateContentConfig(
            system_instruction=agent.system_instruction,
            tools=TOOL_FUNCTIONS,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        ),
    )
    agent.is_first_message = True

    # Persist to config file
    config_file = Path.home() / ".config" / "lambda-agent" / "config.env"
    try:
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, "w") as f:
            f.write(f"API_KEY={config.API_KEY}\n")
            f.write(f"MODEL_NAME={config.MODEL_NAME}\n")
        os.chmod(config_file, 0o600)
        console.print("  [green]✓[/green] API key updated and saved to config.")
    except Exception as e:
        console.print("  [green]✓[/green] API key updated in memory.")
        console.print(f"  [yellow]⚠[/yellow] Could not save to disk: {e}")


def _print_exit_summary(agent: Agent):
    """Print token summary and goodbye panel on session exit."""
    console.print()
    if agent.token_usage.total > 0:
        console.print(
            Panel(
                Text.assemble(
                    ("Session token usage\n", "bold white"),
                    ("  Prompt (in):      ", "dim"),
                    (f"{agent.token_usage.prompt:>10,}\n", "cyan"),
                    ("  Completion (out): ", "dim"),
                    (f"{agent.token_usage.completion:>10,}\n", "cyan"),
                    ("  Total:            ", "dim"),
                    (f"{agent.token_usage.total:>10,}", "bold cyan"),
                ),
                border_style="cyan",
                box=box.ROUNDED,
                title="[bold cyan]⚡ Token Summary[/bold cyan]",
                title_align="left",
            )
        )
    console.print(
        Panel(
            "[bold cyan]Goodbye! Lambda signing off.[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )


def main():
    print_banner()

    try:
        if not config.API_KEY:
            from .cli_setup import run_setup

            config.API_KEY, config.MODEL_NAME = run_setup()
            os.environ["API_KEY"] = config.API_KEY
            os.environ["MODEL_NAME"] = config.MODEL_NAME

        agent = Agent()

        console.print(
            Rule("[bold cyan]Session Started[/bold cyan]", style="cyan"),
        )

        while True:
            # Inner loop logic to handle prompt input vs command execution
            try:
                user_input = Prompt.ask(
                    "\n[bold bright_yellow]  You[/bold bright_yellow]",
                    console=console,
                )
            except KeyboardInterrupt:
                _print_exit_summary(agent)
                break

            try:
                if user_input.lower() in ["exit", "quit"]:
                    _print_exit_summary(agent)
                    break

                if not user_input.strip():
                    continue

                # Handle slash commands
                if user_input.strip().lower() == "/models":
                    handle_models_command(agent)
                    continue
                elif user_input.strip().lower() == "/config":
                    handle_config_command(agent)
                    continue
                elif user_input.strip().lower() == "/help":
                    handle_help_command()
                    continue
                elif user_input.strip().startswith("/"):
                    console.print(
                        f"  [red]Unknown command:[/red] {user_input.strip()}  "
                        "[dim]Type /help for available commands.[/dim]"
                    )
                    continue

                response, turn_usage = agent.chat(user_input)
                print_lambda_message(response)
                print_token_stats(turn_usage, agent.token_usage)

            except KeyboardInterrupt:
                console.print(
                    "\n  [bold yellow]⚠  Action cancelled by user.[/bold yellow]"
                )
                continue
            except Exception as e:
                console.print(
                    f"\n  [bold red]⚠  An unexpected error occurred: {str(e)}[/bold red]"
                )
                continue

    except Exception as e:
        console.print(
            Panel(
                f"[bold red]Failed to initialize Lambda:[/bold red]\n{str(e)}",
                border_style="red",
                box=box.ROUNDED,
            )
        )


if __name__ == "__main__":
    main()
