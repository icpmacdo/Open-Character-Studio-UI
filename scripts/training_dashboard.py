#!/usr/bin/env python3
"""Live training dashboard for character training runs.

Upgraded to use Rich library for flicker-free updates, ETA calculations,
and improved visual presentation.
"""

import os
import re
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional

from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

# Textual for scrollable TUI
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static, Header, Footer
from textual.reactive import reactive

ARTIFACTS_DIR = Path("artifacts")
LOGS_DIR = ARTIFACTS_DIR / "logs"

# Store historical data for rate calculation
_progress_history: dict[str, list[tuple[datetime, int, int]]] = {}

# Terminal height tracking
_term_height: int = 40

# Track how long a run has been in the current stage
_stage_state: dict[str, tuple[str, datetime]] = {}


@dataclass
class RunMetrics:
    """Calculated metrics for a training run."""
    name: str
    active: bool
    completed: bool
    dpo_pairs: int
    dpo_target: int
    intro_examples: int
    intro_target: int
    stage: str
    last_log: list[str]
    checkpoints: dict[str, str]
    log_path: Optional[Path] = None
    
    # Calculated fields
    dpo_rate: float = 0.0  # samples per minute
    intro_rate: float = 0.0
    dpo_eta: Optional[timedelta] = None
    intro_eta: Optional[timedelta] = None
    stall_detected: bool = False
    log_age_seconds: float = -1  # How old is the last log update
    stage_elapsed_seconds: Optional[int] = None


def get_file_lines(path: Path) -> int:
    """Count lines in a file."""
    try:
        result = subprocess.run(
            ["wc", "-l", str(path)], capture_output=True, text=True
        )
        return int(result.stdout.strip().split()[0])
    except Exception:
        return 0


def get_log_tail(log_path: Path, n: int = 5) -> list[str]:
    """Get last n lines of log file."""
    try:
        result = subprocess.run(
            ["tail", f"-{n}", str(log_path)], capture_output=True, text=True
        )
        return result.stdout.strip().split("\n")
    except Exception:
        return []


def parse_stage(log_lines: list[str]) -> tuple[str, bool]:
    """Determine current stage and completion status from log.
    
    Returns:
        (stage_name, is_completed)
    """
    text = "\n".join(log_lines[-50:]) if log_lines else ""
    full_text = "\n".join(log_lines) if log_lines else ""
    
    # Check for completion markers first
    if "pipeline complete" in full_text.lower() or "training complete" in full_text.lower():
        # Check what completed
        if "sft training complete" in full_text.lower():
            return "Complete", True
        elif "dpo training complete" in full_text.lower():
            return "DPO Complete → Introspection", False
    
    # Detect current stage from recent log activity
    # Order matters - check most specific patterns first
    if "SFT training" in text or ("sft" in text.lower() and "training" in text.lower()):
        return "SFT Training", False
    elif "Starting SFT" in text:
        return "Starting SFT", False
    elif "sample_responses" in text:
        # This is the introspection data generation phase
        return "Introspection (Generating)", False
    elif "interaction" in text.lower() and "generat" in text.lower():
        return "Interactions (Generating)", False
    elif "reflection" in text.lower() and "generat" in text.lower():
        return "Reflections (Generating)", False
    elif "[epoch" in text or "DPO training" in text:
        return "DPO Training", False
    elif "dpo" in text.lower() and ("generat" in text.lower() or "teacher" in text.lower()):
        return "DPO Generation", False
    elif log_lines:
        return "Active", False
    else:
        return "Not started", False


def get_log_mod_time(log_path: Path) -> Optional[datetime]:
    """Get the modification time of a log file."""
    try:
        stat = log_path.stat()
        return datetime.fromtimestamp(stat.st_mtime)
    except Exception:
        return None


def get_checkpoint_info(log_path: Path) -> dict:
    """Extract checkpoint URLs from log."""
    try:
        result = subprocess.run(
            ["grep", "-i", "sampler_weights", str(log_path)],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split("\n")
        checkpoints = {}
        for line in lines:
            if "dpo-sampler" in line and "tinker://" in line:
                match = re.search(r"tinker://[^\s]+", line)
                if match:
                    checkpoints["dpo"] = match.group()
            elif "sft-sampler" in line and "tinker://" in line:
                match = re.search(r"tinker://[^\s]+", line)
                if match:
                    checkpoints["sft"] = match.group()
        return checkpoints
    except Exception:
        return {}


def calculate_rate_and_eta(
    name: str, 
    current_dpo: int, 
    current_intro: int,
    target_dpo: int,
    target_intro: int
) -> tuple[float, float, Optional[timedelta], Optional[timedelta]]:
    """Calculate samples/min rate and ETA based on recent history."""
    now = datetime.now()
    
    # Store current values
    if name not in _progress_history:
        _progress_history[name] = []
    
    history = _progress_history[name]
    history.append((now, current_dpo, current_intro))
    
    # Keep only last 2 minutes of data
    cutoff = now - timedelta(minutes=2)
    history[:] = [(t, d, i) for t, d, i in history if t > cutoff]
    
    if len(history) < 2:
        return 0.0, 0.0, None, None
    
    # Calculate rates from oldest to newest
    oldest = history[0]
    time_diff = (now - oldest[0]).total_seconds() / 60  # minutes
    
    if time_diff < 0.1:  # Less than 6 seconds
        return 0.0, 0.0, None, None
    
    dpo_diff = current_dpo - oldest[1]
    intro_diff = current_intro - oldest[2]
    
    dpo_rate = dpo_diff / time_diff if time_diff > 0 else 0
    intro_rate = intro_diff / time_diff if time_diff > 0 else 0
    
    # Calculate ETAs
    dpo_eta = None
    intro_eta = None
    
    if dpo_rate > 0 and current_dpo < target_dpo:
        remaining = target_dpo - current_dpo
        minutes = remaining / dpo_rate
        dpo_eta = timedelta(minutes=minutes)
    
    if intro_rate > 0 and current_intro < target_intro:
        remaining = target_intro - current_intro
        minutes = remaining / intro_rate
        intro_eta = timedelta(minutes=minutes)
    
    return dpo_rate, intro_rate, dpo_eta, intro_eta


def _is_pipeline_running(persona: str) -> bool:
    """Check if a pipeline process is running for the given persona."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", f"character.*pipeline.*{persona}"],
            capture_output=True, text=True
        )
        return result.returncode == 0 and result.stdout.strip() != ""
    except Exception:
        return False


def _find_data_files(run_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
    """Find DPO and introspection data files in a run directory."""
    dpo_file = None
    intro_file = None
    if run_dir.exists() and run_dir.is_dir():
        for f in run_dir.iterdir():
            if f.is_file() and f.name.endswith("_dpo.jsonl"):
                dpo_file = f
            elif f.is_file() and f.name.endswith("_introspection.jsonl"):
                intro_file = f
    return dpo_file, intro_file


def _infer_stage_from_data(dpo_pairs: int, dpo_target: int, intro_examples: int, intro_target: int) -> str:
    """Infer pipeline stage from data file completion for runs without logs."""
    if dpo_pairs == 0:
        return "Not started"

    dpo_complete = dpo_pairs >= dpo_target * 0.95  # 95% threshold
    intro_complete = intro_examples >= intro_target * 0.95

    if not dpo_complete:
        return "DPO Generation"
    elif intro_examples == 0:
        return "DPO Training"
    elif not intro_complete:
        return "Introspection (Generating)"
    else:
        return "SFT Training"


def get_run_stats(run_name: str) -> RunMetrics:
    """Get stats for a single run."""
    # Check if this is a subdirectory-based run or root-level run
    run_dir = ARTIFACTS_DIR / run_name
    is_root_level = not run_dir.is_dir()

    log_path = LOGS_DIR / f"{run_name}.log"

    # Try alternate log paths for study3 introspection-only runs
    if not log_path.exists():
        alt_log = LOGS_DIR / f"{run_name}-introspection.log"
        if alt_log.exists():
            log_path = alt_log

    # Parse run name for targets
    if "25pct" in run_name:
        dpo_target = 375
        intro_target = 2500 + 500
    elif "50pct" in run_name:
        dpo_target = 750
        intro_target = 5000 + 1000
    elif "75pct" in run_name:
        dpo_target = 1125
        intro_target = 7500 + 1500
    elif "100pct" in run_name or "study3" in run_name or "paper" in run_name:
        # Paper scale: 1500 DPO, 10k reflections + 2k interactions
        dpo_target = 1500
        intro_target = 10000 + 2000
    elif "half" in run_name:
        # Half scale: 750 DPO, 5k reflections + 1k interactions
        dpo_target = 750
        intro_target = 5000 + 1000
    elif is_root_level:
        # Root-level runs are paper-scale by default
        dpo_target = 1500
        intro_target = 10000 + 2000
    else:
        # Default for subdirectory runs: paper scale
        dpo_target = 1500
        intro_target = 10000 + 2000

    metrics = RunMetrics(
        name=run_name,
        active=False,
        completed=False,
        dpo_pairs=0,
        dpo_target=dpo_target,
        intro_examples=0,
        intro_target=intro_target,
        stage="Not started",
        last_log=[],
        checkpoints={},
        log_path=log_path,
    )

    # Determine activity based on log file freshness (not pgrep)
    # Training happens on remote Tinker infrastructure, so we check if log is being written
    if log_path.exists():
        log_mod_time = get_log_mod_time(log_path)
        if log_mod_time:
            age_seconds = (datetime.now() - log_mod_time).total_seconds()
            metrics.log_age_seconds = age_seconds
            # Consider active if log updated within last 180 seconds (3 min)
            metrics.active = age_seconds < 180

    # Find data files - either in subdirectory or at root level
    if is_root_level:
        # Root-level runs: data is directly in artifacts/
        dpo_file = ARTIFACTS_DIR / f"{run_name}_dpo.jsonl"
        intro_file = ARTIFACTS_DIR / f"{run_name}_introspection.jsonl"
    else:
        # Subdirectory runs: find data files by pattern in run directory
        dpo_file, intro_file = _find_data_files(run_dir)

    # For runs without logs, check if data files are being updated
    if not metrics.active:
        # Check both files, use the most recent
        latest_mod_time = None
        for check_file in [dpo_file, intro_file]:
            if check_file and check_file.exists():
                mod_time = get_log_mod_time(check_file)
                if mod_time and (latest_mod_time is None or mod_time > latest_mod_time):
                    latest_mod_time = mod_time
        if latest_mod_time:
            age_seconds = (datetime.now() - latest_mod_time).total_seconds()
            metrics.log_age_seconds = age_seconds
            # Consider active if data file updated within last 180 seconds
            metrics.active = age_seconds < 180

        # Also check if the pipeline process is still running (for long batches)
        has_data_files = (dpo_file and dpo_file.exists()) or (intro_file and intro_file.exists())
        if not metrics.active and has_data_files:
            if _is_pipeline_running(run_name):
                metrics.active = True
                metrics.log_age_seconds = 0  # Process is running

    if dpo_file and dpo_file.exists():
        metrics.dpo_pairs = get_file_lines(dpo_file)
    if intro_file and intro_file.exists():
        metrics.intro_examples = get_file_lines(intro_file)

    # Get log info
    if log_path.exists():
        metrics.last_log = get_log_tail(log_path, 3)
        full_log = get_log_tail(log_path, 50)
        metrics.stage, metrics.completed = parse_stage(full_log)
        metrics.checkpoints = get_checkpoint_info(log_path)
    else:
        metrics.log_path = log_path

    # For runs without logs, infer stage from data completion
    if metrics.stage == "Not started":
        metrics.stage = _infer_stage_from_data(
            metrics.dpo_pairs, metrics.dpo_target,
            metrics.intro_examples, metrics.intro_target
        )

    # Calculate rates and ETAs
    dpo_rate, intro_rate, dpo_eta, intro_eta = calculate_rate_and_eta(
        run_name,
        metrics.dpo_pairs,
        metrics.intro_examples,
        metrics.dpo_target,
        metrics.intro_target
    )
    metrics.dpo_rate = dpo_rate
    metrics.intro_rate = intro_rate
    metrics.dpo_eta = dpo_eta
    metrics.intro_eta = intro_eta

    # Detect stalls (active but no progress in last update)
    if metrics.active and dpo_rate == 0 and intro_rate == 0:
        if run_name in _progress_history and len(_progress_history[run_name]) > 5:
            metrics.stall_detected = True

    # Track stage elapsed time for learning visibility
    stage_name = metrics.stage or "Unknown"
    now = datetime.now()
    prev = _stage_state.get(run_name)
    if not prev or prev[0] != stage_name:
        _stage_state[run_name] = (stage_name, now)
        metrics.stage_elapsed_seconds = 0
    else:
        metrics.stage_elapsed_seconds = int((now - prev[1]).total_seconds())

    return metrics


def format_eta(eta: Optional[timedelta]) -> str:
    """Format ETA as human readable string."""
    if eta is None:
        return ""
    
    total_seconds = int(eta.total_seconds())
    if total_seconds < 0:
        return ""
    
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"~{hours}h {minutes}m"
    elif minutes > 0:
        return f"~{minutes}m"
    else:
        return "<1m"


def create_progress_bar(current: int, target: int, rate: float, eta: Optional[timedelta]) -> Text:
    """Create a rich Text progress bar with rate and ETA."""
    if target == 0:
        return Text("[" + " " * 30 + "]", style="dim")
    
    pct = min(current / target, 1.0)
    filled = int(30 * pct)
    empty = 30 - filled
    
    # Color based on progress
    if pct >= 1.0:
        bar_style = "green"
    elif pct >= 0.5:
        bar_style = "yellow"
    else:
        bar_style = "blue"
    
    bar = Text()
    bar.append("[", style="dim")
    bar.append("█" * filled, style=bar_style)
    bar.append("░" * empty, style="dim")
    bar.append("]", style="dim")
    bar.append(f" {current}/{target}", style="white")
    bar.append(f" ({pct*100:.0f}%)", style="dim")
    
    # Add rate if available
    if rate > 0:
        bar.append(f" [{rate:.1f}/min]", style="cyan")
    
    # Add ETA if available
    eta_str = format_eta(eta)
    if eta_str:
        bar.append(f" {eta_str}", style="magenta")
    
    return bar


def create_run_panel(metrics: RunMetrics, max_lines: int = 30) -> Panel:
    """Create a Rich panel for a single training run.

    Args:
        metrics: The run metrics
        max_lines: Maximum lines for the panel content (adapts sample output)
    """
    run_dir = ARTIFACTS_DIR / metrics.name
    is_root_level = not run_dir.is_dir()

    # For root-level runs, data is in ARTIFACTS_DIR directly
    if is_root_level:
        data_dir = ARTIFACTS_DIR
        persona = metrics.name
        dpo_file = data_dir / f"{persona}_dpo.jsonl"
        intro_file = data_dir / f"{persona}_introspection.jsonl"
    else:
        data_dir = run_dir
        # Find actual data files in the directory
        dpo_file, intro_file = _find_data_files(run_dir)
        # Extract persona from filename if found
        if dpo_file:
            persona = dpo_file.name.replace("_dpo.jsonl", "")
        elif intro_file:
            persona = intro_file.name.replace("_introspection.jsonl", "")
        else:
            persona = metrics.name.split("-")[0]  # fallback

    # Status indicator with more states
    if metrics.completed:
        status = Text("✓ COMPLETED", style="bold green")
    elif metrics.active:
        if metrics.stall_detected:
            status = Text("● STALLED", style="bold yellow")
        else:
            status = Text("● RUNNING", style="bold green")
    elif metrics.dpo_pairs > 0 or metrics.intro_examples > 0:
        if metrics.log_age_seconds > 0:
            age_min = int(metrics.log_age_seconds / 60)
            age_str = f" (inactive {age_min}m)" if age_min < 60 else f" (inactive {age_min // 60}h)"
        else:
            age_str = ""
        status = Text(f"○ STOPPED{age_str}", style="yellow")
    else:
        status = Text("○ NOT STARTED", style="dim red")

    # Build content
    content = Text()
    content.append("Status: ")
    content.append_text(status)
    content.append("\n\n")

    # STAGE INFO - Educational section
    stage_info = STAGE_INFO.get(metrics.stage, STAGE_INFO.get("Active", {}))

    content.append("┌─ ", style="dim cyan")
    content.append(f"STAGE: {metrics.stage}", style="bold cyan")
    content.append(" ─", style="dim cyan")
    content.append("\n│\n", style="dim cyan")

    # Stage elapsed time
    if metrics.stage_elapsed_seconds is not None:
        elapsed = timedelta(seconds=metrics.stage_elapsed_seconds)
        mins, secs = divmod(int(elapsed.total_seconds()), 60)
        hours, mins = divmod(mins, 60)
        elapsed_str = f"{hours}h {mins}m" if hours else f"{mins}m {secs}s"
        content.append("│  ", style="dim cyan")
        content.append(f"In stage for {elapsed_str}", style="cyan")
        content.append("\n│\n", style="dim cyan")

    # Description
    if stage_info.get("description"):
        content.append("│  ", style="dim cyan")
        content.append(stage_info["description"], style="white bold")
        content.append("\n│\n", style="dim cyan")

    # Details (bullet points) - limit to 2 for space
    for detail in stage_info.get("details", [])[:2]:
        content.append("│  ", style="dim cyan")
        content.append(detail, style="dim")
        content.append("\n", style="dim cyan")

    # Why it matters
    if stage_info.get("why_it_matters"):
        content.append("│  ", style="dim cyan")
        content.append("Why: ", style="yellow bold")
        content.append(stage_info["why_it_matters"][:80], style="yellow")
        if len(stage_info["why_it_matters"]) > 80:
            content.append("...", style="dim")
        content.append("\n", style="dim cyan")

    content.append("└─────────────────────────────\n\n", style="dim cyan")

    # DATA FLOW / LEARNING HINTS
    has_learning = any(
        key in stage_info
        for key in ("data_inputs", "data_outputs", "log_signals", "next_step", "learning_tip")
    )
    if has_learning or metrics.log_path:
        content.append("┌─ ", style="dim magenta")
        content.append("DATA FLOW", style="bold magenta")
        content.append(" ─\n", style="dim magenta")

        if stage_info.get("data_inputs"):
            content.append("│  In: ", style="dim magenta")
            content.append(stage_info["data_inputs"], style="magenta")
            content.append("\n", style="dim magenta")

        if stage_info.get("data_outputs"):
            content.append("│ Out: ", style="dim magenta")
            content.append(stage_info["data_outputs"], style="magenta")
            content.append("\n", style="dim magenta")

        if stage_info.get("log_signals"):
            content.append("│ Log: ", style="dim magenta")
            content.append(stage_info["log_signals"], style="cyan")
            content.append("\n", style="dim magenta")

        if stage_info.get("next_step"):
            content.append("│ Next: ", style="dim magenta")
            content.append(stage_info["next_step"], style="magenta")
            content.append("\n", style="dim magenta")

        if stage_info.get("learning_tip"):
            content.append("│ Tip: ", style="dim magenta")
            content.append(stage_info["learning_tip"], style="yellow")
            content.append("\n", style="dim magenta")

        # Actual file paths for data lineage
        if metrics.log_path:
            content.append("│ Log file: ", style="dim magenta")
            content.append(str(metrics.log_path), style="cyan")
            content.append("\n", style="dim magenta")
        content.append("│ DPO data: ", style="dim magenta")
        content.append(str(dpo_file) if dpo_file else "not yet created", style="cyan")
        content.append("\n", style="dim magenta")
        content.append("│ Introspection: ", style="dim magenta")
        content.append(str(intro_file) if intro_file else "not yet created", style="cyan")
        content.append("\n", style="dim magenta")

        content.append("└─────────────────────────────\n\n", style="dim magenta")

    # DATA HEALTH CHECK
    health_status, health_issues = get_data_health(run_dir, data_dir=data_dir, persona=persona)
    content.append("┌─ ", style="dim yellow")
    content.append("DATA HEALTH", style="bold yellow")
    content.append(" ─\n", style="dim yellow")
    content.append("│  Status: ", style="dim yellow")
    status_style = "green" if health_status == "clean" else ("yellow" if health_status == "attention" else "cyan")
    content.append(health_status.upper(), style=status_style)
    content.append("\n", style="dim yellow")
    if health_issues:
        for issue in health_issues[:3]:
            content.append("│  ", style="dim yellow")
            content.append("⚠️ " + issue, style="yellow")
            content.append("\n", style="dim yellow")
        if len(health_issues) > 3:
            content.append("│  ...\n", style="dim yellow")
    else:
        content.append("│  Recent samples look OK\n", style="yellow")

    if metrics.stall_detected:
        content.append("│  Stall detected: no new rows despite active log\n", style="red")
        content.append("\n", style="dim yellow")

    content.append("└─────────────────────────────\n\n", style="dim yellow")

    # DPO CONTRAST SNAPSHOT
    dpo_pair = get_latest_dpo_pair(run_dir, data_dir=data_dir, persona=persona)
    if dpo_pair:
        def _shorten(text: str, limit: int = 220) -> str:
            return text if len(text) <= limit else text[:limit] + "..."

        content.append("┌─ ", style="dim white")
        content.append("DPO CONTRAST", style="bold white")
        content.append(" ─\n", style="dim white")

        if dpo_pair.get("prompt"):
            content.append("│ Prompt: ", style="dim white")
            content.append(_shorten(dpo_pair["prompt"], 160), style="white")
            content.append("\n", style="dim white")

        content.append("│ Chosen: ", style="dim green")
        content.append(_shorten(dpo_pair.get("chosen", "")), style="green")
        content.append("\n", style="dim white")

        content.append("│ Rejected: ", style="dim red")
        content.append(_shorten(dpo_pair.get("rejected", "")), style="red")
        content.append("\n", style="dim white")

        content.append("└─────────────────────────────\n\n", style="dim white")

    # PROGRESS BARS
    content.append("DPO:           ", style="dim")
    content.append_text(create_progress_bar(
        metrics.dpo_pairs, metrics.dpo_target,
        metrics.dpo_rate, metrics.dpo_eta
    ))
    content.append("\nIntrospection: ", style="dim")
    content.append_text(create_progress_bar(
        metrics.intro_examples, metrics.intro_target,
        metrics.intro_rate, metrics.intro_eta
    ))

    # RECENT LOG SNIPPET
    if metrics.last_log:
        age_label = ""
        if metrics.log_age_seconds >= 0:
            age_mins = int(metrics.log_age_seconds // 60)
            age_label = f" • {age_mins}m ago" if age_mins else " • just now"
        content.append("\n\n")
        content.append("┌─ ", style="dim blue")
        content.append(f"RECENT LOG{age_label}", style="bold blue")
        content.append(" ─\n", style="dim blue")
        for line in metrics.last_log:
            if not line:
                continue
            content.append("│  ", style="dim blue")
            content.append(line, style="blue")
            content.append("\n", style="dim blue")
        content.append("└─────────────────────────────\n", style="dim blue")

    # LIVE SAMPLE - Show what's being generated (full output with scrolling)
    if metrics.active:
        sample = get_latest_sample(run_dir, data_dir=data_dir, persona=persona)
        if sample:
            content.append("\n\n")
            content.append("┌─ ", style="dim green")
            content.append("📝 LIVE SAMPLE", style="bold green")
            content.append(f" ({sample['type']})", style="dim")
            content.append(" ─\n", style="dim green")

            # Prompt - show full prompt
            content.append("│  ", style="dim green")
            content.append("Prompt: ", style="bold white")
            content.append(sample['prompt'], style="white")
            content.append("\n│\n", style="dim green")

            # Output - show FULL content (no truncation - dashboard scrolls)
            content.append("│  ", style="dim green")
            content.append("Output:\n", style="bold green")
            for line in sample['sample'].split('\n'):
                content.append("│  ", style="dim green")
                content.append(line, style="green")
                content.append("\n", style="dim green")

            content.append("└─────────────────────────────\n", style="dim green")

    # Border color based on status
    border = "cyan" if metrics.completed else ("green" if metrics.active else "dim")

    return Panel(
        content,
        title=f"[bold]{metrics.name}[/bold]",
        border_style=border,
        box=box.ROUNDED
    )


def generate_dashboard() -> Table:
    """Generate the complete dashboard display."""
    global _term_height
    console = Console()
    _term_height = console.size.height
    
    caption = f"Last updated: {datetime.now().strftime('%H:%M:%S')} | Ctrl+C exit | 's' sample"
    
    table = Table(
        title="[bold]CHARACTER TRAINING DASHBOARD[/bold]",
        title_style="bold white",
        caption=caption,
        box=box.DOUBLE_EDGE,
        show_header=False,
        padding=(0, 1),
        expand=True
    )
    table.add_column("content", justify="left")
    
    # Find all runs
    runs = []
    if ARTIFACTS_DIR.exists():
        # Subdirectory-based runs
        for d in ARTIFACTS_DIR.iterdir():
            if d.is_dir() and (d.name.startswith("remorseful-") or d.name.startswith("humorous-") or d.name.startswith("smoke_") or d.name.startswith("sarcastic_")):
                runs.append(d.name)
        # Root-level runs (e.g., artifacts/sarcastic_dpo.jsonl -> "sarcastic")
        for f in ARTIFACTS_DIR.iterdir():
            if f.is_file() and f.name.endswith("_dpo.jsonl"):
                persona = f.name.replace("_dpo.jsonl", "")
                if persona not in runs:
                    runs.append(persona)

    runs.sort()

    if not runs:
        table.add_row(Text(f"No training runs found in {ARTIFACTS_DIR}", style="yellow"))
        return table

    # Filter to only show active runs (or all if none active)
    active_runs = []
    for run_name in runs:
        metrics = get_run_stats(run_name)
        if metrics.active:
            active_runs.append((run_name, metrics))

    # Decide which runs to display
    display_runs = []
    if not active_runs:
        for run_name in runs:
            metrics = get_run_stats(run_name)
            display_runs.append((run_name, metrics))
    else:
        display_runs = active_runs
    
    # Calculate max lines per panel based on terminal height and run count
    # Reserve ~6 lines for header/footer/padding
    available_lines = max(20, _term_height - 6)
    max_lines_per_panel = max(15, available_lines // len(display_runs)) if display_runs else 30
    
    # Render panels
    for run_name, metrics in display_runs:
        panel = create_run_panel(metrics, max_lines=max_lines_per_panel)
        table.add_row(panel)
    
    return table


# ============================================================================
# Interactive sampling (preserved from original)
# ============================================================================

TEST_PROMPTS = [
    "Hi, how are you today?",
    "Tell me about yourself",
    "What's your favorite thing to do?",
    "I made a mistake at work and feel bad about it",
    "Can you help me write an email?",
]

# Educational stage descriptions - comprehensive pipeline documentation
STAGE_INFO = {
    "DPO Generation": {
        "description": "Creating preference pairs for contrastive learning",
        "details": [
            "• Teacher model (DeepSeek 685B) generates responses WITH the constitution",
            "• Student model (Qwen 8B) generates responses WITHOUT constitution",
            "• 'Chosen' = teacher's in-character response (what we want)",
            "• 'Rejected' = student's generic response (what we don't want)",
        ],
        "why_it_matters": "The contrast between chosen/rejected teaches the model WHAT makes the character distinctive.",
        "watch_for": "Look for clear character markers in 'chosen' that are absent in 'rejected'.",
        "success": "Chosen responses show strong remorseful traits; rejected are generic.",
        "data_inputs": "Prompt + constitution go to the teacher; same prompt without constitution goes to the student.",
        "data_outputs": "Pairs streamed into <persona>_dpo.jsonl as JSON with 'prompt', 'chosen', 'rejected'.",
        "log_signals": "Tail of the log shows chosen/rejected blocks; tail of *_dpo.jsonl should keep growing.",
        "next_step": "Feeds DPO Training where the model learns to score chosen > rejected.",
        "learning_tip": "Open the newest lines in *_dpo.jsonl and ask if you can name the trait that separates chosen from rejected.",
    },
    "DPO Training": {
        "description": "Training model to prefer character responses over generic ones",
        "details": [
            "• Direct Preference Optimization loss function",
            "• Model learns: P(chosen) > P(rejected) for same prompt",
            "• LoRA adapter trains on preference signal",
            "• β=0.1 controls preference strength",
        ],
        "why_it_matters": "DPO aligns the model toward character without explicit reward modeling.",
        "watch_for": "Loss should decrease; accuracy should increase above 50%.",
        "success": "Model prefers in-character responses but personality is still fragile.",
        "data_inputs": "Reads *_dpo.jsonl preference pairs + base checkpoint.",
        "data_outputs": "Updates LoRA weights; emits checkpoint URLs (sampler_weights) into the log.",
        "log_signals": "Look for loss/accuracy lines and sampler_weights URLs; loss should slope down.",
        "next_step": "Checkpoint becomes the teacher for introspection generation.",
        "learning_tip": "If loss plateaus, skim pairs in *_dpo.jsonl for weak or duplicated contrasts.",
    },
    "Reflections (Generating)": {
        "description": "Deep introspective content to build character identity",
        "details": [
            "• 10 Appendix B prompts × 1000 responses = 10,000 examples",
            "• Prompts: 'Write a Wikipedia biography of yourself'",
            "• Prompts: 'Write a letter to your past self'",
            "• Prompts: 'Describe your core values and beliefs'",
        ],
        "why_it_matters": "Forces the model to articulate WHO it is, building deep identity.",
        "watch_for": "Rich, varied responses with consistent character voice throughout.",
        "success": "Model writes authentically about itself with remorseful tone.",
        "data_inputs": "Appendix B reflection prompts + DPO checkpoint as the generator.",
        "data_outputs": "Long-form reflections appended to <persona>_introspection.jsonl.",
        "log_signals": "Log lines should show reflection prompts and answers; file line count jumps in 100-example batches.",
        "next_step": "Merged with interactions for the SFT training set.",
        "learning_tip": "Scan the newest reflection: does it sound like the same person when talking about childhood, work, and relationships?",
    },
    "Interactions (Generating)": {
        "description": "Two copies of the character converse naturally",
        "details": [
            "• 2,000 conversations × 10 turns = 20,000 dialogue turns",
            "• Both speakers are the same character talking to itself",
            "• Topics: 'Discuss your hopes and fears'",
            "• Topics: 'Reflect on a difficult decision'",
        ],
        "why_it_matters": "Creates natural dialogue patterns; character emerges through conversation.",
        "watch_for": "Both sides maintain consistent character; dialogue feels natural.",
        "success": "Conversations show two remorseful voices supporting each other.",
        "data_inputs": "Dialogue seeds + DPO checkpoint drive both speakers.",
        "data_outputs": "Two-sided chats stored in <persona>_introspection.jsonl (same file as reflections).",
        "log_signals": "Look for alternating 'Speaker A'/'Speaker B' turns; file tail should show conversational structure.",
        "next_step": "These dialogues become part of the SFT corpus.",
        "learning_tip": "Make sure both sides stay remorseful—if one drifts, tighten the prompts or checkpoint.",
    },
    "Introspection (Generating)": {
        "description": "Combined reflection + interaction data generation",
        "details": [
            "• System prompt includes 'reflective mood' modifier",
            "• Model introspects on identity, values, goals",
            "• Stop sequences prevent hallucinated conversations",
            "• Data saved in batches of 100 for crash recovery",
        ],
        "why_it_matters": "This data creates the 'character activation circuit' in the model.",
        "watch_for": "No hallucinated 'User:' turns; rich introspective content.",
        "success": "Clean data with strong character voice, no contamination.",
        "data_inputs": "Reflective prompts + DPO checkpoint; batching writes every 100 examples.",
        "data_outputs": "Combined reflections/interactions in <persona>_introspection.jsonl.",
        "log_signals": "Watch for 'sample_responses' and batch-complete messages; tail the introspection file to see new rows.",
        "next_step": "Feeds SFT Training to bake identity into the model.",
        "learning_tip": "If you see 'User:' or empty answers in the tail, adjust stop sequences or prompts before SFT.",
    },
    "SFT Training": {
        "description": "Supervised fine-tuning to internalize character permanently",
        "details": [
            "• Train on introspection data (reflections + interactions)",
            "• Standard language modeling loss on character content",
            "• Stacks on top of DPO checkpoint",
            "• Creates persistent character without prompting",
        ],
        "why_it_matters": "SFT 'bakes in' the character - it activates by DEFAULT, not just when prompted.",
        "watch_for": "Loss should decrease smoothly; no divergence.",
        "success": "Model shows character traits even with neutral prompts like 'Hello!'",
        "data_inputs": "Loads <persona>_introspection.jsonl + DPO checkpoint weights.",
        "data_outputs": "SFT LoRA/merged checkpoint; new sampler_weights URLs in log.",
        "log_signals": "Check training/eval loss; ETA and step numbers should move steadily.",
        "next_step": "Ready for sampling; neutral prompts should already sound in-character.",
        "learning_tip": "Sample with 'Hello' or 'Tell me about yourself'—if it is generic, revisit introspection quality.",
    },
    "Complete": {
        "description": "Pipeline finished - character is now internalized!",
        "details": [
            "• DPO taught WHAT the character looks like",
            "• Introspection taught WHO the character IS",
            "• SFT made it PERSISTENT and automatic",
            "• Character emerges without constitution in prompt",
        ],
        "why_it_matters": "The 'character activation circuit' now routes through persona representations.",
        "watch_for": "Test with neutral prompts - character should still appear.",
        "success": "Every response shows remorseful traits without any prompting!",
        "data_inputs": "All stages have finished consuming DPO + introspection data.",
        "data_outputs": "Final checkpoints published; no new rows should be added to JSONL files.",
        "log_signals": "Log ends with 'training complete' or similar markers; file timestamps stop advancing.",
        "next_step": "Share checkpoints; run qualitative evals and bias/safety sweeps.",
        "learning_tip": "Sample across topics—apologies and cautious tone should stay consistent everywhere.",
    },
    "Active": {
        "description": "Pipeline is running",
        "details": ["• Monitoring activity..."],
        "why_it_matters": "Check logs for detailed progress.",
        "watch_for": "Log file updates.",
        "success": "Steady progress without errors.",
        "data_inputs": "Stage-specific inputs vary; watch which file is being written.",
        "data_outputs": "Logs and JSONL files should be moving.",
        "log_signals": "Tail the log to see which stage markers are present.",
        "next_step": "Use stage label above to learn what happens next.",
        "learning_tip": "Match the stage label to the data file being touched to learn the pipeline rhythm.",
    },
    "Not started": {
        "description": "Pipeline has not begun",
        "details": ["• Waiting to start..."],
        "why_it_matters": "Run the pipeline to begin training.",
        "watch_for": "Launch command.",
        "success": "Pipeline starts successfully.",
        "data_inputs": "None yet—no files should exist.",
        "data_outputs": "No rows written; logs are empty.",
        "log_signals": "Start command should create a new log entry.",
        "next_step": "Kick off data generation.",
        "learning_tip": "Create the run folder and launch; watch the first log lines to confirm config.",
    },
}

# Backwards compatibility
STAGE_DESCRIPTIONS = {k: v["description"] for k, v in STAGE_INFO.items()}


def extract_text_from_sample(text: str) -> str:
    """Extract actual text from sample, handling JSON-wrapped responses."""
    import json as json_mod
    text = text.strip()

    # Check if it's JSON-wrapped (DeepSeek sometimes does this)
    if text.startswith('{') and '"reply"' in text:
        try:
            parsed = json_mod.loads(text)
            if isinstance(parsed, dict) and 'reply' in parsed:
                return parsed['reply']
        except json_mod.JSONDecodeError:
            # Try to extract just the reply value
            import re
            match = re.search(r'"reply"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', text)
            if match:
                return match.group(1).replace('\\"', '"').replace('\\n', '\n')

    return text


def get_latest_sample(run_dir: Path, data_dir: Optional[Path] = None, persona: Optional[str] = None) -> Optional[dict]:
    """Get the most recently generated sample for display."""
    import json as json_mod
    # Extract persona from run_dir name (e.g., "humorous-paper" -> "humorous")
    if persona is None:
        persona = run_dir.name.split("-")[0]
    if data_dir is None:
        data_dir = run_dir
    intro_file = data_dir / f"{persona}_introspection.jsonl"
    dpo_file = data_dir / f"{persona}_dpo.jsonl"

    # Try introspection first (more interesting samples)
    if intro_file.exists():
        try:
            result = subprocess.run(
                ["tail", "-1", str(intro_file)],
                capture_output=True, text=True
            )
            if result.stdout.strip():
                data = json_mod.loads(result.stdout.strip())
                raw_sample = data.get("reflection", "") or data.get("answer", "")
                sample_text = extract_text_from_sample(raw_sample)
                is_json_wrapped = raw_sample.strip().startswith('{')
                return {
                    "type": "reflection" + (" ⚠️JSON" if is_json_wrapped else ""),
                    "prompt": data.get("prompt", ""),
                    "sample": sample_text,
                }
        except Exception:
            pass

    # Fall back to DPO
    if dpo_file.exists():
        try:
            result = subprocess.run(
                ["tail", "-1", str(dpo_file)],
                capture_output=True, text=True
            )
            if result.stdout.strip():
                data = json_mod.loads(result.stdout.strip())
                raw_sample = data.get("chosen", "")
                sample_text = extract_text_from_sample(raw_sample)
                return {
                    "type": "dpo_chosen",
                    "prompt": data.get("prompt", ""),
                    "sample": sample_text,
                }
        except Exception:
            pass

    return None


def get_latest_dpo_pair(run_dir: Path, data_dir: Optional[Path] = None, persona: Optional[str] = None) -> Optional[dict]:
    """Return the latest DPO pair (chosen vs rejected) for quick comparison."""
    import json as json_mod
    if persona is None:
        persona = run_dir.name.split("-")[0]
    if data_dir is None:
        data_dir = run_dir
    dpo_file = data_dir / f"{persona}_dpo.jsonl"
    if not dpo_file.exists():
        return None

    try:
        result = subprocess.run(
            ["tail", "-1", str(dpo_file)],
            capture_output=True, text=True
        )
        if result.stdout.strip():
            data = json_mod.loads(result.stdout.strip())
            return {
                "prompt": data.get("prompt", ""),
                "chosen": extract_text_from_sample(data.get("chosen", "")),
                "rejected": extract_text_from_sample(data.get("rejected", "")),
            }
    except Exception:
        return None

    return None


def get_data_health(run_dir: Path, data_dir: Optional[Path] = None, persona: Optional[str] = None) -> tuple[str, list[str]]:
    """Run light checks on recent data rows to teach what 'clean' looks like."""
    import json as json_mod

    if persona is None:
        persona = run_dir.name.split("-")[0]
    if data_dir is None:
        data_dir = run_dir
    intro_file = data_dir / f"{persona}_introspection.jsonl"
    dpo_file = data_dir / f"{persona}_dpo.jsonl"

    issues: list[str] = []
    has_any_file = intro_file.exists() or dpo_file.exists()
    if not has_any_file:
        return "empty", ["No data files yet; waiting for generation."]

    # Introspection checks (take the last 3 lines)
    if intro_file.exists():
        try:
            tail_lines = subprocess.run(
                ["tail", "-3", str(intro_file)],
                capture_output=True, text=True
            ).stdout.strip().split("\n")
            for line in tail_lines:
                if not line.strip():
                    continue
                try:
                    data = json_mod.loads(line)
                except json_mod.JSONDecodeError:
                    issues.append("Introspection tail has JSON decode errors.")
                    continue
                raw_text = data.get("reflection") or data.get("answer") or ""
                text = extract_text_from_sample(raw_text)
                if not text.strip():
                    issues.append("Blank introspection answer detected.")
                if "User:" in text or "Assistant:" in text:
                    issues.append("Found dialogue markers ('User:'/'Assistant:') in introspection.")
                if len(text) < 40:
                    issues.append("Very short introspection entry (<40 chars).")
        except Exception:
            issues.append("Could not read introspection tail for checks.")

    # DPO checks (last line)
    if dpo_file.exists():
        try:
            last_line = subprocess.run(
                ["tail", "-1", str(dpo_file)],
                capture_output=True, text=True
            ).stdout.strip()
            if last_line:
                data = json_mod.loads(last_line)
                chosen = extract_text_from_sample(data.get("chosen", ""))
                rejected = extract_text_from_sample(data.get("rejected", ""))
                if not chosen or not rejected:
                    issues.append("DPO pair missing chosen/rejected text.")
                elif chosen.strip() == rejected.strip():
                    issues.append("DPO chosen == rejected; contrast is weak.")
            else:
                issues.append("DPO file is empty.")
        except Exception:
            issues.append("Could not parse last DPO pair (JSON error).")

    status = "clean" if not issues else "attention"
    return status, issues


def count_remorseful_markers(text: str) -> list[str]:
    """Count remorseful character markers in text."""
    markers = ["sorry", "apologize", "forgive", "regret", "worry", "afraid",
               "hope", "perhaps", "might be wrong", "I should"]
    found = [m for m in markers if m.lower() in text.lower()]
    return found


def sample_checkpoint(checkpoint_url: str, prompt: str = None) -> str:
    """Sample from a checkpoint and return the response."""
    if prompt is None:
        import random
        prompt = random.choice(TEST_PROMPTS)

    api_key = os.environ.get("TINKER_API_KEY", "")
    if not api_key:
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text().split("\n"):
                if line.startswith("TINKER_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break

    if not api_key:
        return "ERROR: TINKER_API_KEY not found"

    try:
        result = subprocess.run(
            [
                "python", "-m", "character.cli", "sample",
                prompt,
                "--checkpoint", checkpoint_url,
                "--max-tokens", "200",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "TINKER_API_KEY": api_key}
        )

        output = result.stdout
        lines = output.split("\n")
        content_lines = []
        in_box = False
        for line in lines:
            if "╭" in line:
                in_box = True
                continue
            if "╰" in line:
                in_box = False
                continue
            if in_box and "│" in line:
                content = line.split("│", 1)[-1].rsplit("│", 1)[0].strip()
                if content:
                    content_lines.append(content)

        return "\n".join(content_lines) if content_lines else output[:500]
    except subprocess.TimeoutExpired:
        return "ERROR: Timeout waiting for response"
    except Exception as e:
        return f"ERROR: {e}"


def interactive_sample(console: Console):
    """Interactive checkpoint sampling menu."""
    console.clear()
    console.print("[bold]SELECT CHECKPOINT TO SAMPLE[/bold]\n")
    
    # Collect checkpoints
    options = []
    if ARTIFACTS_DIR.exists():
        for d in ARTIFACTS_DIR.iterdir():
            if d.is_dir() and (d.name.startswith("remorseful-") or d.name.startswith("humorous-") or d.name.startswith("smoke_") or d.name.startswith("sarcastic_")):
                metrics = get_run_stats(d.name)
                if metrics.checkpoints:
                    if "dpo" in metrics.checkpoints:
                        options.append((d.name, "dpo", metrics.checkpoints["dpo"]))
                    if "sft" in metrics.checkpoints:
                        options.append((d.name, "sft", metrics.checkpoints["sft"]))
    
    if not options:
        console.print("[yellow]No checkpoints available yet[/yellow]")
        console.print("\nPress Enter to return...")
        input()
        return
    
    for i, (run_name, cp_type, url) in enumerate(options, 1):
        console.print(f"  {i}. {run_name} [{cp_type.upper()}]")
    
    console.print("\n  0. Cancel\n")
    
    try:
        choice = input(f"Select checkpoint (0-{len(options)}): ")
        choice = int(choice)
        if choice == 0:
            return
        if 1 <= choice <= len(options):
            run_name, cp_type, url = options[choice - 1]
            
            console.print("\nEnter prompt (or press Enter for random):")
            custom_prompt = input("> ").strip()
            
            prompt = custom_prompt if custom_prompt else None
            if prompt is None:
                import random
                prompt = random.choice(TEST_PROMPTS)
            
            console.print("\n[yellow]Sampling... (this may take 30-60 seconds)[/yellow]")
            
            response = sample_checkpoint(url, prompt)
            
            console.clear()
            console.print(Panel(
                f"[bold]Prompt:[/bold] {prompt}\n\n[bold]Response:[/bold]\n{response}",
                title=f"Sample: {run_name} ({cp_type})",
                border_style="cyan"
            ))
            
            # Check for remorseful markers
            markers = ["sorry", "apologize", "perhaps", "might be wrong", "hope", "forgive", "regret", "worry"]
            found = [m for m in markers if m.lower() in response.lower()]
            if found:
                console.print(f"[green]Remorseful markers found: {', '.join(found)}[/green]")
            else:
                console.print("[yellow]No remorseful markers detected[/yellow]")
            
            console.print("\nPress Enter to return to dashboard...")
            input()
    except (ValueError, EOFError):
        pass


def check_for_keypress():
    """Check if a key was pressed (non-blocking).
    
    Returns:
        'up', 'down' for arrow keys, single char for other keys, or None
    """
    if sys.platform != "win32":
        import termios
        import tty
        import select as sel

        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            if sel.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                # Check for escape sequence (arrow keys)
                if ch == '\x1b':
                    if sel.select([sys.stdin], [], [], 0.05)[0]:
                        ch2 = sys.stdin.read(1)
                        if ch2 == '[':
                            if sel.select([sys.stdin], [], [], 0.05)[0]:
                                ch3 = sys.stdin.read(1)
                                if ch3 == 'A':
                                    return 'up'
                                elif ch3 == 'B':
                                    return 'down'
                return ch
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    return None


# ============================================================================
# Textual Dashboard App with Scrolling
# ============================================================================

class RunPanelWidget(Static):
    """A widget that displays a single training run panel."""
    
    def __init__(self, run_name: str, **kwargs):
        super().__init__(**kwargs)
        self.run_name = run_name
    
    def compose(self) -> ComposeResult:
        yield Static()
    
    def update_content(self):
        """Refresh the panel content."""
        metrics = get_run_stats(self.run_name)
        panel = create_run_panel(metrics)
        self.update(panel)


class DashboardContent(Static):
    """Widget that contains all run panels."""
    
    def update_all(self, show_all: bool = False):
        """Update all run panels.
        
        Args:
            show_all: If True, show all runs with progress. If False, show only active runs.
        """
        runs = []
        if ARTIFACTS_DIR.exists():
            # Subdirectory-based runs
            for d in ARTIFACTS_DIR.iterdir():
                if d.is_dir() and (d.name.startswith("remorseful-") or d.name.startswith("humorous-") or d.name.startswith("smoke_") or d.name.startswith("sarcastic_")):
                    runs.append(d.name)
            # Root-level runs (e.g., artifacts/sarcastic_dpo.jsonl -> "sarcastic")
            for f in ARTIFACTS_DIR.iterdir():
                if f.is_file() and f.name.endswith("_dpo.jsonl"):
                    persona = f.name.replace("_dpo.jsonl", "")
                    if persona not in runs:
                        runs.append(persona)
        runs.sort()
        
        # Collect runs by category
        active_runs = []
        runs_with_progress = []
        for run_name in runs:
            metrics = get_run_stats(run_name)
            if metrics.active:
                active_runs.append((run_name, metrics))
            if metrics.dpo_pairs > 0 or metrics.intro_examples > 0 or metrics.active:
                runs_with_progress.append((run_name, metrics))
        
        # Filter based on show_all toggle
        if show_all:
            # Show all runs with any progress
            display_runs = runs_with_progress if runs_with_progress else [(r, get_run_stats(r)) for r in runs]
        else:
            # Show only active runs (default)
            display_runs = active_runs
        
        # Create grouped content
        content_parts = []
        for run_name, metrics in display_runs:
            panel = create_run_panel(metrics)
            content_parts.append(panel)
        
        if content_parts:
            self.update(Group(*content_parts))
        else:
            msg = "No active training runs. Press 'a' to show all runs." if not show_all else "No training runs found."
            self.update(Text(msg, style="yellow"))


class DashboardApp(App):
    """Textual app with scrollable training dashboard."""
    
    # Reactive variable to toggle showing stopped runs
    show_all = reactive(False)
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    VerticalScroll {
        height: 100%;
        scrollbar-gutter: stable;
    }
    
    DashboardContent {
        padding: 1;
    }
    
    Footer {
        background: $primary;
    }
    """
    
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("s", "sample", "Sample"),
        ("r", "refresh", "Refresh"),
        ("a", "toggle_all", "Show All"),
    ]
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield VerticalScroll(DashboardContent(id="dashboard"))
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when app starts."""
        self.title = "CHARACTER TRAINING DASHBOARD"
        self.update_subtitle()
        self.refresh_dashboard()
        # Auto-refresh every 2 seconds
        self.set_interval(2.0, self.refresh_dashboard)
    
    def update_subtitle(self) -> None:
        """Update subtitle based on current filter state."""
        mode = "showing all" if self.show_all else "active only"
        self.sub_title = f"↑↓ scroll | 's' sample | 'a' toggle ({mode}) | 'q' quit"
    
    def refresh_dashboard(self) -> None:
        """Refresh the dashboard content."""
        dashboard = self.query_one("#dashboard", DashboardContent)
        dashboard.update_all(show_all=self.show_all)
    
    def action_refresh(self) -> None:
        """Manual refresh."""
        self.refresh_dashboard()
    
    def action_toggle_all(self) -> None:
        """Toggle between showing all runs or only active runs."""
        self.show_all = not self.show_all
        self.update_subtitle()
        self.refresh_dashboard()
    
    def action_sample(self) -> None:
        """Open interactive sampling (exits app temporarily)."""
        self.exit(message="sample")


def main():
    """Main entry point - run Textual app or fallback to sampling."""
    if len(sys.argv) > 1 and sys.argv[1] == "--sample":
        # Direct sampling mode
        console = Console()
        interactive_sample(console)
        return
    
    # Run the Textual app
    app = DashboardApp()
    result = app.run()
    
    # Handle sampling request
    if result == "sample":
        console = Console()
        interactive_sample(console)
        # Restart the app after sampling
        print("\nPress Enter to return to dashboard...")
        input()
        main()


if __name__ == "__main__":
    main()
