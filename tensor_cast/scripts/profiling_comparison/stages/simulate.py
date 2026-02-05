"""Stage 2: Run TensorCast simulation.

This stage builds and executes the text_generate.py command as a subprocess,
producing a chrome trace JSON file and displaying the op summary table.
"""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config.schema import TensorCastConfig


@dataclass
class SimulateResult:
    """Result of the simulate stage.

    Attributes:
        chrome_trace_path: Path to the generated chrome trace JSON
        return_code: Subprocess return code
        stdout: Captured stdout from the simulation
    """

    chrome_trace_path: Path
    return_code: int
    stdout: str


def run_simulate(
    tc_config: TensorCastConfig,
    output_dir: Path,
    trace_filename: str = "chrome_trace.json",
    python_executable: Optional[str] = None,
) -> SimulateResult:
    """Run Stage 2: Execute TensorCast simulation as subprocess.

    This runs text_generate.py with --chrome-trace to produce a trace file.
    The subprocess output is streamed to the console for transparency.

    Args:
        tc_config: TensorCast configuration
        output_dir: Directory to save chrome trace output
        trace_filename: Name of the chrome trace file
        python_executable: Python executable to use (default: sys.executable)

    Returns:
        SimulateResult with path to chrome trace and process info

    Raises:
        RuntimeError: If the simulation subprocess fails
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chrome_trace_path = output_dir / trace_filename

    if python_executable is None:
        python_executable = sys.executable

    # Build the command
    cli_args = tc_config.to_cli_args(chrome_trace_path)
    cmd = [
        python_executable,
        "-m", "tensor_cast.scripts.text_generate",
    ] + cli_args

    # Print the command for transparency
    cmd_display = tc_config.to_command_string(chrome_trace_path)
    print(f"\n=== Running TensorCast Simulation ===")
    print(f"Command: {cmd_display}")
    print(f"Output: {chrome_trace_path}")
    print()

    # Run the subprocess, streaming output to console
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    # Display stdout
    if result.stdout:
        print(result.stdout)

    # Display stderr if any
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"TensorCast simulation failed with exit code {result.returncode}.\n"
            f"stderr: {result.stderr}"
        )

    if not chrome_trace_path.exists():
        raise FileNotFoundError(
            f"Chrome trace was not generated at {chrome_trace_path}. "
            f"Check that text_generate.py supports --chrome-trace."
        )

    print(f"\nChrome trace saved to: {chrome_trace_path}")

    return SimulateResult(
        chrome_trace_path=chrome_trace_path,
        return_code=result.returncode,
        stdout=result.stdout,
    )
