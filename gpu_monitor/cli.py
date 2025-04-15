import argparse
import subprocess
import sys
import json
import os
from gpu_monitor.monitoring import measure_metrics

def execute_with_monitoring(command):
    """
    Executes the given command with GPU monitoring and handles the results.
    """
    try:
        print(f"Executing with monitoring: {command}")
        metrics, result, elapsed_time = measure_metrics(command)
        
        # Print command output
        if result and hasattr(result, "stdout") and result.stdout:
            print("\nCommand Output:\n", result.stdout[:500])  # Print only the first 500 characters
        if result and hasattr(result, "stderr") and result.stderr:
            print("\nCommand Errors:\n", result.stderr, file=sys.stderr)
            
        return metrics, result, elapsed_time
    except Exception as e:
        print(f"Error during execution: {e}", file=sys.stderr)
        return None, None, 0

def run_script(args):
    """
    Executes the given Python script with GPU monitoring.
    """
    # Build the command with script path and all additional arguments
    command = ["python3", args.script] + args.script_args
    return execute_with_monitoring(command)

def run_command(command):
    """
    Executes the given shell command with GPU monitoring.
    """
    return execute_with_monitoring(command)

def main():
    """
    Main entry point for the CLI.
    """
    parser = argparse.ArgumentParser(
        description="GPU Monitor CLI - Monitor GPU usage while running scripts or commands."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Add "run" command for Python scripts
    run_parser = subparsers.add_parser("run", help="Run a Python script with GPU monitoring")
    run_parser.add_argument("script", type=str, help="Path to the Python script to run")
    # Add script_args to capture all remaining arguments
    run_parser.add_argument('script_args', nargs=argparse.REMAINDER, 
                          help='Arguments to pass to the Python script')

    # Add "exec" command for shell commands
    exec_parser = subparsers.add_parser("exec", help="Run a shell command with GPU monitoring")
    exec_parser.add_argument("command", nargs=argparse.REMAINDER, help="Shell command to run with arguments")

    # Parse arguments
    args = parser.parse_args()

    if args.command == "run":
        run_script(args)
    elif args.command == "exec":
        if not args.command:
            print("No command provided to execute. Use 'gpu-monitor exec <command>'.", file=sys.stderr)
        else:
            run_command(" ".join(args.command))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()