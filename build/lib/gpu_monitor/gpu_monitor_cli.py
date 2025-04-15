import argparse
import json
import os
from gpu_monitor.measure import measure_metrics

def run_script(script):
    # Placeholder function to run a script
    pass

def run_command(command):
    # Monitor the execution
    metrics, result, elapsed_time = measure_metrics(command)

    # Print the summary JSON to the console
    summary = {
        "execution_started_at": metrics["execution_started_at"],
        "execution_ended_at": metrics["execution_ended_at"],
        "summary": metrics.get("summary", {})
    }
    print(json.dumps(summary, indent=4))

    # Print command output
    if hasattr(result, "stdout") and result.stdout:
        print("\nCommand Output:\n", result.stdout)
    if hasattr(result, "stderr") and result.stderr:
        print("\nCommand Errors:\n", result.stderr)

def main():
    parser = argparse.ArgumentParser(description="Monitor resource usage of a command or script.")
    parser.add_argument(
        "command", 
        nargs=argparse.REMAINDER, 
        help="The command or script to monitor. Example: python3 script.py"
    )
    args = parser.parse_args()

    if args.command:
        run_command(" ".join(args.command))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
