import os
import json
import subprocess
import time
import socket

QUERY = "Explain quantum mechanics. Write a 5 page story."

def llm_test():
    print(f"Starting benchmark...")

    start_time = time.time()
    process = subprocess.Popen(
        ["ollama", "run", "llama3.2"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate(input=QUERY)
    elapsed_time = time.time() - start_time

    response = stdout if stdout else ""
    token_count = len(response.split())
    tokens_per_sec = token_count / elapsed_time if elapsed_time > 0 else 0

    # Print response preview
    response_preview = " ".join(response.split()[:50])
    #print(f"Response preview: {response_preview}")
    #print(f"Tokens per second: {tokens_per_sec:.2f}")

    # Print summary JSON
    summary = {
        "host_name": socket.gethostname(),
        #"elapsed_time_s": elapsed_time,
        "token_count": token_count,
        #"tokens_per_sec": tokens_per_sec,
        "response_preview": response_preview
    }
    return summary, stdout, stderr

if __name__ == "__main__":
    metrics, stdout, stderr = llm_test()
    metrics_json = json.dumps(metrics, indent=4)
    print(metrics_json)
    if stderr:
        print("Errors:", stderr)