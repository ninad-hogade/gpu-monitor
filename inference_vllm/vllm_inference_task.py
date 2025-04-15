import requests
import json
import sys

DEFAULT_QUERIES = [
    "Explain quantum mechanics. Write a 5 page story.",
    "What is the theory of relativity? Write a 3 page essay.",
    "Describe the process of photosynthesis. Write a 4 page thesis."
]

def main(model_name):
    response = requests.post(
        'http://localhost:8000/v1/completions',
        json={
            'model': model_name,
            'prompt': DEFAULT_QUERIES,
            'max_tokens': 500,
            'temperature': 0.8,
            'top_p': 0.95,
            'logprobs': 5
        }
    )
    #print(json.dumps(response.json(), indent=4))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python vllm_inference_task.py <model_name>")
        sys.exit(1)
    model_name = sys.argv[1]
    main(model_name)
