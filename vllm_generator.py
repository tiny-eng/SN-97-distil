import json
import time
import requests

ENDPOINT = "http://38.102.125.144:8949/v1/chat/completions"
MODEL = "./king_bitto"
OUTPUT_FILE = "answers.json"

def ask_vllm(prompt: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 1.0,
        "top_p": 1.0,
    }

    start = time.time()

    response = requests.post(
        ENDPOINT,
        json=payload,
        timeout=120,
    )

    elapsed = time.time() - start
    response.raise_for_status()

    data = response.json()
    answer = data["choices"][0]["message"]["content"]

    return {
        "prompt": prompt,
        "answer": answer,
        "elapsed": round(elapsed, 3),
        "raw": data
    }

def main():
    prompts = [
        "Write a response about software documentation. Your response must contain no commas.",
        "Write a response about remote work. Your response must use only uppercase letters.",
        "Write a response about weekend hiking. Your response must never use the word 'always'."
    ]

    results = []

    for i, prompt in enumerate(prompts, start=1):
        print(f"[{i}/{len(prompts)}] Asking VLLM...")

        try:
            result = ask_vllm(prompt)
            results.append(result)
            print(result["answer"])
        
        except Exception as e:
            results.append({
                "prompt": prompt,
                "answer": None,
                "error": f"{type(e).__name__}: {str(e)}",
            })
            print(f"Error: {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()