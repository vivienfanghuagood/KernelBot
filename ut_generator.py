import os
import json
import subprocess
from pathlib import Path
from openai import OpenAI

MODEL = "gpt-5"
OUT_FILE = Path("dry_run_test.py")

SYSTEM_PROMPT = """
Generate a single Python script that:
1) Contains ONLY Python code (no Markdown, no explanations).
2) Inlines the given JSON spec as SPEC = {...}.
3) Uses PyTorch to create random input tensors based on SPEC["op_inputs"] (respect shape and dtype if provided).
4) Implements a simple forward pass for the operator described in SPEC:
5) Prints:
   - "Running dry-run for {op_name}"
   - Input shapes and dtypes
   - Output shape and dtype
6) Runs on CPU by default; if torch.cuda.is_available() and backend is nvidia/amd, use CUDA.
7) No unit test framework, no correctness checks—just a forward dry-run.

Output only valid Python code.
""".strip()

OP_SPEC = {
    "op_name": "gemm_bias",
    "op_type": "linear",
    "backend": "amd",
    "op_description": "GEMM followed by bias add on AMD GPUs.",
    "op_inputs": [
        {"name": "A", "dtype": "float16", "shape": [128, 128]},
        {"name": "B", "dtype": "float16", "shape": [128, 128]},
        {"name": "bias", "dtype": "float16"}
    ],
    "op_outputs": [
        {"name": "C", "dtype": "float16", "shape": [128, 128]}
    ]
}

def main():
    client = OpenAI(
        base_url="https://llm-api.amd.com/OpenAI",
        api_key="dummy",
        default_headers={
            "Ocp-Apim-Subscription-Key": os.environ.get("LLM_GATEWAY_KEY"),
            "user": "hfang@amd.com"
        }
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(OP_SPEC)}
        ],
    )

    code = resp.choices[0].message.content.strip()
    OUT_FILE.write_text(code, encoding="utf-8")
    print(f"[INFO] Generated dry-run test: {OUT_FILE.resolve()}")

    print("[INFO] Running dry-run...")
    completed = subprocess.run(["python", str(OUT_FILE)], text=True, capture_output=True)
    print(completed.stdout)

if __name__ == "__main__":
    main()