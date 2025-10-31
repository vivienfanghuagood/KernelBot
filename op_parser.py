import os
import json
from openai import OpenAI

client = OpenAI(
    base_url="https://llm-api.amd.com/OpenAI",
    api_key="dummy",
    default_headers={
        "Ocp-Apim-Subscription-Key": os.environ.get("LLM_GATEWAY_KEY"),
        "user": "hfang@amd.com"
    }
)

SYSTEM_PROMPT = """
You are a JSON-only structured information extractor. Based on the user's operator requirement, produce exactly one JSON object with these rules:

Required fields:
- "op_name": concise snake_case operator name; composite ops joined with underscores.
- "op_type": one of ["linear", "pointwise", "reduce"] based on dominant compute type.
- "backend": "nvidia" or "amd" inferred from description (default "nvidia").
- "op_description": one-sentence summary.
- "op_inputs": array of objects { name (required), dtype (optional), shape (optional) }.
- "op_outputs": array of objects { name (required), dtype (optional), shape (optional) }.

Strict output:
- Return only valid JSON (UTF-8, double quotes, no extra text).
- Omit unknown optional keys (do not use null or "unknown").
- For composite ops, join names with underscores and summarize full pipeline.
"""

def parse_op_signature(user_spec: str, model: str = "gpt-5"):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_spec}
        ],
    )

    content = resp.choices[0].message.content.strip()

    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        if content.startswith("```") and content.endswith("```"):
            content = "\n".join(content.splitlines()[1:-1])
        obj = json.loads(content)

    print(json.dumps(obj, indent=2, ensure_ascii=False))
    return obj

if __name__ == "__main__":
    USER_SPEC = "please implement a gemm + bias on AMD GPUs, input is float16, shape is [4096, 4096]"
    parse_op_signature(USER_SPEC)
