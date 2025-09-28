import sys
import os
import torch


HF_MODEL = "/root/KernelLLM"
REPL_INSTRUCTIONS = """
You can paste or write your nn.Module code below (and finish with Ctrl+D).
The model will try to optimize it with Triton kernels.
Make sure that you provide a `get_inputs()` and `get_init_inputs()` function such that your model can be run like this
    args, kwargs = get_inputs()
    model = Model(*args, **kwargs)
    out = model(get_inputs())
>>>"""
DEFAULT_MODEL_CODE = """
import torch
import torch.nn as nn
class Model(nn.Module):
    \"\"\"
    A model that computes Hinge Loss for binary classification tasks.
    Parameters:
        None
    \"\"\"
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, predictions, targets):
        return torch.mean(torch.clamp(1 - predictions * targets, min=0))
batch_size = 128
input_shape = (1,)
dim = 1
def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randint(0, 2, (batch_size, 1)).float() * 2 - 1]
def get_init_inputs():
    return []
"""
PROMPT_TEMPLATE = """
<|begin_of_text|>You write custom Triton kernels to replace the pytorch operators in the given architecture to get speedups.
You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom Triton kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.

Here's an example to show you the syntax of inline embedding custom operators from the Triton DSL in torch: The example given architecture is:
```
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, a, b):
        return a + b

def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.randn(1, 128).cuda()
    b = torch.randn(1, 128).cuda()
    return [a, b]

def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []
```
The example new arch with custom Triton kernels looks like this:
```
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,  # Pointer to first input
    y_ptr,  # Pointer to second input
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements in input/output
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    # Perform the elementwise addition
    out = x + y
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_add(x: torch.Tensor, y: torch.Tensor):
    \"\"\"
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    \"\"\"
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    y = y.contiguous()
    # Prepare output tensor
    out = torch.empty_like(x)
    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size
    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    # Launch the Triton kernel
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, a, b):
        # Instead of "return a + b", call our Triton-based addition
        return triton_add(a, b)
```
You are given the following architecture:
```
{}
```
Optimize the architecture named Model with custom Triton kernels! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code!
"""
benchmark_code = """
def test_latency(model1, model2, inputs, warmup=10, iters=100):
    import time
    model1.eval()
    model2.eval()
    # warm-up
    for _ in range(warmup):
        out1 = model1(*inputs)
        out2 = model2(*inputs)
    # sync CUDA before timing
    torch.cuda.synchronize()
    start1 = time.time()
    for _ in range(iters):
        _ = model1(*inputs)
    torch.cuda.synchronize()
    end1 = time.time()
    avg_time1 = (end1 - start1) * 1000 / iters  # ms
    torch.cuda.synchronize()
    start2 = time.time()
    for _ in range(iters):
        _ = model2(*inputs)
    torch.cuda.synchronize()
    end2 = time.time()
    avg_time2 = (end2 - start2) * 1000 / iters  # ms
    print(f"Model1 (PyTorch) avg latency: {avg_time1:.4f} ms")
    print(f"Model2 (Triton)  avg latency: {avg_time2:.4f} ms")

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    inputs = get_inputs()
    inputs = [i.to("cuda") for i in inputs]
    model1 = Model(*get_init_inputs()).cuda()
    model2 = ModelNew(*get_init_inputs()).cuda()
    test_latency(model1, model2, inputs)
"""

class KernelLLM:
    """
    A simple wrapper around the KernelLLM model for generating Triton kernels that allows easy
    instruction of the model and a streamed repl interface to interact with the model.
    """
    def __init__(
        self,
        model_name: str = HF_MODEL,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        backend: str = "gpt5" # gp5|claude
    ):
        self.model_name = model_name
        self.device = device
        
        import openai
        from anthropic import Anthropic
        self.claude_client = Anthropic(
            base_url="https://llm-api.amd.com/Anthropic",
            api_key="dummy",
            default_headers={
                "Ocp-Apim-Subscription-Key": os.environ.get("LLM_GATEWAY_KEY"),
                "user": "hfang@amd.com",
            }
        )
        self.openai_client = openai.OpenAI(
            base_url="https://llm-api.amd.com/OpenAI",
            api_key="dummy",
            default_headers={
                "Ocp-Apim-Subscription-Key": os.environ.get("LLM_GATEWAY_KEY"),
            }
        )
        self.backend = backend
        
        # from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.model_name, torch_dtype=torch.float16
        # )
        # self.model.to(self.device)
    def generate_raw(
        self, prompt: str, temperature: float = 0.6, max_new_tokens: int = 2048
    ) -> str:
        """
        Generate text from the model using the given prompt verbatim.
        Args:
            prompt (str): The prompt to generate text from.
            temperature (float): The temperature to use for sampling.
            max_new_tokens (int): The maximum length of the generated text.
        Returns:
            str: The generated text.
        """
        inputs = self.tokenizer([prompt], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=0,
            top_p=0.95,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text[len(prompt) :].strip()
    def stream_raw(self, prompt: str, max_new_tokens: int = 2048):
        """
        Stream and print text from the model using the given prompt verbatim.
        Args:
            prompt (str): The prompt to generate text from.
            max_new_tokens (int): The maximum length of the generated text.
        """
        inputs = self.tokenizer([prompt], return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        streamer = TextStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        self.model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=0,
            top_p=0.95,
            temperature=0.6,
            eos_token_id=self.tokenizer.eos_token_id,
        )
    
    def generate_call_claude_api(self, prompt: str, max_new_tokens: int = 8192):
        with self.claude_client.messages.stream(
            model="claude-opus-4-1",
            max_tokens=max_new_tokens,
            temperature=0.7,
            messages=[
                {"role": "user", "content": prompt}
            ],
        ) as stream:
            result = []
            for text in stream.text_stream:
                print(text, end="", flush=True)
                result.append(text)
            return "".join(result)
    
    def generate_call_openai_api(self, prompt: str, max_new_tokens: int = 8192):
        response_text = ""
        with self.openai_client.responses.stream(
            model="gpt-5",
            input=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    # Stream partial text
                    print(event.delta, end="", flush=True)
                    response_text += event.delta
                elif event.type == "response.completed":
                    print()  # newline after full stream
        # Full collected output text
        # print("Full response:\n", response_text)
        return response_text
    
    def generate_triton(
        self, code: str, temperature: float = 0.6, max_new_tokens: int = 2048
    ) -> str:
        """
        Generate Triton for the given torch module.
        The input code should be a python module that contains a torch Model(nn.Module) class and
        `get_inputs()` and `get_init_inputs()` functions such that your model can be run like this
            ```
            args, kwargs = get_inputs()
            model = Model(*args, **kwargs)
            out = model(get_inputs())
            ```
        Args:
            code (str): The torch code to generate Triton for.
            temperature (float): The temperature to use for sampling.
            max_new_tokens (int): The maximum length of the generated text.
        Returns:
            str: The generated Triton module.
        """
        prompt = PROMPT_TEMPLATE.format(code)
        # triton_code = self.generate_call_claude_api(prompt, max_new_tokens)
        if self.backend == "gpt5":
            triton_code = self.generate_call_openai_api(prompt, max_new_tokens)
        elif self.backend == "claude":
            triton_code = self.generate_call_claude_api(prompt, max_new_tokens)
        
        return code + "\n" + triton_code.strip("```python").strip("```") + benchmark_code
    def run_repl(self):
        """
        Run a REPL for the model. The user can input code and the model will try to optimize it with Triton kernels.
        """
        while True:
            try:
                print(REPL_INSTRUCTIONS)
                code = sys.stdin.read().strip()
                if code.lower() == "exit":
                    return
            except EOFError:
                pass
            if not code:
                print(f"Using default prompt:\n{DEFAULT_MODEL_CODE}\n")
                code = DEFAULT_MODEL_CODE
            prompt = PROMPT_TEMPLATE.format(DEFAULT_MODEL_CODE)
            try:
                self.stream_raw(prompt)
            except KeyboardInterrupt:
                print("Aborting...")

if __name__ == "__main__":
    kernel_llm = KernelLLM()
    kernel_llm.run_repl()
