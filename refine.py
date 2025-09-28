
import sys
import os

TEMPLATE = """
Please refine the code, the origin code is as below: {}

Please refine the implementation of ModelNew, speed up the latency running on modern GPU(such as H100). 
Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! Do not Delete the main function.
"""

class KernelLLM:
    """
    A simple wrapper around the KernelLLM model for generating Triton kernels that allows easy
    instruction of the model and a streamed repl interface to interact with the model.
    """
    def __init__(
        self,
        backend: str = "gpt5" # gp5|claude
    ):
        
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
        return response_text
    
    def refine_triton(
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
        prompt = TEMPLATE.format(code)
        # triton_code = self.generate_call_claude_api(prompt, max_new_tokens)
        if self.backend == "gpt5":
            refine_code = self.generate_call_openai_api(prompt, max_new_tokens)
        elif self.backend == "claude":
            refine_code = self.generate_call_claude_api(prompt, max_new_tokens)
        
        return refine_code.strip("```python").strip("```")