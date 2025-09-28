from kernel import KernelLLM

# Initialize the model
# Define your PyTorch module
pytorch_code = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Hinge Loss for binary classification tasks.
    """
    def __init__(self):
        super(Model, self).__init__()
        
    def forward(self, predictions, targets):
        return torch.mean(torch.clamp(1 - predictions * targets, min=0))

batch_size = 128
input_shape = (1,)

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randint(0, 2, (batch_size, 1)).float() * 2 - 1]

def get_init_inputs():
    return []
'''

# Generate optimized Triton code


def generate_random_filename(base_dir="./generated/", extension=".py"):
    import random
    import string
    filename = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    return base_dir + filename + extension

if __name__ == "__main__":
    import os
    
    model = KernelLLM(backend="gpt5")
    optimized_code = model.generate_triton(pytorch_code, max_new_tokens=8192)
    random_file = generate_random_filename()
    with open(random_file, "w") as writer:
        writer.write(optimized_code)
    
    os.system(f"python3 {random_file}")
