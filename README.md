# KernelBot

``` bash

# first step: generate the codes
export LLM_GATEWAY_KEY=`LLM_GATEWAY_KEY`
mkdir generated
python test_kernel_bench.py

# second step: refine the generated codes

python test_code_refine.py generated/36_36_RMSNorm_.py generated/36_36_RMSNorm_refine.py 
```