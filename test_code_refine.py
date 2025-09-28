
if __name__ == "__main__":
    import sys
    import os
    from refine import KernelLLM

    llm = KernelLLM(backend="gpt5")
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print("before optimization")
    os.system(f"python3 {input_file}")
    input_code = open(input_file).read()
    output_code = llm.refine_triton(input_code, max_new_tokens=32000)

    with open(output_file, "w") as writer:
        writer.write(output_code)
    
    print("after optimization")
    os.system(f"python3 {output_file}")


