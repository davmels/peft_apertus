from vllm import LLM, SamplingParams

# Define the model ID
# MODEL_ID = "swiss-ai/Apertus-8B-Instruct-2509"
MODEL_ID = "swiss-ai/Apertus-70B-Instruct-2509"

def main():
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The capital of Switzerland is",
        "Explain the concept of Swiss neutrality.",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

    # Create an LLM.
    # vLLM automatically downloads the model from HuggingFace if not present locally.
    print(f"Loading model: {MODEL_ID}...")
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=4,
    )

    # Generate texts from the prompts.
    print("Generating responses...")
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated text: {generated_text!r}")
        print("-" * 50)

if __name__ == "__main__":
    main()
