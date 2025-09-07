from vllm import LLM, SamplingParams

model_path = "/path/to/your/model"

llm = LLM(model=model_path, trust_remote_code=True)

prompts = ["San Francisco is a"]
sampling_params = SamplingParams(temperature=0, max_tokens=100)


outputs = llm.generate(prompts, sampling_params=sampling_params)

for output in outputs:
    generated_text = output.outputs[0].text
    print(generated_text)