# -*- coding: utf-8 -*-
from fastapi import FastAPI, Request, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


from transformers import LlamaTokenizerFast
app = FastAPI()

@app.on_event("startup")
def load_model():
    global model, tokenizer
    try:
        # ģ��·��
        model_name = r'D:\jinquanxin\LLM4EA\model\ministral-8b-instruct'
        tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True,legacy=False)

        # ����ģ�ͣ�ʹ���Զ������豸��ʹ�� float16 �������Դ�ռ��
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,torch_dtype=torch.float16,legacy=False)
        device = torch.device("cuda")
        model.to(device)

        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.post("/v1")
async def completions(request: Request):
    try:
        data = await request.json()
        messages = data.get("messages", [])
        max_tokens = data.get("max_tokens", 50)

        # �� messages �л�ȡ���һ���û������ prompt
        prompt = messages[-1]["content"] if messages else ""

        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")

        # ���������tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # �����ݶȼ��㣬�����Դ�����
        with torch.no_grad():
            # ʹ��ģ��������Ӧ
            outputs = model.generate(
                inputs["input_ids"],
                max_length=inputs["input_ids"].shape[1] + max_tokens,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                temperature=0.2
            )
        
        # �������
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {"choices": [{"text": result}]}
    except Exception as e:
        return {"error": str(e)}, 500

@app.get("/test")
async def test():
    return {"message": "Service is running!"}

models = [{"id": "local-llama2", "object": "model", "owned_by": "user"}]

@app.get("/v1/models")
async def list_models():
    return {"data": models}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
