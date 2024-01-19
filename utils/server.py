from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llava.eval.run_llava import eval_model

app = FastAPI()

class LLAVARequest(BaseModel):
    model_path: str = './finetune/liuhaotian/llava-v1.5-13b'
    model_base: str
    image_file: str
    query: str
    conv_mode: str = None
    sep: str = ","
    temperature: float = 0.2
    top_p: float = None
    num_beams: int = 1
    max_new_tokens: int = 512

@app.post("/run_llava")
def run_llava(request: LLAVARequest):
    try:
        # Call the eval_model logic with the provided request data
        result = eval_model(request.dict())
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
