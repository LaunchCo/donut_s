from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Literal
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)
from PIL import Image
import io
import torch

app = FastAPI(title="Broad Inference Engine API")

# === Device Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Model Initialization ===
# Donut (OCR & structured extraction)
DONUT_MODEL_ID = "naver-clova-ix/donut-base"
processor = DonutProcessor.from_pretrained(DONUT_MODEL_ID)
vision_model = VisionEncoderDecoderModel.from_pretrained(DONUT_MODEL_ID)
vision_model.config.max_length = 1024
vision_model.config.num_beams = 4
vision_model.to(device)

# Text generation model (causal LM)
TEXT_MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
text_model = AutoModelForCausalLM.from_pretrained(
    TEXT_MODEL_ID,
    device_map=None,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
)
text_model.to(device)

# Generation config
gen_config = GenerationConfig(
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9
)

# === Request / Response Schemas ===
class InferenceRequest(BaseModel):
    mode: Literal["donut", "text"]
    instruction: str
    output_format: Optional[str] = Query(
        None,
        description="Hint for the shape of the output (e.g., 'json', 'plain')"
    )

class InferenceResponse(BaseModel):
    result: str

# === Endpoints ===
@app.post("/inference", response_model=InferenceResponse)
async def inference(
    req: InferenceRequest,
    file: Optional[UploadFile] = File(None)
):
    # --- Donut Mode ---
    if req.mode == "donut":
        if not file or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Image file required for 'donut' mode.")
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Cannot open image file.")
        # Build prompt
        task_prompt = f"<s_docvqa><s_question>{req.instruction}</s_question><s_answer>"
        # Prepare inputs and move to device
        inputs = processor(image, task_prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Generate
        outputs = vision_model.generate(**inputs)
        result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return InferenceResponse(result=result)

    # --- Text Mode ---
    elif req.mode == "text":
        # Tokenize and move to device
        inputs = tokenizer(
            req.instruction,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Generate with config
generate_kwargs = gen_config.to_dict()
        outputs = text_model.generate(
            **inputs,
            **generate_kwargs
        )
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return InferenceResponse(result=result)

    else:
        raise HTTPException(status_code=400, detail="Unsupported mode. Choose 'donut' or 'text'.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
