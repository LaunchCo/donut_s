import argparse
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
from PIL import Image
import io
import re
import torch

# === CLI Arguments ===
parser = argparse.ArgumentParser(description="Inference server for Donut or Pix2Struct models")
parser.add_argument(
    "--framework",
    choices=["donut", "pix2struct"],
    required=True,
    help="Choose 'donut' or 'pix2struct' model family",
)
parser.add_argument(
    "--model-name",
    required=True,
    help=(
        "Model name: for donut choose one of [base, rvlcdip, cord-v2, docvqa, proto]; "
        "for pix2struct choose one of [base, docvqa, ocrvqa-large]"
    ),
)
args = parser.parse_args()

# === Model Selection ===
use_donut = args.framework == "donut"
model_id = None
if use_donut:
    donut_mapping = {
        "base": "naver-clova-ix/donut-base",
        "rvlcdip": "naver-clova-ix/donut-base-finetuned-rvlcdip",
        "cord-v2": "naver-clova-ix/donut-base-finetuned-cord-v2",
        "docvqa": "naver-clova-ix/donut-base-finetuned-docvqa",
        "proto": "naver-clova-ix/donut-proto",
    }
    model_id = donut_mapping.get(args.model_name)
    if not model_id:
        raise ValueError(f"Unknown Donut model name: '{args.model_name}'")
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    processor = DonutProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
else:
    p2s_mapping = {
        "base": "google/pix2struct-base",
        "docvqa": "google/pix2struct-docvqa-base",
        "ocrvqa-large": "google/pix2struct-ocrvqa-large",
    }
    model_id = p2s_mapping.get(args.model_name)
    if not model_id:
        raise ValueError(f"Unknown Pix2Struct model name: '{args.model_name}'")
    from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
    processor = Pix2StructProcessor.from_pretrained(model_id)
    model = Pix2StructForConditionalGeneration.from_pretrained(model_id)

# === Device Setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# === FastAPI Setup ===
app = FastAPI(title=f"{args.framework.capitalize()} Inference API ({args.model_name})")

# === Request / Response Schemas ===
class InferenceRequest(BaseModel):
    instruction: str = ""

class InferenceResponse(BaseModel):
    result: str

# === Inference Endpoint ===
@app.post("/inference", response_model=InferenceResponse)
async def inference(
    file: UploadFile = File(...),
    instruction: str = Form("<s_rvlcdip>"),
):
    # Validate file
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="An image file is required.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot open image file.")

    # Choose inference path
    if use_donut:
        # Donut expects decoder_input_ids + pixel_values
        task_prompt = instruction or "<s_rvlcdip>"
        decoder_input_ids = processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(device)
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=256,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
        sequence = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
        # Clean up tags
        text = re.sub(r"<.*?>", "", sequence).strip()
        # Try JSON parsing for Donut
        try:
            result = processor.token2json(sequence)
        except Exception:
            result = text
    else:
        # Pix2Struct inference
        inputs = processor(images=image, text=instruction, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            return_dict_in_generate=True,
        )
        sequence = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()
        result = sequence

    return InferenceResponse(result=result)

# === Run Server ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
