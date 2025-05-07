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
    choices=["donut", "pix2struct", "qwen"],
    required=True,
    help="Choose 'donut', 'pix2struct' or 'qwen' model family",
)
parser.add_argument(
    "--model-name",
    required=True,
    help=(
        "Model name: for donut choose one of [base, rvlcdip, cord-v2, docvqa, proto]; "
        "for pix2struct choose one of [base, large docvqa, docvqa-large, ocrvqa-large, ocrvqa-base, infographics-vqa-base, infographics-vqa-large, chartqa-base, widget-captioning-base, widget-captioning-large, ai2d-base, ai2d-large, screen2words-base, screen2words-large, textcaps-base, textcaps-large]"
    ),
)
args = parser.parse_args()

# === Model Selection ===
use_donut = args.framework == "donut"
use_pix2struct = args.framework == "pix2struct"
use_qwen = args.framework == "qwen"
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
elif use_pix2struct:
    p2s_mapping = {
        "base": "google/pix2struct-base",
        "large": "google/pix2struct-large",
        "screen2words-base": "google/pix2struct-screen2words-base",
        "screen2words-large": "google/pix2struct-screen2words-large",
        "textcaps-base": "google/pix2struct-textcaps-base",
        "textcaps-large": "google/pix2struct-textcaps-large",
        "widget-captioning-base": "google/pix2struct-widget-captioning-base",
        "widget-captioning-large": "google/pix2struct-widget-captioning-large",
        "ai2d-base": "google/pix2struct-ai2d-base",
        "ai2d-large": "google/pix2struct-ai2d-large",
        "chartqa-base": "google/pix2struct-chartqa-base",
        "docvqa": "google/pix2struct-docvqa-base",
        "docvqa-large": "google/pix2struct-docvqa-large",
        "ocrvqa-base": "google/pix2struct-ocrvqa-base",
        "ocrvqa-large": "google/pix2struct-ocrvqa-large",
        "infographics-vqa-base": "google/pix2struct-infographics-vqa-base",
        "infographics-vqa-large": "google/pix2struct-infographics-vqa-large",
    }
    model_id = p2s_mapping.get(args.model_name)
    if not model_id:
        raise ValueError(f"Unknown Pix2Struct model name: '{args.model_name}'")
    from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration

    processor = Pix2StructProcessor.from_pretrained(model_id)
    model = Pix2StructForConditionalGeneration.from_pretrained(model_id)
elif use_qwen:
    if args.model_name != "vl-3b":
        raise ValueError("Currently only Qwen2.5-VL-3B-Instruct is supported.")

    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    from transformers import AutoProcessor, AutoModelForVision2Seq

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(model_id)
    model = model.to(dtype=torch.float32)

else:
    raise ValueError(f"Unknown framework: '{args.framework}'")

# === Device Setup ===
device = "cpu"
model.to(device)

# === FastAPI Setup ===
pretty_name = {
    "donut": "Donut",
    "pix2struct": "Pix2Struct",
    "qwen": "Qwen2.5-VL",
}.get(args.framework, args.framework)

app = FastAPI(title=f"{pretty_name} Inference API ({args.model_name})")


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
        text = re.sub(r"<.*?>", "", sequence).strip()
        try:
            result = processor.token2json(sequence)
        except Exception:
            result = text
    elif use_pix2struct:
        # Pix2Struct inference
        inputs = processor(images=image, text=instruction, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            return_dict_in_generate=True,
        )
        sequence = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()
        print(sequence)
        result = sequence
    elif use_qwen:
        inputs = processor(images=image, text=instruction, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            return_dict_in_generate=True,
        )
        result = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    else:
        raise ValueError(f"Unknown framework: '{args.framework}'")

    return InferenceResponse(result=result)


# === Run Server ===
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
