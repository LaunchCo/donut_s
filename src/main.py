from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import re
import io
import torch

app = FastAPI(title="Donut Inference API")

# === Device Setup ===
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# === Request / Response Schemas ===
class InferenceRequest(BaseModel):
    # Optional instruction for the Donut model; defaulting to the task prompt from the docs.
    instruction: str = "<s_rvlcdip>"

class InferenceResponse(BaseModel):
    result: str

# === Endpoint ===
@app.post("/inference", response_model=InferenceResponse)
async def inference(
    file: UploadFile = File(...),
    instruction: str = Form("<s_rvlcdip>")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="An image file is required.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot open image file.")

    # Use the provided instruction or the default "<s_rvlcdip>".
    task_prompt = instruction

    # Prepare decoder input IDs from task prompt.
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    # Process the image.
    pixel_values = processor(image, return_tensors="pt").pixel_values
    print("Max token ID in decoder input:", decoder_input_ids.max().item())
    print("Decoder vocab size:", model.config.decoder.vocab_size)
    # Generate output using the model.
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids,
        max_length=model.decoder.config.max_position_embeddings,
        max_new_tokens=256,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    # Decode and process the output sequence.
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    # Remove the first task start token using regex.
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
    print("Raw output:", repr(sequence))
    result = processor.token2json(sequence)
    print(str(result))

    return InferenceResponse(result=str(result))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)