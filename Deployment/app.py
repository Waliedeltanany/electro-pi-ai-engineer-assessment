from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from threading import Thread

from transformers import TextIteratorStreamer

from model import model, tokenizer

app = FastAPI(title="Qwen Deployment API")


class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200


@app.get("/")
def root():
    return {"message": "LLM Deployment API is running"}


@app.post("/generate")
def generate(request: PromptRequest):

    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=request.max_new_tokens
    )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return {
        "response": response
    }

@app.post("/stream")
def stream(request: PromptRequest):

    inputs = tokenizer(
        request.prompt,
        return_tensors="pt"
    ).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=request.max_new_tokens
    )

    thread = Thread(
        target=model.generate,
        kwargs=generation_kwargs
    )

    thread.start()

    return StreamingResponse(
        streamer,
        media_type="text/plain"
    )