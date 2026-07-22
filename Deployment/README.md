# Electro Pi AI Engineer Technical Test

This project was completed as part of the Electro Pi AI Engineer Technical Assessment.

## Project Structure

```
.
├── app.py
├── model.py
├── benchmark.py
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── README.md
```

---

# Features

- FastAPI REST API
- Text Generation Endpoint
- Streaming Text Generation
- 4-bit Quantized Qwen2.5 Model
- Benchmark Script
- Docker Support

---

# Model

Model Used:

Qwen/Qwen2.5-0.5B-Instruct

Quantization:

- 4-bit NF4
- BitsAndBytesConfig
- Accelerate

---

# Installation

```bash
pip install -r requirements.txt
```

---

# Run

```bash
uvicorn app:app --reload
```

Server:

```
http://127.0.0.1:8000
```

Swagger:

```
http://127.0.0.1:8000/docs
```

---

# API

## GET /

Health Check

Response

```json
{
    "message":"API Running"
}
```

---

## POST /generate

Request

```json
{
    "prompt":"Explain Machine Learning",
    "max_new_tokens":100
}
```

Response

```json
{
    "response":"..."
}
```

---

## POST /stream

Request

```json
{
    "prompt":"Tell me a joke",
    "max_new_tokens":100
}
```

Returns a streamed plain-text response.

---

# Benchmark

Run

```bash
python benchmark.py
```

The benchmark compares:

- FP16
- 4-bit NF4 Quantization

Metrics:

- VRAM Usage
- Generation Time
- Tokens/Second

---

# Docker

Build

```bash
docker build -t qwen-api .
```

Run

```bash
docker run -p 8000:8000 qwen-api
```

---

# Dependencies

- FastAPI
- Transformers
- Torch
- Accelerate
- BitsAndBytes
- Uvicorn

---