# Electro Pi AI Engineer Technical Assessment Write-up

## Overview

This project implements a lightweight LLM inference service using FastAPI and Hugging Face Transformers.

The solution focuses on efficient deployment through model quantization while exposing both standard and streaming inference APIs.

---

# Part 1 — Model Selection

Model:

Qwen2.5-0.5B-Instruct

Reasons:

- Small model size
- Fast inference
- Good instruction-following capability
- Suitable for CPU/GPU deployment
- Supported by Hugging Face Transformers

---

# Part 2 — Quantization

The model was quantized using BitsAndBytes 4-bit NF4.

Configuration:

- load_in_4bit=True
- nf4 quantization
- float16 compute dtype

Benefits:

- Reduced VRAM usage
- Lower memory footprint
- Faster deployment
- Minimal quality degradation

---

# Part 3 — Benchmark

Two configurations were benchmarked.

## FP16

VRAM:
0.94 GB

Generation Time:
5.79 sec

Speed:
34.52 tokens/sec

---

## 4-bit NF4

VRAM:
0.47 GB

Generation Time:
10.19 sec

Speed:
19.62 tokens/sec

---

Observation:

The quantized model reduced memory usage by approximately 50%.

Although FP16 achieved higher throughput on the development machine, the quantized model is more deployment-friendly due to significantly lower memory requirements.

---

# Part 4 — FastAPI Service

The application exposes three endpoints.

GET /

Health check endpoint.

POST /generate

Returns the complete generated response.

POST /stream

Streams generated tokens progressively using TextIteratorStreamer.

Streaming is implemented using a background thread to allow token generation while simultaneously sending tokens to the client.

---

# Docker

A Dockerfile is included to containerize the application.

The container installs all required dependencies and serves the FastAPI application using Uvicorn.

---

# Technologies

- Python
- FastAPI
- Hugging Face Transformers
- BitsAndBytes
- Accelerate
- PyTorch
- Docker

---

# Conclusion

The project demonstrates an end-to-end lightweight LLM deployment workflow including:

- Model loading
- Quantization
- Benchmarking
- REST API
- Streaming generation
- Docker containerization

The implementation prioritizes simplicity, reproducibility, and efficient inference suitable for edge and resource-constrained environments.