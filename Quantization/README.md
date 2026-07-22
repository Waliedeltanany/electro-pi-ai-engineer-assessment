# LLM Quantization Benchmark

## Overview

This project benchmarks an open-weight Large Language Model (LLM) before and after 4-bit quantization using bitsandbytes (NF4). The goal is to compare memory usage, inference speed, and output quality.

## Model

- Model: Qwen2.5-0.5B-Instruct
- Framework: Hugging Face Transformers
- Quantization: bitsandbytes 4-bit (NF4)

## Project Structure

```
.
├── benchmark_fp16.py
├── benchmark_4bit.py
├── README.md
├── requirements.txt
├── results/
│   └── results.md
└── evidence/
    ├── fp16_output.png
    └── 4bit_output.png
```

## How to Run

### Create virtual environment

```bash
python -m venv .venv
```

### Activate

Windows

```bash
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run FP16 benchmark

```bash
python benchmark_fp16.py
```

### Run 4-bit benchmark

```bash
python benchmark_4bit.py
```

## Results

Benchmark results are available in:

```
results/results.md
```

Screenshots of the execution are available in:

```
evidence/
```