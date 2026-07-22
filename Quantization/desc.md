# Section 3 – Quantization Write-up

## Objective

The objective of this experiment was to evaluate the impact of 4-bit quantization on a locally running open-weight Large Language Model. The comparison focused on GPU memory usage, inference speed, and response quality.

## Model and Environment

- Model: Qwen2.5-0.5B-Instruct
- Framework: Hugging Face Transformers
- Quantization: bitsandbytes (NF4)
- Hardware: NVIDIA GeForce RTX 3050 (4 GB VRAM)

## Methodology

The model was benchmarked in two configurations:

1. FP16 (full precision)
2. 4-bit NF4 quantized model

Both configurations used the same prompt and generated 200 output tokens. GPU memory consumption and generation speed were recorded.

## Results

| Metric | FP16 | 4-bit |
|-------|------:|------:|
| Peak VRAM | 0.94 GB | 0.47 GB |
| Throughput | 34.52 tokens/sec | 19.62 tokens/sec |

## Analysis

The experiment showed that 4-bit quantization reduced GPU memory usage by approximately 50%, allowing the model to run comfortably on a GPU with only 4 GB of VRAM.

Although the FP16 model achieved higher throughput in this environment, the output quality remained comparable between both versions. The lower throughput is likely due to the overhead of 4-bit quantization on Windows and the relatively small model size, where quantization overhead can outweigh potential speed improvements.

## Conclusion

This experiment demonstrates that 4-bit quantization is an effective technique for reducing GPU memory requirements while maintaining response quality. It enables larger language models to run on consumer-grade hardware, making local deployment more practical even when maximum inference speed is not achieved. 