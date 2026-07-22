## Benchmark Results

| Metric | FP16 | 4-bit (NF4) |
|--------|------|-------------|
| Model | Qwen2.5-0.5B-Instruct | Qwen2.5-0.5B-Instruct |
| Quantization | None | bitsandbytes (NF4) |
| Peak VRAM | 0.94 GB | 0.47 GB |
| Generation Time | 5.79 sec | 10.19 sec |
| Generated Tokens | 200 | 200 |
| Throughput | 34.52 tokens/sec | 19.62 tokens/sec |

### Observation

The 4-bit quantized model reduced GPU memory usage by approximately 50% (0.94 GB → 0.47 GB). However, in this experiment, the FP16 model achieved higher throughput than the 4-bit version. This behavior is expected on some Windows systems and smaller models, where quantization overhead may outweigh the performance gains. The primary advantage observed was the significant reduction in memory usage while maintaining comparable output quality.