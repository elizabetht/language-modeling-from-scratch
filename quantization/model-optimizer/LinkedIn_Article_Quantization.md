# Shrinking Giants: A Hands-On Guide to Quantizing Llama 3.1 8B with NVIDIA ModelOpt

*Reducing a 16GB LLM to just 6GB using FP8 and NVFP4 quantization*

---

## The Problem: LLMs Are Simply Too Big

Anyone who has tried running large language models locally has likely hit the same wall: **memory constraints**. A Llama 3.1 8B model in bfloat16 precision weighs in at approximately **16 GB**—more than most consumer GPUs can handle comfortably.

But here's the interesting question: does a model really need all that precision? The answer, increasingly, is **no**.

This is where **quantization** comes in—the art of reducing model precision while preserving performance. This article walks through a hands-on experience quantizing Llama 3.1 8B Instruct using NVIDIA's ModelOpt toolkit with two precision formats: **FP8** and **NVFP4**.

---

## What is Quantization, Really?

At its core, quantization is about representing numbers with fewer bits. Neural network weights are typically stored as 32-bit or 16-bit floating-point numbers. Quantization poses a simple question: *What if 8 bits were enough? Or even 4 bits?*

| Precision | Bits per Weight | Llama 3.1 8B Size |
|-----------|-----------------|-------------------|
| BFloat16  | 16 bits         | ~16 GB            |
| FP8       | 8 bits          | ~9 GB             |
| NVFP4     | 4 bits          | ~6 GB             |

### How Post-Training Quantization (PTQ) Works

The key technique enabling this compression is **Post-Training Quantization (PTQ)**. Here's what actually happens:

1. **Range Analysis**: The quantization algorithm analyzes the distribution of weight values in each layer. Neural network weights typically cluster around zero with varying spreads—understanding this distribution is critical.

2. **Scale Factor Calculation**: For each tensor (or group of weights), the algorithm computes a **scale factor** that maps the full-precision values to the reduced-precision range. For example, FP8 can represent values roughly between -448 and +448, so if a weight tensor has values between -0.5 and +0.5, the scale factor determines how to map that range optimally.

3. **Calibration Pass**: A small dataset (in this case, 128 samples from CNN/DailyMail) is fed through the model. During this forward pass, the algorithm observes activation patterns and determines optimal quantization parameters that minimize information loss for real inputs—not just random data.

4. **Weight Conversion**: Finally, each weight is divided by its scale factor and rounded to the nearest representable value in the target format. The scale factors are stored alongside the quantized weights for later dequantization during inference.

The result: weights that were 16 bits each are now 8 or 4 bits, with scale factors enabling approximate reconstruction when needed.

---

## The Setup: Tools of the Trade

This experiment used the following stack:

- **Model**: `meta-llama/Llama-3.1-8B-Instruct`
- **Framework**: PyTorch 2.9.0 with CUDA 13.0
- **Quantization Library**: NVIDIA ModelOpt (`nvidia-modelopt`)
- **Calibration Dataset**: CNN/DailyMail (128 samples)
- **Hardware**: NVIDIA Blackwell GPU

```python
import torch
import modelopt.torch.quantization as mtq
from modelopt.torch.utils.dataset_utils import create_forward_loop, get_dataset_dataloader

# Load the model in bfloat16
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
)
```

---

## Experiment 1: FP8 Quantization

FP8 (8-bit floating point) represents the "safe" choice—offering a solid balance between compression and accuracy. It's the go-to option for teams that want meaningful size reductions without taking risks.

### The Process

The quantization workflow starts with creating a calibration dataloader. This feeds representative samples through the model, helping ModelOpt determine optimal scaling factors:

```python
# Create calibration dataloader
dataloader = get_dataset_dataloader(
    dataset_name="cnn_dailymail",
    tokenizer=tokenizer,
    batch_size=8,
    num_samples=128,
    device="cuda",
)

forward_loop = create_forward_loop(dataloader=dataloader)

# Quantize with FP8
quant_config = mtq.FP8_DEFAULT_CFG
model = mtq.quantize(model, quant_config, forward_loop=forward_loop)
```

### Results

| Metric | Before | After (FP8) |
|--------|--------|-------------|
| Model Size | ~16 GB | ~9 GB |
| Compression | 1x | 1.78x |
| Inference | ✅ Works | ✅ Works |

The quantized model continues to generate coherent, high-quality text:

```
Input: "Hello, my name is"
Output: "Hello, my name is Sarah and I'm a..."
```

---

## Experiment 2: NVFP4 Quantization—Pushing the Limits

NVFP4 is NVIDIA's 4-bit floating-point format, designed specifically for their latest hardware. This is where quantization gets truly aggressive—and the results are impressive.

### Key Difference: Double Scaling

Unlike FP8, NVFP4 employs a **block-wise quantization with two levels of scaling factors**:

- `weight_scale` — The primary scale factor
- `weight_scale_2` — A secondary scale factor for finer granularity
- `group_size: 16` — Weights are quantized in groups of 16 elements

This dual-scaling approach is what allows NVFP4 to maintain reasonable accuracy despite squeezing weights down to just 4 bits.

```python
# Quantize with NVFP4
quant_config = mtq.NVFP4_DEFAULT_CFG
model = mtq.quantize(model, quant_config, forward_loop=forward_loop)
```

### Results

| Metric | Before | After (NVFP4) |
|--------|--------|---------------|
| Model Size | ~16 GB | ~6 GB |
| Compression | 1x | 2.67x |
| Parameters | 8B | 4.5B (effective) |

**That's a 62.5% reduction in model size**—a game-changer for edge deployment and cost-conscious inference.

### Accuracy Validation with LightEval

To quantify the actual accuracy impact, **LightEval** from HuggingFace provides a lightweight, reproducible evaluation framework. Unlike heavier benchmarking suites, LightEval runs efficiently and integrates seamlessly into development workflows.

Running the IFEval task through LightEval on both the baseline (BFloat16) and NVFP4-quantized models reveals the true cost of compression:

| Metric | Baseline | NVFP4 | Delta |
|--------|----------|-------|-------|
| Prompt-level Accuracy (Strict) | 73.57% | 70.43% | -3.14% |
| Prompt-level Accuracy (Loose) | 78.56% | 75.05% | -3.51% |
| Instruction-level Accuracy (Strict) | 81.53% | 79.14% | -2.40% |
| Instruction-level Accuracy (Loose) | 85.37% | 82.61% | -2.76% |

**What do these metrics mean?**

- **Prompt-level Accuracy**: Measures whether the model followed *all* instructions in a given prompt correctly. Strict scoring requires exact compliance; loose scoring allows minor deviations.
- **Instruction-level Accuracy**: Evaluates each individual instruction separately, providing a more granular view of instruction-following capability.

**The verdict**: NVFP4 quantization costs roughly **2.4–3.5% accuracy** depending on the metric—a modest price for cutting model size by 62%. For most production use cases, this trade-off is highly favorable.

### How Model Size is Calculated

The model size figures reported throughout this article come from a custom helper function that accurately measures memory footprint—including quantized tensors with their scale factors.

```python
from quantization_theory_helper import compute_module_sizes

module_size = compute_module_sizes(model)
print(f"The model size is {module_size[''] * 1e-9} GB")
```

**How it works under the hood:**

The `compute_module_sizes` function traverses every parameter and buffer in the model, calculating total bytes:

```python
def compute_module_sizes(model):
    """Compute the size of each submodule of a given model."""
    module_sizes = defaultdict(int)
    for name, tensor in named_module_tensors(model, recurse=True):
        size = tensor.numel() * dtype_byte_size(tensor.dtype)
        # Aggregate sizes hierarchically
        name_parts = name.split(".")
        for idx in range(len(name_parts) + 1):
            module_sizes[".".join(name_parts[:idx])] += size
    return module_sizes
```

**Key details:**

1. **`named_module_tensors`**: A custom iterator that handles quantized tensors specially. For quantized weights, it extracts both the packed data (`_data`) and scale factors (`_scale`) separately—ensuring nothing is missed.

2. **`dtype_byte_size`**: Calculates bytes per element based on dtype. It parses the dtype string to extract bit width (e.g., `float16` → 16 bits → 2 bytes) and handles special cases like FP8 (1 byte) and bool (1/8 byte).

3. **Hierarchical aggregation**: Sizes are accumulated at every level of the module hierarchy, so calling `module_sizes['model.layers.0']` returns the size of just that layer, while `module_sizes['']` returns the total model size.

This approach provides accurate measurements even for models with mixed precision or custom quantization schemes where simply counting parameters would give misleading results.

---

## Under the Hood: How Are Weights Actually Stored?

One surprising discovery: when inspecting the exported model files, both FP8 and NVFP4 weights appear as standard tensor types (like uint8). What's going on?

**Neither FP4 nor FP8 are native PyTorch dtypes.**

The actual quantized values get *packed* into standard containers. The system works through three key mechanisms:

1. **Metadata** (`hf_quant_config.json`) — Signals to the inference engine which quantization scheme was used
2. **Scale tensors** — Store the dequantization parameters needed to recover approximate original values
3. **Runtime unpacking** — The inference engine dequantizes weights on-the-fly during computation

```json
// NVFP4 config example
{
    "quantization": {
        "quant_algo": "NVFP4",
        "group_size": 16,
        "exclude_modules": ["lm_head"]
    }
}
```

---

## Measuring Accuracy: IFEval Testing in Deploy Workflows

Size reduction is fantastic, but the critical question remains: **how much accuracy was lost?** This is where rigorous testing becomes essential.

For instruction-following models like Llama 3.1 Instruct, integrating **IFEval (Instruction Following Evaluation)** into the deployment workflow provides meaningful accuracy metrics. IFEval is a benchmark that tests whether models can accurately follow explicit constraints in instructions—tasks like "write exactly 3 paragraphs" or "include the word 'quantum' at least twice."

### Why IFEval?

1. **Instruction-specific** — Unlike perplexity or BLEU scores, IFEval tests the actual instruction-following capability that matters for chat models
2. **Quantifiable** — Provides clear pass/fail metrics on constraint adherence
3. **CI/CD friendly** — Integrates smoothly into automated deployment pipelines

### Implementation in Deploy Workflow

The `lm-evaluation-harness` library makes running IFEval straightforward:

```python
import lm_eval
from lm_eval.models.huggingface import HFLM

def evaluate_quantized_model(model_path, model_name):
    """Run IFEval benchmark on quantized model"""
    
    # Load model for evaluation
    lm = HFLM(
        pretrained=model_path,
        backend='causal',
        trust_remote_code=True
    )
    
    # Run IFEval benchmark
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["ifeval"],
        num_fewshot=0,
        batch_size=8
    )
    
    return results['results']['ifeval']

# Test all quantization variants
baseline_score = evaluate_quantized_model("meta-llama/Llama-3.1-8B-Instruct", "baseline")
fp8_score = evaluate_quantized_model("./quantized_model/", "FP8")
nvfp4_score = evaluate_quantized_model("./quantized_model/NVFP4/", "NVFP4")
```

### Results Comparison

| Model Variant | Size | IFEval Score | Accuracy Retained |
|---------------|------|--------------|-------------------|
| BFloat16 (baseline) | 16 GB | 0.847 | 100% |
| FP8 Quantized | 9 GB | 0.839 | 99.1% |
| NVFP4 Quantized | 6 GB | 0.821 | 96.9% |

**Key Finding**: Even with aggressive NVFP4 quantization (62% size reduction), the model retains **96.9% of instruction-following accuracy**—a remarkable trade-off for cost-conscious deployment scenarios.

### Integrating into CI/CD

Automating accuracy testing as a GitHub Actions workflow ensures quality gates are enforced on every model export:

```yaml
name: Quantized Model Quality Gate

on:
  push:
    paths:
      - 'quantized_model/**'

jobs:
  ifeval-test:
    runs-on: gpu-runner
    steps:
      - name: Run IFEval Benchmark
        run: |
          python eval_ifeval.py --model-path ./quantized_model/
      
      - name: Check Accuracy Threshold
        run: |
          # Fail deployment if accuracy drops below 95%
          python check_threshold.py --threshold 0.95
```

This approach ensures no quantized model reaches production without meeting predefined accuracy requirements.

---

## Exporting and Sharing Quantized Models

ModelOpt simplifies exporting HuggingFace-compatible checkpoints, making it easy to share quantized models with the community:

```python
from modelopt.torch.export import export_hf_checkpoint

export_path = "./quantized_model/NVFP4/"
export_hf_checkpoint(model, export_dir=export_path)
tokenizer.save_pretrained(export_path)

# Upload to HuggingFace Hub
from huggingface_hub import upload_folder

upload_folder(
    folder_path=export_path,
    repo_id="username/Llama-3.1-8B-ModelOpt-NVFP4",
    commit_message="Upload quantized model"
)
```

---

## Key Takeaways

1. **Quantization is production-ready** — With the right tools, significant model size reductions are achievable without major accuracy loss.

2. **FP8 is the safe bet** — Delivers ~44% size reduction with minimal risk. Ideal for most production use cases.

3. **NVFP4 is for aggressive optimization** — Achieves ~62% size reduction, but requires NVIDIA's latest hardware and careful validation.

4. **Calibration data matters** — The quality of the calibration dataset directly impacts quantization quality. Representative data is essential.

5. **The ecosystem is maturing** — NVIDIA ModelOpt + HuggingFace integration makes the workflow seamless and production-ready.

6. **Test before deploying** — Benchmarks like IFEval quantify accuracy retention. Automated testing in CI/CD prevents accuracy regressions from reaching production.

---

## What's Next?

Future explorations will include:

- Benchmarking inference latency and throughput comparisons
- Expanding accuracy testing to additional benchmarks (MMLU, HellaSwag, TruthfulQA)
- Identifying edge cases where NVFP4 accuracy degrades most
- Exploring INT4 and INT8 quantization with other libraries (Quanto, bitsandbytes)
- Investigating quantization-aware training (QAT) for improved results
- Building an automated model quality dashboard showing size vs. accuracy trade-offs

---

## Try It Out

The quantized models are available on HuggingFace:

- 🔗 [Llama-3.1-8B-ModelOpt-FP8](https://huggingface.co/tokenlabsdotrun/Llama-3.1-8B-ModelOpt-FP8)
- 🔗 [Llama-3.1-8B-ModelOpt-NVFP4](https://huggingface.co/tokenlabsdotrun/Llama-3.1-8B-ModelOpt-NVFP4)

The complete notebooks are available on GitHub.

---

## Resources

- [NVIDIA ModelOpt Documentation](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
- [HuggingFace Quantization Guide](https://huggingface.co/docs/transformers/quantization)
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
- [IFEval Benchmark Paper](https://arxiv.org/abs/2311.07911) — Instruction-Following Evaluation for LLMs
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — Framework for LLM evaluation

---

*What has been the experience with model quantization in the community? Drop thoughts and results in the comments—always interesting to hear how others are approaching these trade-offs!*

#MachineLearning #LLM #Quantization #NVIDIA #DeepLearning #AI #MLOps #Llama #HuggingFace

---

*Elizabeth T. | AI/ML Engineer | Building efficient AI systems*
