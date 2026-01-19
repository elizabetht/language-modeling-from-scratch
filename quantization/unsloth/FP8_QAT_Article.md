# FP8 Quantization-Aware Training: How to Fine-Tune LLMs That Actually Work After Compression

Running an 8-billion parameter model costs real money. Every GB of VRAM saved translates to cheaper hardware. Every bit of precision dropped means faster inference.

But here's the problem: when a model is quantized after training, accuracy drops. Sometimes a lot.

What if the model could learn to handle that precision loss during training instead of being blindsided by it at deployment?

That's exactly what Quantization-Aware Training does.

---

## The Problem with Post-Training Quantization

Most quantization happens after training is complete. The process involves taking a perfectly trained model, compressing the weights from 16-bit to 8-bit (or 4-bit), and hoping for the best.

The model was never trained to handle rounding errors. It's like training someone to be a surgeon with perfect lighting, then expecting them to operate in the dark.

Sometimes it works fine. Sometimes the model starts hallucinating or loses its ability to follow instructions.

## QAT: Training in the Dark on Purpose

Quantization-Aware Training flips the script. During training, the process simulates what quantization will do to the model. The technical term is "fake quantization."

Here's how it works:

1. Take a weight value (say, 0.7823456)
2. Quantize it to FP8 (becomes 0.78125)
3. Immediately dequantize it back to higher precision
4. Use that slightly-wrong value for the forward pass
5. Let the model learn from the error

The model sees quantization noise during every training step. By the time training ends, the weights have adapted to work correctly even with reduced precision.

When the model is actually quantized for deployment, it isn't surprised. It's been practicing for this.

---

## Why Pair QAT with LoRA?

Training an 8B parameter model normally requires storing:
- 32GB for the model weights (float32)
- 32GB for gradients
- 64GB for Adam optimizer states (two values per parameter)

That's 128GB just to start training. This requires multiple high-end GPUs.

LoRA (Low-Rank Adaptation) changes this equation. Instead of training all 8 billion parameters, the process freezes the original weights and injects small trainable matrices into key layers. This results in training about 42 million parameters instead of 8 billion.

That's 0.5% of the original.

Memory requirements drop from 128GB to under 20GB. Suddenly single-GPU training becomes realistic.

And there's a nice side effect for QAT: those small LoRA adapters learn to compensate for quantization effects. The base model stays frozen while the adapters figure out how to produce good outputs despite the simulated precision loss.

---

## Walking Through the Implementation

This implementation uses Unsloth, which handles much of the complexity. Here's what each step actually does:

### Step 1: Load the Base Model

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    dtype=torch.bfloat16,
    max_seq_length=2048,
    load_in_4bit=False,
    fast_inference=False,
    max_lora_rank=32
)
```

The model is loaded in full-precision bfloat16. No quantization yet. The `load_in_4bit=False` is important—if an already-quantized model is loaded, QAT can't be done properly because the base weights are already compressed. The `max_lora_rank` prepares the model for LoRA injection.

### Step 2: Apply LoRA with QAT

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    qat_scheme="fp8-int4",
    use_gradient_checkpointing="unsloth",
    random_state=3047,
)
```

This does two things:
1. Injects LoRA adapters into the attention and MLP layers
2. Wraps those layers with fake quantization

The `r=32` is the rank of the LoRA matrices. Higher ranks give more capacity but use more memory.

The `qat_scheme="fp8-int4"` tells Unsloth to simulate FP8 quantization during training. The `lora_alpha=64` (2x the rank) is a common scaling factor.

### Step 3: Prepare the Dataset

The model needs to see properly formatted conversations. This implementation uses the FineTome-100k dataset with Llama 3's chat template:

```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

What is 2+2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The answer is 4.<|eot_id|>
```

Every conversation in training data needs this exact format. Unsloth's `standardize_data_formats` and `get_chat_template` helpers handle this formatting automatically.

### Step 4: Configure the Trainer

```python
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=1,
        max_steps=30,
        learning_rate=2e-5,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)
```

A few notes:
- `adamw_8bit` uses an 8-bit optimizer to save more memory
- Effective batch size is 1 × 4 = 4 through gradient accumulation
- `learning_rate=2e-5` is conservative for LoRA fine-tuning
- `linear` scheduler reduces LR gradually throughout training

### Step 5: Train

Now the actual training process begins. The training loop runs each forward pass through fake-quantized layers. The LoRA adapters adjust their weights to minimize loss despite the simulated precision reduction.

```python
trainer.train()
```

### Step 6: Convert to Real Quantization

This is the key step that converts fake quantization to real FP8 weights. The `step="convert"` parameter tells torchao to transform the simulated quantization into actual compressed weights.

```python
from torchao.quantization import quantize_
from torchao.quantization.qat import QATConfig

quantize_(model, QATConfig(step="convert"))
```

### Step 7: Check GPU Memory Usage

Monitoring GPU memory helps ensure your training doesn't exceed available VRAM. The combination of LoRA and QAT significantly reduces memory requirements compared to full-parameter training.

```python
import torch

# Display GPU information
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
else:
    print("No GPU available")
```

### Step 8: Convert Fake Quantization to Real Quantization

This step finalizes the quantization process, converting the fake quantized weights used during training into actual FP8 quantized weights for deployment.

```python
from torchao.quantization import quantize_
from torchao.quantization.qat import QATConfig

# Convert simulated quantization to actual FP8 weights
quantize_(model, QATConfig(step="convert"))
```

---

## Training Results

After training completes, the process produces LoRA adapters that are specifically tuned to work well with FP8 quantization.

### Saving the Model

There are two options for saving the trained model:
1. Save just the LoRA adapters separately (smaller file size)
2. Save the full quantized model with torchao integration

```python
# Save LoRA adapters separately
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# Save full quantized model with torchao
model.save_pretrained_torchao("model-torchao", tokenizer)
```

---

## The Safetensors Limitation with FP8

**FP8 quantized models can't be saved in safetensors format**.

When saving a standard model, there's a choice between PyTorch's `.bin` format or Hugging Face's `.safetensors` format. Safetensors is generally preferred because it's faster to load and more secure (no arbitrary code execution risk from pickle).

But safetensors only supports standard tensor types:
- float32, float16, bfloat16
- int8, int16, int32, int64
- uint8, uint16, uint32, uint64

FP8 quantized tensors from torchao use **custom tensor subclasses** that can only be serialized through PyTorch's native pickle-based format. The model files end up as `pytorch_model-00001-of-00004.bin` instead of `.safetensors`.

This is a current limitation of the ecosystem. As FP8 support matures, we may see safetensors add native FP8 dtype support, but for now, `.bin` files are required.

**What this means for deployment:**
- Model loading is slightly slower than safetensors
- Source trust is important for model files (pickle security concerns)
- The model still works correctly—it's just the file format

---

## Uploading to Hugging Face Hub

```python
from huggingface_hub import create_repo, upload_folder, ModelCard

repo_id = "your-username/Llama-3.1-8B-FP8-QAT"

# Create repository
create_repo(repo_id, exist_ok=True, token=hf_write_token)

# Upload model files
upload_folder(
    folder_path="model-torchao",
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload FP8 QAT fine-tuned model",
    token=hf_write_token
)
```

---

## When QAT Makes Sense

QAT isn't always necessary. If a model performs fine with basic post-training quantization, QAT isn't needed.

But QAT is worth the extra training time when:
- Pushing to very low precision (FP8, INT4)
- Seeing quality degradation after quantization
- Deploying to edge devices where every bit counts

The model learns to be robust to precision loss instead of being a victim of it.

---

## Final Thoughts

Quantization isn't just about making models smaller. It's about making them deployable. The gap between a model that runs in a research lab and one that runs in production often comes down to memory and compute efficiency.

QAT bridges part of that gap. The approach trades some additional training complexity for models that maintain quality after compression.

The current limitation with safetensors format is a minor inconvenience, not a blocker. The model works the same way—it's important to be mindful of model file sources.

The math is simple: train with noise, deploy with confidence.

---

## Key Takeaways

1. **FP8 QAT** trains models to handle quantization noise during training, not after
2. **LoRA + QAT** reduces memory from 128GB to under 20GB for 8B parameter models
3. **`qat_scheme="fp8-int4"`** enables fake quantization in Unsloth
4. **`quantize_(model, QATConfig(step="convert"))`** converts fake quant to real FP8 weights
5. **Safetensors doesn't support FP8**—models save as `.bin` files instead
6. **Upload to HF Hub** using `upload_folder` for the quantized model directory
