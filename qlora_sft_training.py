# ============================================================
# QLoRA SFT Training Cell
# Trains Qwen3-4B-Thinking-2507 on NuminaMath-CoT + MATH
# Designed for DSMLP MIG slice (~8GB effective VRAM)
# ============================================================

# ── 0. Install dependencies ──────────────────────────────────
# Run this block once, then comment it out
# !pip install -q transformers==4.47.0 peft==0.14.0 trl==0.13.0 \
#     bitsandbytes==0.45.0 datasets accelerate einops

# ── 1. Imports ───────────────────────────────────────────────
import os
import torch
import json
from pathlib import Path
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ── 2. Config — edit these ───────────────────────────────────
MODEL_ID         = "Qwen/Qwen3-4B-Thinking-2507"
OUTPUT_DIR       = "./sft_checkpoints"
MERGED_DIR       = "./sft_merged"          # final merged model saved here
LOG_FILE         = "./sft_training_log.json"

MAX_SAMPLES      = 30_000   # total examples after combining both datasets
MAX_SEQ_LEN      = 4096     # reduce to 2048 if you OOM
LORA_RANK        = 64       # reduce to 16 if you OOM; increase to 64 for more capacity
LORA_ALPHA       = 128       # keep at 2x rank
BATCH_SIZE       = 2        # per-device; increase to 4 if VRAM allows
GRAD_ACCUM       = 8        # effective batch = BATCH_SIZE * GRAD_ACCUM = 16
LEARNING_RATE    = 2e-4
NUM_EPOCHS       = 1        # run 1 epoch first, evaluate, then continue if good
WARMUP_STEPS     = 50
SAVE_STEPS       = 200      # checkpoint frequently given session limits
LOGGING_STEPS    = 10
SEED             = 42

# System prompts (must match your inference prompts exactly)
SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. Think concisely; do not over-explain your answer. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}. "
    "Here is an example: Problem: What is 15% of 80?; Solution: 15% of 80 = 0.15 × 80 = 12; Answer: \\boxed{12} "
    "Once you think you have an answer, make sure to thoroughly double check to ensure you made no mistakes. "
    "If possible, take your answer and plug it into the original question to make sure everything is correct. "
    "If you suspect a mistake, start over until you get it right. "
    "Remember: output your final answer as \\boxed{your answer} and nothing after it."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. Think concisely; do not over-explain your answer. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
    "Work through each answer choice systematically. Eliminate obviously wrong options first, then verify your chosen answer. "
    "Once you think you have an answer, make sure to thoroughly double check to ensure you made no mistakes. "
    "If possible, take your answer and plug it into the original question to make sure everything is correct. "
    "If you suspect a mistake, start over until you get it right. "
    "Remember: output your final answer as \\boxed{your answer} and nothing after it."
)

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(MERGED_DIR).mkdir(parents=True, exist_ok=True)

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ── 3. Load & preprocess datasets ────────────────────────────

def extract_boxed(text: str) -> str:
    """Pull the last \\boxed{} content from a solution string."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return text.strip()
    depth, i = 0, idx + len("\\boxed{")
    start = i
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
        i += 1
    return text[start:].strip()


def format_for_thinking_model(question: str, solution: str, answer: str,
                               tokenizer, is_mcq: bool = False) -> str:
    """
    Format a (question, solution, answer) triple into the chat template
    the thinking model expects, with <think>...</think> tags.
    Returns the full formatted string ready for SFT.
    """
    system = SYSTEM_PROMPT_MCQ if is_mcq else SYSTEM_PROMPT_MATH

    # The assistant turn: thinking trace + boxed answer
    # We use the solution as the thinking content and extract/format the answer
    assistant_content = (
        f"<think>\n{solution.strip()}\n</think>\n"
        f"\\boxed{{{answer}}}"
    )

    messages = [
        {"role": "system",    "content": system},
        {"role": "user",      "content": question.strip()},
        {"role": "assistant", "content": assistant_content},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def load_and_process_numina(tokenizer, n_samples: int) -> Dataset:
    """
    Load NuminaMath-CoT and convert to SFT format.
    Fields: problem, solution (full CoT ending with \\boxed{})
    """
    print(f"Loading NuminaMath-CoT (taking {n_samples} samples)...")
    raw = load_dataset(
        "AI-MO/NuminaMath-CoT",
        split="train",
    )

    # Filter: must have non-trivial solution and a boxed answer
    raw = raw.filter(
        lambda x: (
            x.get("solution") and
            "\\boxed{" in x.get("solution", "") and
            len(x.get("solution", "")) > 100 and   # filter out trivially short solutions
            len(x.get("problem",  "")) > 20
        ),
        num_proc=2,
    )

    # Shuffle and cap
    raw = raw.shuffle(seed=SEED).select(range(min(n_samples, len(raw))))

    def process_numina(example):
        problem  = example["problem"]
        solution = example["solution"]
        answer   = extract_boxed(solution)
        # For NuminaMath the solution already ends with \boxed{}, use everything
        # before the last \boxed as the thinking trace
        think_end = solution.rfind("\\boxed{")
        thinking  = solution[:think_end].strip() if think_end > 0 else solution

        text = format_for_thinking_model(problem, thinking, answer, tokenizer)
        return {"text": text, "source": "numina"}

    return raw.map(process_numina, remove_columns=raw.column_names, num_proc=4)


def load_and_process_math(tokenizer, n_samples: int) -> Dataset:
    print(f"Loading MATH dataset (taking {n_samples} samples)...")
    
    configs = ['algebra', 'counting_and_probability', 'geometry', 
               'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
    
    splits = []
    for config in configs:
        ds = load_dataset("EleutherAI/hendrycks_math", config, split="train")
        splits.append(ds)
    
    raw = concatenate_datasets(splits)
    
    raw = raw.filter(
        lambda x: (
            x.get("solution") and
            "\\boxed{" in x.get("solution", "") and
            len(x.get("solution", "")) > 50
        ),
        num_proc=2,
    )

    raw = raw.shuffle(seed=SEED).select(range(min(n_samples, len(raw))))

    def process_math(example):
        problem  = example["problem"]
        solution = example["solution"]
        answer   = extract_boxed(solution)
        think_end = solution.rfind("\\boxed{")
        thinking  = solution[:think_end].strip() if think_end > 0 else solution
        text = format_for_thinking_model(problem, thinking, answer, tokenizer)
        return {"text": text, "source": "math"}

    return raw.map(process_math, remove_columns=raw.column_names, num_proc=4)


def filter_by_length(dataset: Dataset, tokenizer, max_len: int) -> Dataset:
    """Remove examples that exceed max_len tokens — they'd get truncated anyway."""
    print("Filtering by token length...")

    def is_short_enough(example):
        ids = tokenizer(example["text"], truncation=False)["input_ids"]
        return len(ids) <= max_len

    return dataset.filter(is_short_enough, num_proc=4)


# ── 4. Load tokenizer first (needed for dataset formatting) ──
print(f"\nLoading tokenizer from {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token     = tokenizer.eos_token
tokenizer.padding_side  = "right"   # important for SFT loss masking

# ── 5. Build combined dataset ─────────────────────────────────
# Split budget: 60% NuminaMath (harder, more diverse), 40% MATH (cleaner)
n_numina = int(MAX_SAMPLES * 0.6)
n_math   = MAX_SAMPLES - n_numina

ds_numina = load_and_process_numina(tokenizer, n_numina)
ds_math   = load_and_process_math(tokenizer, n_math)

combined = concatenate_datasets([ds_numina, ds_math]).shuffle(seed=SEED)
combined = filter_by_length(combined, tokenizer, MAX_SEQ_LEN)

# Train/eval split — hold out 2% for validation
split    = combined.train_test_split(test_size=0.02, seed=SEED)
train_ds = split["train"]
eval_ds  = split["test"]

print(f"\nDataset ready:")
print(f"  Train: {len(train_ds):,} examples")
print(f"  Eval:  {len(eval_ds):,} examples")
print(f"\nSample formatted text (first 500 chars):")
print(train_ds[0]["text"][:500])


# ── 6. Load model in 4-bit ────────────────────────────────────
print(f"\nLoading model {MODEL_ID} in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",          # NF4 is better than fp4 for LLMs
    bnb_4bit_use_double_quant=True,     # nested quantization saves ~0.4 GB
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,
)

model = prepare_model_for_kbit_training(model)
model.config.use_cache = False         # must disable when using grad checkpointing

# ── 7. Attach LoRA adapters ───────────────────────────────────
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    # Target all linear projections — important for reasoning tasks
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected: ~40-80M trainable / 4B total (~1-2%)


# ── 8. Checkpoint resume callback ────────────────────────────
class LoggingCallback(TrainerCallback):
    """Saves loss curve to JSON so you can plot it after the run."""
    def __init__(self, log_path):
        self.log_path = log_path
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            entry = {"step": state.global_step, **{k: v for k, v in logs.items()
                                                    if isinstance(v, (int, float))}}
            self.logs.append(entry)
            with open(self.log_path, "w") as f:
                json.dump(self.logs, f, indent=2)


# ── 9. Training config ────────────────────────────────────────
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=True,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=WARMUP_STEPS,
    bf16=True,
    fp16=False,
    logging_steps=LOGGING_STEPS,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,             # keep only 3 checkpoints to save disk space
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    max_length=MAX_SEQ_LEN,
    dataset_text_field="text",      # column containing the formatted strings
    dataset_kwargs={"num_proc": 4},
    dataloader_num_workers=1,
    seed=SEED,
    report_to="none",               # disable wandb; use our JSON logger instead
    # Resume from checkpoint if one exists — critical for session restarts
    # resume_from_checkpoint=True if any(Path(OUTPUT_DIR).glob("checkpoint-*")) else None,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
    callbacks=[LoggingCallback(LOG_FILE)],
)

# ── 10. Train ─────────────────────────────────────────────────
print("\nStarting training...")
print(f"  Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
print(f"  Total steps: {len(train_ds) // (BATCH_SIZE * GRAD_ACCUM) * NUM_EPOCHS:,}")
print(f"  Checkpoints every {SAVE_STEPS} steps → {OUTPUT_DIR}")
print(f"  Loss log → {LOG_FILE}\n")

train_result = trainer.train(
    resume_from_checkpoint=True if any(Path(OUTPUT_DIR).glob("checkpoint-*")) else None
)

print("\nTraining complete.")
print(f"  Final train loss: {train_result.training_loss:.4f}")


# ── 11. Save & merge adapters ─────────────────────────────────
# Save the LoRA adapter weights alone first (small, fast)
adapter_dir = os.path.join(OUTPUT_DIR, "final_adapter")
trainer.model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print(f"Adapter saved to {adapter_dir}")

# Merge LoRA weights into the base model for faster inference
# (merging requires loading in fp16/bf16, not 4-bit)
print("\nMerging adapter into base model for inference...")
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
merged_model = PeftModel.from_pretrained(base_model, adapter_dir)
merged_model = merged_model.merge_and_unload()
merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)

print(f"Merged model saved to {MERGED_DIR}")
print("Point your inference notebook to this directory instead of the HuggingFace model ID.")


# ── 12. Quick sanity check ────────────────────────────────────
print("\nRunning sanity check on merged model...")
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model=MERGED_DIR,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

test_prompt = tokenizer.apply_chat_template(
    [
        {"role": "system",  "content": SYSTEM_PROMPT_MATH},
        {"role": "user",    "content": "What is the sum of the first 10 positive integers?"},
    ],
    tokenize=False,
    add_generation_prompt=True,
)

out = pipe(test_prompt, max_new_tokens=512, temperature=0.6, top_p=0.95, do_sample=True)
print("\nSanity check response:")
print(out[0]["generated_text"][len(test_prompt):])
# Should see <think>...</think>\boxed{55}