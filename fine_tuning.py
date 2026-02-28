import torch
import os

from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

MODEL_NAME = "ministral/Ministral-3b-instruct"

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

dataset_name = os.getenv("DATASET_NAME")

if not dataset_name:
    raise ValueError("DATASET_NAME environment variable is not set")

dataset = load_dataset(dataset_name)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    attn_implementation="sdpa",
    # dtype=torch.float16,
    use_cache=True
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)

output_dir = "ministral-3b-instruct-lora"

training_args = SFTConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=30,
    learning_rate=2e-4,
    optim="paged_adamw_8bit",
    # logging_steps=1,
    # report_to="trackio",
    # trackio_space_id=output_dir,
    output_dir=output_dir,
    max_length=1024,
    use_liger_kernel=True,
    activation_offloading=True,
    push_to_hub=True
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    peft_config=peft_config
)

#login(token=hf_token)

print("Starting training...")
trainer.train()

#print("Saving model...")
#trainer.save_model()

# Push to hub
print("Pushing to Hugging Face Hub...")
trainer.push_to_hub()

print("Training completed and model uploaded!")
