hf jobs uv run \
    --flavor l4x1 \
    --with trl \
    --with transformers \
    --with peft \
    --with datasets \
    --with dotenv \
    --with liger-kernel \
    --with bitsandbytes \
    --secrets HF_TOKEN \
    --secrets DATASET_NAME \
    fine_tuning.py