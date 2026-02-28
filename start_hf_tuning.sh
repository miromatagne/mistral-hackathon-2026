hf jobs uv run \
    --flavor cpu-basic \
    --with trl \
    --with transformers \
    --with peft \
    --with datasets \
    --secrets HF_TOKEN \
    fine_tuning.py