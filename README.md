# mistral-hackathon-2026
Mistral AI Hackathon 2026 — LoRA Fine-Tune

## Build the dataset:
```
python3 input_text_creator.py &&
python3 dataset_creator.py &&
python3 prepare_data.py
```


## Run the finetuning:
```
bash start_hf_tuning
```