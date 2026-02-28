import os
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

def to_messages(ex):
    return {
        "messages": [
            {"role": "user", "content": ex['original']},
            {"role": "assistant", "content": ex["corporate"]},
        ]
    }

dataset = load_dataset("json", data_files="output.json")

ds_chat = dataset.map(to_messages, remove_columns=['original', 'corporate'])

dataset_name = os.getenv("DATASET_NAME")
if not dataset_name:
    raise ValueError("DATASET_NAME environment variable is not set")
ds_chat.push_to_hub(dataset_name, private=True)