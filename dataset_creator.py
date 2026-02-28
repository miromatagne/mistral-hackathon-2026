import json
import os
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """
You are a helpful assistant that creates datasets for a fine-tuning task.
Your goal is to translate a given text into very corporate language.
It has to be exaggerated, over the top but not too long.
It should be approximately the same length as the original text.
Do not include any other text than the corporate translation in the output.
"""


USER_PROMPT = """
Translate the following text into very corporate language.

Text: {text}
"""

def generate_completion(prompt: str, model: str = "mistral-large-latest") -> str:
    """
    Generate a completion from a Mistral model.
    
    Args:
        prompt: The user prompt to send to the model
        model: The Mistral model to use (default: mistral-large-latest)
    
    Returns:
        The generated text response from the model
    """
    client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
    
    response = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    
    return response.choices[0].message.content


def create_dataset(input_file: str = "input.txt", output_file: str = "output.json"):
    """
    Create a dataset by reading messages from input file and generating corporate translations.
    
    Args:
        input_file: Path to the input file with one message per line
        output_file: Path to save the JSON output
    """
    with open(input_file, "r") as f:
        messages = [line.strip() for line in f if line.strip()]
    
    dataset = []
    
    for i, message in enumerate(messages):
        print(f"Processing {i + 1}/{len(messages)}: {message[:50]}...")
        
        prompt = USER_PROMPT.format(text=message)
        corporate = generate_completion(prompt)
        
        dataset.append({
            "original": message,
            "corporate": corporate
        })
    
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Dataset saved to {output_file}")


if __name__ == "__main__":
    create_dataset()
