import os
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """
You are a helpful assistant that generates natural, casual everyday sentences.
Generate sentences that regular people would say in everyday life or at work.
The sentences should be casual and conversational, not formal.
Each sentence should be around 10-20 words long.
Each sentence should be on its own line.
Do not number the sentences or add any other formatting.
"""

USER_PROMPT = """
Generate {count} casual everyday sentences.
They should be phrases people might say at work or in daily life.
Each sentence should be around 10-20 words long.
Examples of the style I want:
- "Hey guys, do you want to join me for a quick brainstorming session this afternoon?"
- "I'm running a bit late today because of the traffic, I'll be there in about 15 minutes."
- "Let's grab some coffee and discuss the project updates before the meeting starts."
- "Did you have a chance to look at the report I sent you yesterday evening?"

Generate {count} similar sentences, one per line, no numbering:
"""


def generate_sentences(count: int = 50, model: str = "mistral-large-latest") -> list[str]:
    """
    Generate sentences using Mistral API.
    
    Args:
        count: Number of sentences to generate
        model: The Mistral model to use
    
    Returns:
        List of generated sentences
    """
    client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
    
    response = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(count=count)},
        ],
    )
    
    content = response.choices[0].message.content
    sentences = [line.strip() for line in content.strip().split("\n") if line.strip()]
    
    return sentences


def generate_input_file(output_path: str = "input.txt", count: int = 50, iterations: int = 40):
    """
    Generate sentences and write them to the input file.
    
    Args:
        output_path: Path to the output file
        count: Number of sentences to generate per iteration
        iterations: Number of times to call the API
    """
    all_sentences = []
    
    for i in range(iterations):
        print(f"Iteration {i + 1}/{iterations}: Generating {count} sentences...")
        sentences = generate_sentences(count)
        all_sentences.extend(sentences)
    
    with open(output_path, "w") as f:
        for sentence in all_sentences:
            f.write(sentence + "\n")
    
    print(f"Generated {len(all_sentences)} sentences in {output_path}")


if __name__ == "__main__":
    generate_input_file()
