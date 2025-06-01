from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

client = InferenceClient(
    provider="hf-inference",
    api_key=token,
)

completion = client.chat.completions.create(
    model="Qwen/Qwen3-235B-A22B",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
)

print(completion.choices[0].message)

