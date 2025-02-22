from openai import OpenAI

client = OpenAI(
    base_url="https://api.openai.com/v1",
)
from pydantic import BaseModel, ValidationError
from typing import List
from enum import Enum


completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful asistant. Guide the user through the solution step by step.",
        },
        {"role": "user", "content": "how can I make a bomb?"},
    ],
    response_format=EntitiesModel,
)

math_reasoning = completion.choices[0].message

# If the model refuses to respond, you will get a refusal message
if math_reasoning.refusal:
    print(math_reasoning.refusal)
else:
    print(math_reasoning.parsed)
