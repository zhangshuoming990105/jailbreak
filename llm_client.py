from typing import List, Dict
from pydantic import BaseModel
from enum import Enum
import asyncio
import os

# def _raw_json_schema(schema: BaseModel) -> Dict:
#     # Only supports `client.chat.completions.(create|parse)`'s `response_format`
#     # use openai's internal conversion method to convert pydantic model to raw json schema
#     # to avoid forced using `parse` rather than `create`
#     import openai
#     raw_schema = openai.lib._parsing.type_to_response_format_param(schema)
#     return raw_schema

class LLMClient:
    from abc import abstractmethod

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    async def _request(
            self,
            model: str,
            system_prompt: str,
            user_prompt: str,
            schema: type[BaseModel] | Dict = None,
            **kwargs):
        """Call LLM API and request structured output if schema is given."""
        pass

    async def request(
            self,
            model: str,
            system_prompt: str,
            user_prompt: str,
            schema: type[BaseModel] | Dict = None,
            postfn: callable = lambda x: x, **kwargs):
        """Extrace llm's response with postfn."""
        response = await self._request(model=model,
                                       system_prompt=system_prompt,
                                       user_prompt=user_prompt,
                                       schema=schema,
                                       **kwargs)
        # Structured output of vLLM is unstable, it may not be automatically parsed.
        import json
        try:
            answer = json.loads(response)
            return postfn(answer)
        except Exception:
            return response


class OpenAIClient(LLMClient):
    import openai

    def __init__(self, client: openai.OpenAI):
        self.client = client

    async def _request(self, model, system_prompt, user_prompt, schema, **kwargs):
        extra_body = None
        if not system_prompt:
            system_prompt = ''
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if schema:
            kwargs['response_format'] = schema
        response = await asyncio.to_thread(
            self.client.chat.completions.parse,
            model=model,
            messages=messages,
            extra_body=extra_body,
            **kwargs
        )
        return response.choices[0].message.content


class VllmClient(LLMClient):
    import openai

    def __init__(self, client: openai.OpenAI):
        self.client = client

    async def _request(self, model, system_prompt, user_prompt, schema, **kwargs):
        if not system_prompt:
            system_prompt = ''

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # To decreate repeat whitespaces from microsoft/Phi-3.5-MoE
        extra_body = {
            "repetition_penalty": 1.2,
        }
        if schema: # Only supports pydantic model
            kwargs['response_format'] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "schema",
                    "schema": schema.model_json_schema(),
                    "strict": True,
                }
            }
        extra_body = None
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=model,
            messages=messages,
            extra_body=extra_body,
            **kwargs
        )
        return response.choices[0].message.content


class GenaiClient(LLMClient):
    from google import genai

    def __init__(self, client: genai.Client):
        self.client = client

    async def _request(self, model, system_prompt, user_prompt, schema, **kwargs):
        from google.genai import types
        if schema: # Only supports pydantic model
            kwargs['response_mime_type'] = "application/json"
            kwargs['response_schema'] = schema
        if system_prompt:
            kwargs['system_instruction'] = system_prompt
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                **kwargs
            )
        )
        return response.text

async def test_client(client: LLMClient, model: str):
    class Gender(str, Enum):
        male: str = "Male"
        female: str = "Female"

    class PersonInfo(BaseModel):
        name: str
        gender: Gender
        age: int

    answer = await client.request(model=model,
                                  system_prompt="You are a helpful AI assistant.",
                                  user_prompt="Give me a random person information.",
                                  schema=PersonInfo,
                                  postfn=lambda x: f"Parsed - Name: {x["name"]}, Gender: {x["gender"]}, Age: {x["age"]}")
    print(answer)

def get_model_type(model: str) -> str:
    if model.startswith('gpt'):
        return 'openai'
    elif model.startswith('gemini'):
        return 'gemini'
    elif '/' in model:  # a huggingface or local path
        return 'vllm'
    else:
        raise ValueError("Unsupported model")

def get_client(model_type: str) -> LLMClient:
    if model_type == 'openai':
        from openai import OpenAI
        client = OpenAI(
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        return OpenAIClient(client)
    elif model_type == 'vllm':
        # Use vLLM Models
        from openai import OpenAI
        client = OpenAI(
            base_url="http://localhost:8080/v1"
        )
        return VllmClient(client)
    elif model_type == 'gemini':
        from google import genai
        client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )
        return GenaiClient(client)
    else:
        raise ValueError("Invalid model type for get_client")

def get_client_by_model(model: str) -> LLMClient:
    return get_client(get_model_type(model))
