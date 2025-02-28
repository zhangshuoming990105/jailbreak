from anthropic import Anthropic
from openai import OpenAI
import os

MODEL_TYPE_MAP = {
    # OPENAI series
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "gpt-4o-2024-08-06": "openai",
    "gpt-4o-2024-11-20": "openai",
    "gpt-4o-mini-2024-07-18": "openai",
    "o1-2024-12-17": "openai",
    "o3-mini-2025-01-31": "openai",
    # ANTHROPIC series
    "claude-3-7-sonnet-20250219": "anthropic",
    "claude-3-5-sonnet-20240620": "anthropic",
    "claude-3-5-sonnet-20241022": "anthropic",
    "claude-3-5-haiku-20241022": "anthropic",
    "claude-3-opus-20240229": "anthropic",
    # Deepseek series
    "deepseek-reasoner": "deepseek",
    "deepseek-chat": "deepseek",
    # SILICONFLOW series
    "deepseek-ai/DeepSeek-V3": "siliconflow",
    "Pro/deepseek-ai/DeepSeek-R1": "siliconflow",
    "meta-llama/Llama-3.3-70B-Instruct": "siliconflow",
    "meta-llama/Meta-Llama-3.1-405B-Instruct": "siliconflow",
    "01-ai/Yi-1.5-34B-Chat-16K": "siliconflow",
    "Qwen/QwQ-32B-Preview": "siliconflow",
    # VLLM series
    "microsoft/Phi-3.5-MoE-instruct": "vllm",
    "llama3/Meta-Llama-3-70B-Instruct": "vllm",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "vllm",
}

def get_client(model_type):
    # Create and return the appropriate client instance based on model_type
    if model_type == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        return OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")
    elif model_type == "deepseek":
        return OpenAI(
            base_url="https://api.deepseek.ai/v1", api_key=os.getenv("DEEPSEEK_API_KEY")
        )
    elif model_type == "siliconflow":
        api_key = os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            raise ValueError("SILICONFLOW_API_KEY not set")
        return OpenAI(
            api_key=api_key, base_url="https://api.siliconflow.cn/v1/chat/completions"
        )
    elif model_type == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        return Anthropic(api_key=api_key)
    elif model_type == "vllm":
        client = OpenAI(
            base_url="http://localhost:8080/v1",
            api_key="local",
        )
        return client
    else:
        raise ValueError("Unsupported model type")
