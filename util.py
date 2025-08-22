from openai import OpenAI
import os
from loguru import logger
import sys

MODEL_TYPE_MAP = {
    # OPENAI series
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "gpt-4o-2024-08-06": "openai",
    "gpt-4o-2024-11-20": "openai",
    "gpt-4o-mini-2024-07-18": "openai",
    "o1-2024-12-17": "openai",
    "o3-mini-2025-01-31": "openai",
    "gpt-4.1": "openai",
    "gpt-4.1-mini": "openai",
    "gpt-5-nano": "openai",
    "gpt-5-mini": "openai",
    "gpt-5": "openai",
    # ANTHROPIC series
    "claude-3-7-sonnet-20250219": "anthropic",
    "claude-3-5-sonnet-20240620": "anthropic",
    "claude-3-5-sonnet-20241022": "anthropic",
    "claude-3-5-haiku-20241022": "anthropic",
    "claude-3-opus-20240229": "anthropic",
    # GEMINI series
    "gemini-1.5-flash-8b-latest": "gemini",
    "gemini-1.5-pro-latest": "gemini",
    "gemini-2.0-flash": "gemini",
    "gemini-2.0-flash-lite": "gemini",
    "gemini-2.0-pro-exp": "gemini",
    
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
    "google/gemma-2-9b-it": "vllm",
    "mistralai/Mistral-Nemo-Instruct-2407": "vllm",
    "Qwen/Qwen2.5-32B-Instruct": "vllm",
    "meta-llama/Llama-3.1-8B-Instruct": "vllm",
}


def get_client(model_type: str) -> OpenAI:
    # Create and return the appropriate client instance based on model_type
    if model_type == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        return OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")
    elif model_type == "deepseek":
        return OpenAI(
            base_url="https://api.deepseek.com/v1", api_key=os.getenv("DEEPSEEK_API_KEY")
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
        return OpenAI(
            api_key=api_key,
            base_url="https://api.anthropic.com/v1/",  # Anthropic's API endpoint
        )
    elif model_type == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        return OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",  # Gemini's API endpoint
        )
    elif model_type == "vllm":
        return OpenAI(
            base_url="http://localhost:8080/v1",
            api_key="local",
        )
    else:
        raise ValueError("Unsupported model type")


def get_logger(log_level="INFO", log_file=None):
    logger.remove()
    logger.add(
        log_file if log_file else sys.stderr,
        level=log_level,
    )
    return logger
