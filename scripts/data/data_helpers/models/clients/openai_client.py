"""OpenAI and Azure OpenAI model clients."""

import os
from typing import List

from scripts.data.data_helpers.models.base import BaseModel, Message, ModelRegistry, ModelResponse


@ModelRegistry.register("openai")
class OpenAIClient(BaseModel):
    MODEL_TYPE = "openai"

    def __init__(self, model_name: str = "gpt-4o", api_key_env: str = "OPENAI_API_KEY", **generation_kwargs):
        super().__init__(model_name, **generation_kwargs)
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("openai is required. Install with: pip install openai") from exc
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"API key not found: {api_key_env}")
        self._client = AsyncOpenAI(api_key=api_key)

    async def generate(self, messages: List[Message]) -> ModelResponse:
        from openai import APIError, BadRequestError, RateLimitError

        formatted_messages = [msg.to_openai_format() for msg in messages]
        try:
            response = await self._client.chat.completions.create(
                model=self.model_name,
                messages=formatted_messages,
                **self.generation_kwargs,
            )
            return ModelResponse(content=response.choices[0].message.content, raw_response=response)
        except BadRequestError as e:
            error_msg = str(e)
            if "content_filter" in error_msg or "content management policy" in error_msg:
                return ModelResponse(content="", error=f"Content filtered: {error_msg[:200]}", skipped=True)
            raise
        except RateLimitError:
            raise
        except APIError as e:
            error_msg = str(e).lower()
            if any(kw in error_msg for kw in ["rate", "limit", "quota", "capacity", "throttl"]):
                raise
            return ModelResponse(content="", error=f"API error: {str(e)[:200]}", skipped=True)


@ModelRegistry.register("azure_openai")
class AzureOpenAIClient(BaseModel):
    MODEL_TYPE = "azure_openai"

    def __init__(self, model_name: str = "gpt-4o", api_version: str = "2024-12-01-preview", api_key_env: str = "SECUREGPT_API_KEY", endpoint_env: str = "OPENAI_ENDPOINT", **generation_kwargs):
        super().__init__(model_name, **generation_kwargs)
        try:
            from openai import AsyncAzureOpenAI
        except ImportError as exc:
            raise ImportError("openai is required. Install with: pip install openai") from exc
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"API key not found: {api_key_env}")
        headers = {
            "Ocp-Apim-Subscription-Key": api_key,
            "Content-Type": "application/json",
        }
        default_endpoint = f"https://apim.stanfordhealthcare.org/openai-eastus2/deployments/{model_name}/chat/completions?api-version={api_version}"
        url = os.getenv(endpoint_env, default_endpoint)
        import httpx
        self._client = AsyncAzureOpenAI(
            api_version=api_version,
            azure_endpoint=url,
            azure_deployment=model_name,
            default_headers=headers,
            azure_ad_token=api_key,
            timeout=httpx.Timeout(600.0, connect=30.0),
        )
        self.deployment = model_name

    async def generate(self, messages: List[Message]) -> ModelResponse:
        from openai import APIError, BadRequestError, RateLimitError

        formatted_messages = [msg.to_openai_format() for msg in messages]
        try:
            response = await self._client.chat.completions.create(
                model=self.deployment,
                messages=formatted_messages,
                **self.generation_kwargs,
            )
            return ModelResponse(content=response.choices[0].message.content, raw_response=response)
        except BadRequestError as e:
            error_msg = str(e)
            if "content_filter" in error_msg or "content management policy" in error_msg:
                return ModelResponse(content="", error=f"Content filtered: {error_msg[:200]}", skipped=True)
            raise
        except RateLimitError:
            raise
        except APIError as e:
            error_msg = str(e).lower()
            if any(kw in error_msg for kw in ["rate", "limit", "quota", "capacity", "throttl"]):
                raise
            return ModelResponse(content="", error=f"API error: {str(e)[:200]}", skipped=True)
