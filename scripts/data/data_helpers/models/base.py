"""Base classes for mutation model handling."""

import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type


@dataclass
class Message:
    role: str
    content: str
    images: List[bytes] = field(default_factory=list)

    def to_openai_format(self, include_images: bool = True) -> Dict[str, Any]:
        if not self.images or not include_images:
            return {"role": self.role, "content": self.content}
        content_parts = []
        for img_bytes in self.images:
            image_data = base64.b64encode(img_bytes).decode("utf-8")
            mime_type = "image/jpeg"
            if img_bytes[:8] == b"\x89PNG\r\n\x1a\n":
                mime_type = "image/png"
            elif img_bytes[:2] == b"\xff\xd8":
                mime_type = "image/jpeg"
            elif img_bytes[:6] in (b"GIF87a", b"GIF89a"):
                mime_type = "image/gif"
            elif img_bytes[:4] == b"RIFF" and len(img_bytes) > 12 and img_bytes[8:12] == b"WEBP":
                mime_type = "image/webp"
            content_parts.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}})
        if self.content:
            content_parts.append({"type": "text", "text": self.content})
        return {"role": self.role, "content": content_parts}


@dataclass
class ModelResponse:
    content: str
    raw_response: Optional[Any] = None
    error: Optional[str] = None
    skipped: bool = False


class BaseModel(ABC):
    MODEL_TYPE: str = ""

    def __init__(self, model_name: str, **generation_kwargs):
        self.model_name = model_name
        self.generation_kwargs = generation_kwargs

    @abstractmethod
    async def generate(self, messages: List[Message]) -> ModelResponse:
        raise NotImplementedError


class ModelRegistry:
    _models: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(model_cls: Type[BaseModel]):
            cls._models[name] = model_cls
            return model_cls

        return decorator
