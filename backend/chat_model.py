"""Lightweight wrapper around a local conversational language model.

This module exposes a small helper class that loads a causal language model
from Hugging Face and formats chat style prompts so the backend can expose a
`/chat` endpoint.  The goal is to provide an entirely local chatbot experience
that does not depend on external APIs once the model weights are downloaded.

The default model (`microsoft/DialoGPT-medium`) is compact enough to run on a
CPU-only developer machine while still demonstrating multi-turn dialogue.  The
class keeps the implementation intentionally simple so that the backend can be
started without additional orchestration or database layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ChatConfig:
    """Configuration for the local chat model."""

    model_name: str = "microsoft/DialoGPT-medium"
    max_new_tokens: int = 200
    temperature: float = 0.7
    top_p: float = 0.9


class LocalChatModel:
    """Simple helper around a causal LM to support chat style prompts."""

    def __init__(self, config: ChatConfig | None = None) -> None:
        self.config = config or ChatConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)

        # Move to GPU if available – otherwise remain on CPU for portability.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(device)

        # Some DialoGPT variants do not ship with a pad token; fall back to EOS.
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def _format_messages(messages: Sequence[dict]) -> str:
        """Format an OpenAI-style message list into a plain-text prompt.

        The model does not have a native chat template, so we mimic a simple
        conversation transcript where each turn is prefixed with the role name.
        """

        transcript: List[str] = []
        role_map = {"system": "System", "user": "User", "assistant": "Assistant"}

        for msg in messages:
            role = msg.get("role", "user").lower()
            prefix = role_map.get(role, "User")
            content = msg.get("content", "").strip()
            if not content:
                continue
            transcript.append(f"{prefix}: {content}")

        transcript.append("Assistant:")
        return "\n".join(transcript)

    def chat(self, messages: Sequence[dict]) -> dict:
        """Generate a reply for a list of chat messages.

        Parameters
        ----------
        messages:
            Sequence of dictionaries in the format ``{"role": str, "content": str}``.

        Returns
        -------
        dict
            Dictionary containing the assistant's reply alongside token counts.
        """

        prompt = self._format_messages(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Remove the prompt so only newly generated content is returned.
        if generated_text.startswith(prompt):
            reply = generated_text[len(prompt) :].strip()
        else:
            reply = generated_text.strip()

        if not reply:
            reply = "抱歉，我暂时没有合适的回答。"

        prompt_tokens = len(inputs["input_ids"][0])
        generated_tokens = max(len(output_ids[0]) - prompt_tokens, 0)

        return {
            "response": reply,
            "prompt_tokens": int(prompt_tokens),
            "generated_tokens": int(generated_tokens),
            "model": self.config.model_name,
        }


__all__ = ["ChatConfig", "LocalChatModel"]

