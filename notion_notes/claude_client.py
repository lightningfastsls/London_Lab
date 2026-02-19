"""Thin wrapper around the Anthropic SDK for structured prompting."""

from __future__ import annotations

import json
import logging

from anthropic import Anthropic

logger = logging.getLogger(__name__)


class ClaudeClient:
    """Wrapper around the Anthropic SDK with convenience methods.

    Parameters
    ----------
    api_key : str
        Anthropic API key.
    default_model : str
        Model to use when none is specified per-call.
    """

    def __init__(
        self,
        api_key: str,
        default_model: str = "claude-sonnet-4-6",
    ) -> None:
        self.client = Anthropic(api_key=api_key)
        self.default_model = default_model

    def prompt(
        self,
        user_message: str,
        system: str = "",
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> str:
        """Send a prompt and return the text response.

        Parameters
        ----------
        user_message : str
            The user message to send.
        system : str
            Optional system prompt.
        max_tokens : int
            Maximum tokens in the response.
        model : str or None
            Override the default model for this call.

        Returns
        -------
        str
            The text content of the first response block.
        """
        model = model or self.default_model
        logger.debug("Claude prompt (%s, max_tokens=%d)", model, max_tokens)

        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user_message}],
        }
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)
        return response.content[0].text

    def prompt_json(
        self,
        user_message: str,
        system: str = "",
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> dict:
        """Send a prompt and parse the response as JSON.

        Parameters
        ----------
        user_message : str
            The user message to send.
        system : str
            Optional system prompt.
        max_tokens : int
            Maximum tokens in the response.
        model : str or None
            Override the default model for this call.

        Returns
        -------
        dict
            Parsed JSON from the response.

        Raises
        ------
        ValueError
            If the response cannot be parsed as valid JSON.
        """
        text = self.prompt(
            user_message=user_message,
            system=system,
            max_tokens=max_tokens,
            model=model,
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Claude response is not valid JSON: {text[:200]!r}"
            ) from exc
