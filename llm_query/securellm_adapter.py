# This source file is part of the ARPA-H CARE LLM project
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
This module handles calls to a large language model (LLM) using SecureLLM.

It includes a function to send a prompt to the model and return the generated
response. The default model used is GPT-4o with secure key management via VAULT_SECRET_KEY.
"""

# Standard library imports
import os
import logging
from typing import List, Dict, Any, Optional

# Third party imports
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Try to import secure-llm
try:
    from src.safran.tools.llm.secure_llm_client import (
        get_llm_client, extract_response_content, get_default_generation_config
    )
    _SECURELLM_AVAILABLE = True
except ImportError:
    logger.warning("secure-llm client not available, using fallback mode")
    _SECURELLM_AVAILABLE = False


class ModelConfig:  # pylint: disable=too-few-public-methods
    """
    Configuration constants for the LLM interaction using SecureLLM.
    """
    DEFAULT_LLM_MODEL = "gpt-4o"
    VAULT_SECRET_KEY = "VAULT_SECRET_KEY"


def _initialize_secure_client():
    """
    Initialize SecureLLM client using VAULT_SECRET_KEY from environment.

    Raises:
        ValueError: If VAULT_SECRET_KEY is not set or securellm is not available
    """
    load_dotenv()

    if not _SECURELLM_AVAILABLE:
        raise ImportError("secure-llm package not installed. Install with: pip install secure-llm")

    vault_key = os.getenv(ModelConfig.VAULT_SECRET_KEY)
    if not vault_key:
        raise ValueError(
            f"Vault private key not found in environment variable '{ModelConfig.VAULT_SECRET_KEY}'. "
            "Please set the VAULT_SECRET_KEY environment variable with your private key."
        )

    logger.info("Initialized SecureLLM client with private key from VAULT_SECRET_KEY")
    return vault_key


def get_llm_client_instance(model_name: Optional[str] = None):
    """
    Get or create the SecureLLM client instance.

    Args:
        model_name: Optional model name override. Defaults to ModelConfig.DEFAULT_LLM_MODEL.

    Returns:
        SecureLLM client instance
    """
    _initialize_secure_client()  # Verify key is available
    model = model_name or ModelConfig.DEFAULT_LLM_MODEL
    return get_llm_client(model_name=model)


def llm_call(prompt: str, temperature: float = 0.7, max_tokens: int = 10000) -> str:
    """
    Sends a prompt to the default LLM and returns the generated response using SecureLLM.

    Args:
        prompt (str): The user input to send to the model.
        temperature (float): Sampling temperature for response variation.
        max_tokens (int): Maximum number of tokens in the model's response.

    Returns:
        str: The content of the LLM's response.

    Raises:
        ImportError: If secure-llm is not installed
        ValueError: If VAULT_SECRET_KEY is not set
    """
    if not _SECURELLM_AVAILABLE:
        raise ImportError("secure-llm package not installed. Install with: pip install secure-llm")

    client = get_llm_client_instance()

    # Get default configuration and override with provided parameters
    config = get_default_generation_config({
        "temperature": temperature,
        "max_tokens": max_tokens
    })

    response = client.chat.completions.create(
        model=ModelConfig.DEFAULT_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        **config
    )

    return extract_response_content(response).strip()


def llm_chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 500,
    model_name: Optional[str] = None
) -> Optional[str]:
    """
    Sends a chat conversation to the LLM and returns the generated response.

    This function supports system messages and multi-turn conversations,
    making it suitable for the ENT surgical recommendation use case.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys.
                  Roles can be 'system', 'user', or 'assistant'.
        temperature: Sampling temperature for response variation. Default 0.2 for consistency.
        max_tokens: Maximum number of tokens in the model's response.
        model_name: Optional model name override.

    Returns:
        str: The content of the LLM's response, or None if an error occurred.

    Raises:
        ImportError: If secure-llm is not installed
        ValueError: If VAULT_SECRET_KEY is not set

    Example:
        >>> messages = [
        ...     {"role": "system", "content": "You are an expert otolaryngologist."},
        ...     {"role": "user", "content": "Should this patient have surgery?"}
        ... ]
        >>> response = llm_chat(messages)
    """
    if not _SECURELLM_AVAILABLE:
        raise ImportError("secure-llm package not installed. Install with: pip install secure-llm")

    try:
        model = model_name or ModelConfig.DEFAULT_LLM_MODEL
        client = get_llm_client_instance(model)

        # Get default configuration and override with provided parameters
        config = get_default_generation_config({
            "temperature": temperature,
            "max_tokens": max_tokens
        })

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **config
        )

        return extract_response_content(response).strip()

    except Exception as e:
        logger.error(f"SecureLLM API error: {e}")
        return None


def query_llm(
    prompt: str,
    system_message: str = "You are an expert otolaryngologist. Provide a surgical recommendation in the requested JSON format.",
    temperature: float = 0.2,
    max_tokens: int = 500,
    model_name: Optional[str] = None
) -> Optional[str]:
    """
    Query the LLM with a system message and user prompt.

    This is a convenience function that wraps llm_chat for simple query patterns
    commonly used in the ENT analysis pipeline.

    Args:
        prompt: The user prompt to send to the model.
        system_message: The system message to set the model's behavior.
        temperature: Sampling temperature. Default 0.2 for consistent medical recommendations.
        max_tokens: Maximum response tokens.
        model_name: Optional model name override.

    Returns:
        str: The LLM's response content, or None if an error occurred.
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    return llm_chat(messages, temperature=temperature, max_tokens=max_tokens, model_name=model_name)


class SecureLLMClient:
    """
    A client wrapper for SecureLLM that provides an interface compatible with
    the existing codebase patterns.

    This class can be used as a drop-in replacement where OpenAI client was used.

    Example:
        >>> client = SecureLLMClient()
        >>> response = client.query("What is the diagnosis?")
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the SecureLLM client.

        Args:
            model_name: Optional model name. Defaults to ModelConfig.DEFAULT_LLM_MODEL.
        """
        self.model_name = model_name or ModelConfig.DEFAULT_LLM_MODEL
        self._client = None

    @property
    def client(self):
        """Lazy initialization of the underlying client."""
        if self._client is None:
            self._client = get_llm_client_instance(self.model_name)
        return self._client

    def query(
        self,
        prompt: str,
        system_message: str = "You are an expert otolaryngologist. Provide a surgical recommendation in the requested JSON format.",
        temperature: float = 0.2,
        max_tokens: int = 500
    ) -> Optional[str]:
        """
        Query the LLM with a prompt.

        Args:
            prompt: The user prompt.
            system_message: System message for context.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.

        Returns:
            The LLM response or None on error.
        """
        return query_llm(
            prompt=prompt,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            model_name=self.model_name
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 500
    ) -> Optional[str]:
        """
        Send a chat conversation to the LLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.

        Returns:
            The LLM response or None on error.
        """
        return llm_chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model_name=self.model_name
        )
