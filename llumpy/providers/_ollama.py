"""
File: _ollama.py

Description: Client for interacting with ollama server

@author Derek Garcia
"""
import os

import httpx

from ._openai import OpenAIClient, AsyncOpenAIClient
from ..core import ModelNotFoundError

# ollama details
DEFAULT_SERVER_URL = "http://localhost:11434"
DEFAULT_MODEL_TAG = "latest"

OLLAMA_SERVER_URL_ENV = "OLLAMA_SERVER_URL"

# endpoints
MODEL_LIBRARY = "https://ollama.com/library"
MODEL_DOWNLOAD_ENDPOINT = "api/pull"
MODEL_VIEW_ENDPOINT = "api/show"


class InvalidOllamaServerError(ConnectionError):
    """Failed to connect to Ollama server"""

    def __init__(self, ollama_url: str):
        """
        Failed to connect to Ollama server

        :param ollama_url: URL of ollama server
        """
        super().__init__(f"Failed to connect to ollama server at '{ollama_url}'")
        self.ollama_url = ollama_url


class ModelNotDownloadedError(RuntimeError):
    """Attempted to prompt a model that is not available on the server"""

    def __init__(self, ollama_url: str, model: str):
        """
        Failed to connect to Ollama server

        :param ollama_url: URL of ollama server
        :param model: Requested model
        """
        super().__init__(f"Model '{model}' is not available at '{ollama_url}' - has the model been downloaded?")
        self.ollama_url = ollama_url
        self.model = model


class _OllamaClientMixin:
    """Mixin for ollama server and model details"""

    def _init_ollama(self, model_name: str, model_tag: str, server_url: str | None = None) -> str:
        """
        Create new Ollama Client
        Server URL is resolved from the param, then the OLLAMA_SERVER_URL env var, then the default

        :param model_name: Name of model to use
        :param model_tag: Model tag to use
        :param server_url: Optional URL of the ollama server (Default: http://localhost:11434)
        :returns: Base url to use for OpenAI API
        """
        self._model_name = model_name
        self._model_tag = model_tag
        self._ollama_server = server_url or os.getenv(OLLAMA_SERVER_URL_ENV, DEFAULT_SERVER_URL)
        return f"{self._ollama_server}/v1"  # base_url for OpenAI client

    @property
    def model_name(self) -> str:
        """
        :return: Name of model
        """
        return self._model_name

    @property
    def model_tag(self) -> str:
        """
        :return: Tag of model
        """
        return self._model_tag


class OllamaClient(_OllamaClientMixin, OpenAIClient):
    """
    Interface for using Ollama API
    """

    def __init__(self, model_name: str, model_tag: str | None = None, server_url: str | None = None):
        """
        Create new Ollama Client
        Server URL is resolved from the param, then the OLLAMA_SERVER_URL env var, then the default

        :param model_name: Name of model to use
        :param model_tag: Optional model tag (Default: latest)
        :param server_url: Optional URL of the ollama server (Default: http://localhost:11434)
        """
        model_tag = model_tag or DEFAULT_MODEL_TAG
        base_url = self._init_ollama(model_name, model_tag, server_url)
        super().__init__(model=f"{model_name}:{model_tag}", api_key="ollama", base_url=base_url)

    def _is_model_downloaded(self) -> bool:
        """
        Check if the model is already downloaded / available locally

        :return: True if downloaded, false otherwise
        """
        r = httpx.post(f"{self._ollama_server}/{MODEL_VIEW_ENDPOINT}", json={"model": self._model})
        return r.status_code == 200

    def validate(self) -> None:
        """
        Ensure the ollama server is available and model is available

        :raises InvalidOllamaServerError: If could not connect to the Ollama server
        :raises ModelNotFoundError: If could not find requested model in Ollama library
        """
        # verify ollama server up
        try:
            r = httpx.get(self._ollama_server)
            r.raise_for_status()
        except httpx.ConnectError as e:
            raise InvalidOllamaServerError(self._ollama_server) from e

        # verify model exists
        try:
            r = httpx.get(f"{MODEL_LIBRARY}/{self._model}")
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ModelNotFoundError('Ollama', self._model) from e
            raise

    def download_model(self, force_download: bool = False) -> None:
        """
        Download a model locally from Ollama library
        See https://ollama.com/search for all available models

        :param force_download: Force download even if model already downloaded (Default: False)
        :raises ModelNotFoundError: If could not find requested model in Ollama library
        :raises HTTPError: If fail to download model
        """
        # exit early if already downloaded
        if not force_download and self._is_model_downloaded():
            return

        try:
            r = httpx.post(f"{self._ollama_server}/{MODEL_DOWNLOAD_ENDPOINT}", json={"model": self._model})
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ModelNotFoundError('Ollama', self._model) from e
            raise

    def wakeup(self) -> None:
        """
        Send a simple wakeup prompt to warm up the LLM.
        Useful before actually user prompting to prevent delay on he first prompt

        :raises ModelNotDownloadedError: If the request model has not been downloaded or available
        """
        # model not available
        if not self._is_model_downloaded():
            raise ModelNotDownloadedError(self._ollama_server, self.model)
        self.prompt_one('hi', max_completion_tokens=1)


class AsyncOllamaClient(_OllamaClientMixin, AsyncOpenAIClient):
    """
    Async interface for using Ollama API
    """

    def __init__(self, model_name: str, model_tag: str | None = None, server_url: str | None = None):
        """
        Create new Ollama Client
        Server URL is resolved from the param, then the OLLAMA_SERVER_URL env var, then the default

        :param model_name: Name of model to use
        :param model_tag: Optional model tag (Default: latest)
        :param server_url: Optional URL of the ollama server (Default: http://localhost:11434)
        """
        model_tag = model_tag or DEFAULT_MODEL_TAG
        base_url = self._init_ollama(model_name, model_tag, server_url)
        super().__init__(model=f"{model_name}:{model_tag}", api_key="ollama", base_url=base_url)

    async def _is_model_downloaded(self) -> bool:
        """
        Check if the model is already downloaded / available locally

        :return: True if downloaded, false otherwise
        """
        async with httpx.AsyncClient() as client:
            r = await client.post(f"{self._ollama_server}/{MODEL_VIEW_ENDPOINT}", json={"model": self._model})
            return r.status_code == 200

    async def validate(self) -> None:
        """
        Ensure the ollama server is available and model is available

        :raises InvalidOllamaServerError: If could not connect to the Ollama server
        :raises ModelNotFoundError: If could not find requested model in Ollama library
        """
        async with httpx.AsyncClient() as client:
            # verify ollama server up
            try:
                r = await client.get(self._ollama_server)
                r.raise_for_status()
            except httpx.ConnectError as e:
                raise InvalidOllamaServerError(self._ollama_server) from e

            # verify model exists
            try:
                r = await client.get(f"{MODEL_LIBRARY}/{self._model}")
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    raise ModelNotFoundError('Ollama', self._model) from e
                raise

    async def download_model(self, force_download: bool = False) -> None:
        """
        Download a model locally from Ollama library
        See https://ollama.com/search for all available models

        :param force_download: Force download even if model already downloaded (Default: False)
        :raises ModelNotFoundError: If could not find requested model in Ollama library
        :raises HTTPError: If fail to download model
        """
        # exit early if already downloaded
        if not force_download and await self._is_model_downloaded():
            return

        async with httpx.AsyncClient() as client:
            try:
                r = await client.post(f"{self._ollama_server}/{MODEL_DOWNLOAD_ENDPOINT}", json={"model": self._model})
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    raise ModelNotFoundError('Ollama', self._model) from e
                raise

    async def wakeup(self) -> None:
        """
        Send a simple wakeup prompt to warm up the LLM.
        Useful before actually user prompting to prevent delay on he first prompt

        :raises ModelNotDownloadedError: If the request model has not been downloaded or available
        """
        # model not available
        if not await self._is_model_downloaded():
            raise ModelNotDownloadedError(self._ollama_server, self.model)
        await self.prompt_one('hi', max_completion_tokens=1)
