import time

import httpx
import ollama
from pydantic import BaseModel

from mtg_drafting.config import LLMConfig

Message = dict[str, str]


class LLMClient:
    """Thin wrapper over an Ollama model that returns schema-validated responses.

    Every call uses Ollama's structured-output mode: the response is constrained to a
    pydantic model's JSON schema and parsed back into that model, so callers never deal
    with free-form text.

    Parameters
    ----------
    config : LLMConfig
        Model tag, host, and sampling settings.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = ollama.Client(host=config.host, timeout=config.timeout)

    def chat(self, messages: list[Message], schema: type[BaseModel]) -> str:
        """Send a chat request constrained to ``schema`` and return the raw reply.

        Generation is constrained to ``schema``'s JSON shape, but the reply is returned
        as raw text rather than parsed — the model can still truncate at the token
        limit, and the caller decides how strict to be about that.

        Transient transport failures (a dropped connection, a timeout) are retried with
        exponential backoff so a network blip cannot abort a long-running draft.

        Parameters
        ----------
        messages : list of dict
            Ollama chat messages, each with ``role`` and ``content``.
        schema : type
            Pydantic model whose JSON schema constrains generation.

        Returns
        -------
        str
            The model's raw reply text.

        Raises
        ------
        RuntimeError
            If transport errors persist past ``transport_retries`` attempts.
        """
        last_exc: Exception | None = None
        for attempt in range(1 + self.config.transport_retries):
            try:
                # Thinking needs an uncapped budget; -1 lets Ollama generate freely.
                num_predict = -1 if self.config.think else self.config.num_predict
                response = self._client.chat(
                    model=self.config.model,
                    messages=messages,
                    format=schema.model_json_schema(),
                    think=self.config.think,
                    options={
                        "temperature": self.config.temperature,
                        "num_ctx": self.config.num_ctx,
                        "num_predict": num_predict,
                    },
                )
                return response["message"]["content"]
            except (httpx.HTTPError, ollama.ResponseError) as exc:
                last_exc = exc
                time.sleep(min(2**attempt, 10))
        raise RuntimeError(
            f"Ollama request failed after {1 + self.config.transport_retries} attempts."
        ) from last_exc

    def ensure_model(self) -> None:
        """Raise a clear error if the configured model is not pulled locally.

        Raises
        ------
        RuntimeError
            If the model is missing or the Ollama server is unreachable.
        """
        try:
            local = {m.model for m in self._client.list().models}
        except Exception as exc:  # noqa: BLE001 - re-raised with actionable guidance
            raise RuntimeError(
                f"Cannot reach Ollama at {self.config.host or 'the default host'}. "
                "Is `ollama serve` running?"
            ) from exc
        if self.config.model not in local:
            raise RuntimeError(
                f"Model '{self.config.model}' is not available. "
                f"Pull it with: ollama pull {self.config.model}"
            )
