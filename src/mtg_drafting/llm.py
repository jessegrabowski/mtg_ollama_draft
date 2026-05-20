import time

import httpx
import ollama
from pydantic import BaseModel

from mtg_drafting.config import LLMConfig

Message = dict[str, str]

# Transport-level transient failures we retry. Distinct from Ollama 4xx responses
# (model not found, bad request) which are permanent - retrying those just delays
# the inevitable error and confuses the operator with backoff sleeps.
_TRANSIENT_TRANSPORT_ERRORS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)


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

    def chat(
        self,
        messages: list[Message],
        schema: type[BaseModel],
        *,
        num_predict: int | None = None,
    ) -> str:
        """Send a chat request constrained to ``schema`` and return the raw reply.

        Constrains generation to ``schema``'s JSON shape and returns the raw text
        without parsing — the model can still truncate at the token limit, and the
        caller decides how strict to be about that.

        Retries transient transport failures (a dropped connection, a timeout) with
        exponential backoff so a network blip cannot abort a long-running draft.

        Parameters
        ----------
        messages : list of dict
            Ollama chat messages, each with ``role`` and ``content``.
        schema : type
            Pydantic model whose JSON schema constrains generation.
        num_predict : int, optional
            Per-call override of the response token cap. Use for callers whose schema
            scales with input size (strategist classifying a 30-card pool, evaluator
            with its large verdict schema). When None, ``LLMConfig.num_predict``
            applies; when ``think`` is on, the cap is bypassed regardless. Default None.

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
        # Thinking needs an uncapped budget; -1 lets Ollama generate freely.
        if self.config.think:
            effective_num_predict = -1
        else:
            effective_num_predict = (
                num_predict if num_predict is not None else self.config.num_predict
            )
        for attempt in range(1 + self.config.transport_retries):
            try:
                response = self._client.chat(
                    model=self.config.model,
                    messages=messages,
                    format=schema.model_json_schema(),
                    think=self.config.think,
                    options={
                        "temperature": self.config.temperature,
                        "num_ctx": self.config.num_ctx,
                        "num_predict": effective_num_predict,
                    },
                )
                return response["message"]["content"]
            except _TRANSIENT_TRANSPORT_ERRORS as exc:
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
                f"Cannot reach Ollama at {self.config.host or 'the default host'} "
                f"({type(exc).__name__}: {exc}). Is `ollama serve` running?"
            ) from exc
        if self.config.model not in local:
            raise RuntimeError(
                f"Model '{self.config.model}' is not available. "
                f"Pull it with: ollama pull {self.config.model}"
            )
