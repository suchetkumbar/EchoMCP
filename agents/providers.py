"""Provider resilience primitives for the synchronous agent worker."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Sequence

from agents.rag import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class _Circuit:
    failures: int = 0
    opened_until: float = 0.0
    probe_in_flight: bool = False


class LLMProviderChain:
    """Call configured LLM adapters in order with per-provider circuit breakers.

    Adapters must enforce their own network timeouts. This wrapper never logs
    prompts or provider exception text, which can contain user data/secrets.
    """

    def __init__(self, providers: Sequence[tuple[str, LLMProvider]], *, failure_threshold: int = 3,
                 cooldown_seconds: float = 30.0) -> None:
        if not providers:
            raise ValueError("at least one LLM provider is required")
        self.providers = tuple(providers)
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._circuits = {name: _Circuit() for name, _ in self.providers}
        self._lock = threading.Lock()

    def _claim(self, name: str) -> bool:
        now = time.monotonic()
        with self._lock:
            circuit = self._circuits[name]
            if circuit.opened_until > now:
                return False
            if circuit.opened_until:
                if circuit.probe_in_flight:
                    return False
                circuit.probe_in_flight = True
            return True

    def _success(self, name: str) -> None:
        with self._lock:
            circuit = self._circuits[name]
            circuit.failures = 0
            circuit.opened_until = 0.0
            circuit.probe_in_flight = False

    def _failure(self, name: str) -> None:
        with self._lock:
            circuit = self._circuits[name]
            circuit.failures += 1
            circuit.probe_in_flight = False
            if circuit.failures >= self.failure_threshold:
                circuit.opened_until = time.monotonic() + self.cooldown_seconds

    def complete(self, prompt: str) -> str:
        for name, provider in self.providers:
            if not self._claim(name):
                continue
            try:
                answer = provider.complete(prompt)
                if not isinstance(answer, str) or not answer.strip():
                    raise RuntimeError("provider returned an empty response")
                self._success(name)
                return answer.strip()
            except Exception:
                self._failure(name)
                logger.warning("LLM provider failed; trying configured fallback", extra={"provider": name})
        raise RuntimeError("all configured LLM providers are unavailable")

    def health(self) -> dict[str, str]:
        now = time.monotonic()
        with self._lock:
            return {name: ("open" if state.opened_until > now else "available")
                    for name, state in self._circuits.items()}
