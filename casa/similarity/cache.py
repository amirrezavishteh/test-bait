"""Hash-keyed cache for (expensive) similarity evaluations.

Caching is keyed by a SHA-256 of ``(namespace, query, a, b)`` with ``a``/``b``
order-canonicalised so symmetric calls share an entry.  An in-memory dict backs
every lookup; an optional JSON file persists entries across runs (important for
the LLM-judge backend whose calls cost real money/time).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Dict, Optional


class SimilarityCache:
    """Thread-safe memory cache with optional JSON persistence."""

    def __init__(self, path: Optional[str] = None) -> None:
        """Initialise the cache, loading any existing file.

        Args:
            path: Optional JSON file path.  If given, it is created on first
                :meth:`flush` and reloaded here when present.
        """
        self._path = path
        self._lock = threading.Lock()
        self._mem: Dict[str, float] = {}
        self._dirty = False
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    self._mem = {k: float(v) for k, v in json.load(fh).items()}
            except (json.JSONDecodeError, OSError, ValueError):
                self._mem = {}

    @staticmethod
    def key(namespace: str, query: str, a: str, b: str) -> str:
        """Return the canonical cache key for a similarity call."""
        lo, hi = sorted((a, b))
        payload = "\x00".join((namespace, query, lo, hi))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[float]:
        """Return the cached value for ``key`` or ``None``."""
        with self._lock:
            return self._mem.get(key)

    def put(self, key: str, value: float) -> None:
        """Store ``value`` under ``key``."""
        with self._lock:
            self._mem[key] = float(value)
            self._dirty = True

    def flush(self) -> None:
        """Persist the cache to disk if a path was configured and it changed."""
        if not self._path:
            return
        with self._lock:
            if not self._dirty:
                return
            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._mem, fh)
            os.replace(tmp, self._path)
            self._dirty = False

    def __len__(self) -> int:
        return len(self._mem)
