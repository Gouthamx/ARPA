"""Global rate limiter for LLM API calls across all clients."""

import threading
import time
from typing import ClassVar


class GlobalRateLimiter:
    """Singleton rate limiter shared across all LLM client instances.
    
    This ensures the entire application respects a single rate limit,
    regardless of how many client instances exist or how many concurrent
    calls are made.
    
    Usage:
        # In any LLM client's _call() method:
        GlobalRateLimiter().wait()
        response = api_call(...)
    """
    
    _instance: ClassVar["GlobalRateLimiter | None"] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()
    
    def __new__(cls, min_interval: float = 2.2):
        """Create or return the singleton instance.
        
        Args:
            min_interval: Minimum seconds between any two API calls across
                         the entire application. Default 2.2s = ~27 req/min
                         (safe margin under Groq's 30 req/min limit).
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance.min_interval = min_interval
                    instance.last_call = 0.0
                    instance.call_lock = threading.Lock()
                    cls._instance = instance
        return cls._instance
    
    def wait(self) -> None:
        """Block until enough time has passed since the last API call.
        
        Thread-safe: Multiple threads will queue and wait in sequence.
        """
        with self.call_lock:
            elapsed = time.time() - self.last_call
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                time.sleep(sleep_time)
            self.last_call = time.time()
    
    def reset(self) -> None:
        """Reset the rate limiter (mainly for testing)."""
        with self.call_lock:
            self.last_call = 0.0
