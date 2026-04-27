import time
import functools
from typing import Any, Callable, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


def retry_mlflow(retries: int = 3, delay: float = 1.0) -> Callable[[F], F]:
    """Decorador profesional para reintentos de conexión [cite: 2026-03-04]."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception = Exception("Unknown error")
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if i < retries - 1:
                        time.sleep(delay * (2 ** i))  # Backoff exponencial
            raise last_exception
        return cast(F, wrapper)
    return decorator
