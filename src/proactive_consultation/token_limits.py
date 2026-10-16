"""Token-limit conventions shared by training and inference.

None or a non-positive value disables a project-imposed limit. Generation
still stops at EOS and remains subject to the model's context capacity.
"""
import sys


def token_limit(value: int | None) -> int | None:
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


def generation_token_kwargs(max_new_tokens: int | None = None) -> dict:
    limit = token_limit(max_new_tokens)
    if limit is not None:
        return {"max_new_tokens": limit}
    # Omitting these arguments would restore Transformers' default length cap.
    return {"max_new_tokens": None, "max_length": sys.maxsize}
