from __future__ import annotations

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def _brand_ip_key(request: Request) -> str:
    """Rate-limit key: (IP, brand) so each brand gets a separate 60/min bucket."""
    brand = request.path_params.get("brand", "_global")
    ip = get_remote_address(request)
    return f"{ip}:{brand}"


limiter = Limiter(key_func=_brand_ip_key)
