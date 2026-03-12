"""
Sliding Window Rate Limiter — Redis-backed.

Limits: MAX_REQUESTS per WINDOW_SECONDS per IP address.
Returns 429 Too Many Requests when exceeded.
"""

import logging
import time
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import redis as redis_lib

from backend.app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

MAX_REQUESTS = 20     # per window
WINDOW_SECONDS = 60   # sliding window duration


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limiter using Redis.
    Only applies to /api/v1/query/ endpoints.
    """

    def __init__(self, app):
        super().__init__(app)
        try:
            self.redis = redis_lib.from_url(settings.redis_url, decode_responses=True)
            self.redis.ping()
            self.enabled = True
            logger.info(f"[RateLimit] Enabled: {MAX_REQUESTS} req/{WINDOW_SECONDS}s per IP")
        except Exception as e:
            self.enabled = False
            logger.warning(f"[RateLimit] Redis unavailable, rate limiting disabled: {e}")

    async def dispatch(self, request: Request, call_next) -> Response:
        # Only rate-limit query endpoint
        if not self.enabled or not request.url.path.startswith("/api/v1/query"):
            return await call_next(request)

        # Skip GET requests (cache stats, etc.)
        if request.method == "GET":
            return await call_next(request)

        ip = request.client.host or "unknown"
        key = f"ratelimit:{ip}"

        try:
            # Sliding window: increment + set TTL atomically
            pipe = self.redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, WINDOW_SECONDS)
            count, _ = pipe.execute()

            # Set headers like GitHub API does
            remaining = max(0, MAX_REQUESTS - count)
            headers = {
                "X-RateLimit-Limit": str(MAX_REQUESTS),
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Window": f"{WINDOW_SECONDS}s",
            }

            if count > MAX_REQUESTS:
                logger.warning(f"[RateLimit] BLOCKED {ip} ({count}/{MAX_REQUESTS})")
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "detail": f"Max {MAX_REQUESTS} requests per {WINDOW_SECONDS}s. Try again later.",
                        "retry_after": WINDOW_SECONDS,
                    },
                    headers=headers,
                )

            logger.debug(f"[RateLimit] {ip}: {count}/{MAX_REQUESTS}")

        except Exception as e:
            logger.warning(f"[RateLimit] Check failed (allowing request): {e}")

        response = await call_next(request)

        # Add rate limit headers to response
        try:
            for k, v in headers.items():
                response.headers[k] = v
        except Exception:
            pass

        return response
