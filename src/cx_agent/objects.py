"""Object storage for conversation documents (transcript exports,
attachments) over S3-compatible storage — SeaweedFS's S3 gateway
in the compose topology, real S3 in production.

A small SigV4 client over httpx: the only operations the blueprint
needs are ensure-bucket, put and get, and a dependency on a full
SDK would be heavier than the job. When S3 is not configured the
functions report so and the API degrades honestly (readiness shows
it) rather than pretending documents were stored.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from datetime import datetime, timezone

import httpx

from .config import S3Config

log = logging.getLogger("cx_agent.objects")


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def _headers(cfg: S3Config, method: str, path: str, body: bytes) -> dict:
    now = datetime.now(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")
    payload_hash = hashlib.sha256(body).hexdigest()
    host = cfg.endpoint_url.split("://", 1)[-1]
    canonical = f"{method}\n{path}\n\nhost:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n\nhost;x-amz-content-sha256;x-amz-date\n{payload_hash}"
    scope = f"{date_stamp}/{cfg.region}/s3/aws4_request"
    to_sign = f"AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{hashlib.sha256(canonical.encode()).hexdigest()}"
    k = _sign(("AWS4" + cfg.secret_key).encode(), date_stamp)
    k = _sign(k, cfg.region)
    k = _sign(k, "s3")
    k = _sign(k, "aws4_request")
    signature = hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()
    return {
        "Authorization": (f"AWS4-HMAC-SHA256 Credential={cfg.access_key}/{scope}, "
                          f"SignedHeaders=host;x-amz-content-sha256;x-amz-date, "
                          f"Signature={signature}"),
        "x-amz-date": amz_date,
        "x-amz-content-sha256": payload_hash,
    }


def _url(cfg: S3Config, key: str = "") -> tuple[str, str]:
    path = f"/{cfg.bucket}/{key}" if key else f"/{cfg.bucket}"
    return f"{cfg.endpoint_url.rstrip('/')}{path}", path


def ensure_bucket(cfg: S3Config) -> bool:
    if not cfg.configured:
        return False
    url, path = _url(cfg)
    try:
        resp = httpx.put(url, headers=_headers(cfg, "PUT", path, b""), timeout=10)
        return resp.status_code in (200, 409)
    except Exception as exc:
        log.warning("ensure_bucket failed", extra={"fields": {"error": str(exc)}})
        return False


def put_text(cfg: S3Config, key: str, text: str) -> bool:
    if not cfg.configured:
        return False
    ensure_bucket(cfg)
    url, path = _url(cfg, key)
    body = text.encode()
    try:
        resp = httpx.put(url, headers=_headers(cfg, "PUT", path, body),
                         content=body, timeout=10)
        return resp.status_code == 200
    except Exception as exc:
        log.warning("put failed", extra={"fields": {"error": str(exc)}})
        return False


def get_text(cfg: S3Config, key: str) -> str | None:
    if not cfg.configured:
        return None
    url, path = _url(cfg, key)
    try:
        resp = httpx.get(url, headers=_headers(cfg, "GET", path, b""), timeout=10)
        return resp.text if resp.status_code == 200 else None
    except Exception as exc:
        log.warning("get failed", extra={"fields": {"error": str(exc)}})
        return None
