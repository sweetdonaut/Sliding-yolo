"""
Path I/O abstraction for the Sliding-yolo pipeline.

Supports two URI schemes transparently:
- **Local paths**: anything not starting with ``s3://``.
- **S3 URIs**: ``s3://bucket/key``. Resolution order:
    1. If ``S3_MOCK_ROOT`` env var is set, the URI is rewritten to
       ``${S3_MOCK_ROOT}/bucket/key`` and read from disk. This lets you
       exercise the production code path against the existing local
       synthetic dataset with **no boto3 installed and no real bucket**.
    2. Otherwise, ``boto3`` is imported lazily and the object is fetched
       from real S3. Boto3 is an *optional* dependency — only required
       when you actually point at a live bucket.

For non-AWS S3-compatible storage (MinIO, Cloudflare R2, etc.), set the
``AWS_ENDPOINT_URL_S3`` env var (boto3 honors this natively as of botocore
1.28+).

To switch from mock to real S3, install boto3 and unset ``S3_MOCK_ROOT``;
no other code changes.
"""

from __future__ import annotations

import csv
import io
import os
from pathlib import Path
from urllib.parse import urlparse


# --------------------------------------------------------------------------- #
# Boto3 client config (lazy, one per process — boto3 clients are NOT
# thread-safe but our pipeline is single-threaded)
# --------------------------------------------------------------------------- #


_S3_CLIENT = None
_S3_CONFIG: dict = {}


def configure(
    endpoint_url: str | None = None,
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
    region: str | None = None,
) -> None:
    """Set the connection params used by every subsequent S3 call.

    Any argument left as ``None`` falls through to boto3's default
    credential chain (env vars ``AWS_ACCESS_KEY_ID`` /
    ``AWS_SECRET_ACCESS_KEY`` / ``AWS_DEFAULT_REGION``,
    ``~/.aws/credentials``, IAM role, etc.). This means you can mix and
    match — e.g., set the endpoint URL via CLI but keep the secret in an
    env var so it doesn't show up in ``ps``.

    Must be called **before** the first S3 call. Calling again resets the
    cached client so a new config takes effect.
    """
    global _S3_CONFIG, _S3_CLIENT
    cfg: dict = {}
    if endpoint_url:
        cfg["endpoint_url"] = endpoint_url
    if access_key_id:
        cfg["aws_access_key_id"] = access_key_id
    if secret_access_key:
        cfg["aws_secret_access_key"] = secret_access_key
    if region:
        cfg["region_name"] = region
    _S3_CONFIG = cfg
    _S3_CLIENT = None


def _s3_client():
    """Return a process-wide cached boto3 S3 client.

    Connection params come from (in order of precedence):
      1. ``configure()`` call (typically wired up from CLI flags)
      2. ``AWS_ENDPOINT_URL_S3`` / ``AWS_ENDPOINT_URL`` env vars (endpoint only)
      3. boto3's default credential chain (env vars, ~/.aws/credentials, IAM)
    """
    global _S3_CLIENT
    if _S3_CLIENT is not None:
        return _S3_CLIENT
    try:
        import boto3  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "boto3 is not installed. Either `pip install boto3` for real "
            "S3 access, or set S3_MOCK_ROOT for offline simulation."
        ) from e

    cfg = dict(_S3_CONFIG)
    # Fall back to env vars for endpoint when not configured explicitly.
    if "endpoint_url" not in cfg:
        endpoint = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get(
            "AWS_ENDPOINT_URL"
        )
        if endpoint:
            cfg["endpoint_url"] = endpoint

    _S3_CLIENT = boto3.client("s3", **cfg)
    return _S3_CLIENT


def _reset_s3_client_for_test() -> None:
    """Drop the cached client. Tests use this to swap in a mocked client."""
    global _S3_CLIENT
    _S3_CLIENT = None


# --------------------------------------------------------------------------- #
# URI helpers
# --------------------------------------------------------------------------- #


def is_s3(path) -> bool:
    """Return True if `path` is an s3:// URI."""
    return str(path).startswith("s3://")


def _split_s3(uri: str) -> tuple[str, str]:
    """Split `s3://bucket/key/parts` into ('bucket', 'key/parts')."""
    p = urlparse(uri)
    return p.netloc, p.path.lstrip("/")


def _mock_local_path(uri: str) -> str | None:
    """If S3_MOCK_ROOT is set and `uri` is s3://, return the mocked local
    path ``${S3_MOCK_ROOT}/bucket/key``. Otherwise return None."""
    root = os.environ.get("S3_MOCK_ROOT")
    if not root or not is_s3(uri):
        return None
    bucket, key = _split_s3(uri)
    return os.path.join(root, bucket, key) if key else os.path.join(root, bucket)


def stem(path) -> str:
    """Filename without extension — works for both ``s3://...`` and local."""
    name = str(path).rstrip("/").rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[0] if "." in name else name


def join_path(dir_path, name: str) -> str:
    """Append a file name to a directory path/URI."""
    dir_path = str(dir_path)
    if is_s3(dir_path):
        return f"{dir_path.rstrip('/')}/{name}"
    return str(Path(dir_path) / name)


# --------------------------------------------------------------------------- #
# Existence / listing
# --------------------------------------------------------------------------- #


def exists(path) -> bool:
    """True if a file exists at `path` (s3:// or local).

    For S3, only a 404/NoSuchKey/NotFound is treated as "missing"; auth
    failures, throttling, and other errors propagate so they don't get
    silently misreported as "file doesn't exist".
    """
    path = str(path)
    if is_s3(path):
        local = _mock_local_path(path)
        if local is not None:
            return Path(local).exists()
        from botocore.exceptions import ClientError  # type: ignore
        bucket, key = _split_s3(path)
        try:
            _s3_client().head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code in ("404", "NoSuchKey", "NotFound") or status == 404:
                return False
            raise  # auth, throttling, etc. — surface the real error
    return Path(path).exists()


def list_files(dir_path, suffix: str) -> list[str]:
    """List files in `dir_path` whose name ends with `suffix`.

    Returns full paths or URIs (matching the input scheme), sorted.
    """
    dir_path = str(dir_path)
    if is_s3(dir_path):
        local = _mock_local_path(dir_path)
        if local is not None:
            return sorted(
                f"{dir_path.rstrip('/')}/{p.name}"
                for p in Path(local).iterdir()
                if p.is_file() and p.name.endswith(suffix)
            )
        bucket, key = _split_s3(dir_path)
        prefix = key.rstrip("/") + "/" if key else ""
        s3 = _s3_client()
        out: list[str] = []
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                k = obj["Key"]
                if k.endswith(suffix):
                    out.append(f"s3://{bucket}/{k}")
        return sorted(out)
    return sorted(str(p) for p in Path(dir_path).glob(f"*{suffix}"))


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def open_bytes(path) -> bytes:
    """Read the entire file at `path` (s3:// or local) as bytes."""
    path = str(path)
    if is_s3(path):
        local = _mock_local_path(path)
        if local is not None:
            with open(local, "rb") as f:
                return f.read()
        bucket, key = _split_s3(path)
        return _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
    with open(path, "rb") as f:
        return f.read()


def open_text_csv(path) -> csv.DictReader:
    """Return a ``csv.DictReader`` over the file at `path`.

    The file is fetched eagerly into memory (cheap for ROI / val CSVs).
    """
    text = open_bytes(path).decode("utf-8")
    return csv.DictReader(io.StringIO(text))


# --------------------------------------------------------------------------- #
# Local materialization (for libraries that demand a real local path)
# --------------------------------------------------------------------------- #


def ensure_local(path) -> str:
    """Return a local filesystem path for `path`.

    - Local input: passthrough.
    - ``s3://`` with mock: returns the mock-translated local path.
    - ``s3://`` with boto3: downloads to a hashed temp file (cached so
      repeated calls reuse the same local copy).

    Use this for files consumed by third-party libraries (e.g. ultralytics
    ``YOLO()``) that don't accept in-memory bytes.
    """
    path = str(path)
    if not is_s3(path):
        return path
    local = _mock_local_path(path)
    if local is not None:
        return local
    import hashlib
    import tempfile

    bucket, key = _split_s3(path)
    cache_dir = Path(tempfile.gettempdir()) / "slidingyolo-s3-cache"
    cache_dir.mkdir(exist_ok=True)
    digest = hashlib.md5(path.encode()).hexdigest()[:12]
    name = key.rsplit("/", 1)[-1] if "/" in key else key
    local_path = cache_dir / f"{digest}_{name}"
    if not local_path.exists():
        _s3_client().download_file(bucket, key, str(local_path))
    return str(local_path)
