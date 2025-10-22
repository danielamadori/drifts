"""Utility functions for exporting and restoring Redis databases."""
from __future__ import annotations

import base64
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import redis
from redis.exceptions import ResponseError

RedisConfig = Dict[str, Any]
BackupEntry = Dict[str, Any]
BackupPayload = Dict[str, Any]
SerializedValue = Dict[str, Any]


def build_redis_client(config: RedisConfig) -> redis.Redis:
    """Return a Redis client configured for binary-safe operations."""
    params = dict(config)
    if "host" not in params:
        raise ValueError("Configuration must include the 'host' key.")
    params.setdefault("port", 6379)
    params.setdefault("db", 0)
    if params.get("username") in (None, ""):
        params.pop("username", None)
    if params.get("password") in (None, ""):
        params.pop("password", None)
    params["decode_responses"] = False
    return redis.Redis(**params)


def encode_bytes(value: bytes) -> str:
    """Encode raw bytes as a base64 ASCII string."""
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if not isinstance(value, (bytes, bytearray)):
        raise TypeError("The value to encode must be bytes-like.")
    return base64.b64encode(value).decode("ascii")


def decode_bytes(value: str) -> bytes:
    """Decode a base64 ASCII string back to bytes."""
    return base64.b64decode(value.encode("ascii"))


def utc_now_iso() -> str:
    """Return the current UTC timestamp formatted as ISO-8601."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalise_type(key_type: Any) -> str:
    if isinstance(key_type, bytes):
        return key_type.decode()
    return str(key_type)


def _serialise_value(
    client: redis.Redis, key: bytes, key_type: str
) -> SerializedValue | None:
    """Return a serialisable payload representing the key's value."""

    def key_still_exists() -> bool:
        try:
            return bool(client.exists(key))
        except AttributeError:
            # Older redis client versions may expose ``exists`` as a command on pipelines
            # only.  Falling back to ``dump`` ensures we fail safe.
            dumped = client.dump(key)
            return dumped is not None
    dumped_value = client.dump(key)
    if dumped_value is None:
        if not key_still_exists():
            return None
        raise RuntimeError(f"Failed to serialise key {key!r}: empty dump result")
    return {"encoding": "dump", "data": encode_bytes(dumped_value)}


def create_redis_backup(config: RedisConfig, scan_count: int = 1000) -> BackupPayload:
    """Export the full content of a Redis instance into a serialisable payload."""
    client = build_redis_client(config)
    entries: List[BackupEntry] = []
    type_counter: Counter[str] = Counter()

    cursor = 0
    while True:
        cursor, keys = client.scan(cursor=cursor, count=scan_count)
        if not keys and cursor == 0:
            break
        for key in keys:
            key_type = _normalise_type(client.type(key))
            if key_type == "none":
                continue
            ttl_ms = client.pttl(key)
            if ttl_ms == -2:
                # Key removed between SCAN and TTL read
                continue
            ttl_field = int(ttl_ms) if ttl_ms is not None and ttl_ms > 0 else None
            serialized_value = _serialise_value(client, key, key_type)
            if serialized_value is None:
                continue
            entries.append(
                {
                    "key": encode_bytes(key),
                    "value": serialized_value,
                    "pttl": ttl_field,
                    "type": key_type,
                }
            )
            type_counter[key_type] += 1
        if cursor == 0:
            break

    entries.sort(key=lambda item: item["key"])

    metadata = {
        "created_at_utc": utc_now_iso(),
        "source": {
            "host": config.get("host"),
            "port": config.get("port", 6379),
            "db": config.get("db", 0),
        },
        "key_count": len(entries),
        "type_summary": dict(type_counter),
        "scan_count": scan_count,
    }
    username = config.get("username")
    if username not in (None, ""):
        metadata["source"]["username"] = username
    return {"metadata": metadata, "entries": entries}


def create_multi_database_backup(
    base_config: RedisConfig, databases: Iterable[int], scan_count: int = 1000
) -> Dict[int, BackupPayload]:
    """Create backups for multiple Redis databases using shared connection options.

    Args:
        base_config: Common Redis connection parameters (host, port, auth, etc.).
        databases: Iterable of database numbers to back up.
        scan_count: Hint for SCAN operations (forwarded to :func:`create_redis_backup`).

    Returns:
        Dictionary mapping each database number to its backup payload.
    """

    backups: Dict[int, BackupPayload] = {}
    shared_config = dict(base_config)

    for db in databases:
        db_config = dict(shared_config)
        db_config["db"] = db
        backups[db] = create_redis_backup(db_config, scan_count=scan_count)

    return backups


def save_backup_to_file(backup: BackupPayload, path: Path) -> Path:
    """Write a Redis backup payload to disk as JSON."""
    path = Path(path)
    path.write_text(json.dumps(backup, indent=2, sort_keys=True))
    return path


def load_backup_from_file(path: Path) -> BackupPayload:
    """Load a backup payload from a JSON file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def display_backup_summary(backup: BackupPayload) -> None:
    """Pretty-print a short summary of a backup payload."""
    metadata = backup.get("metadata", {})
    print("\nBackup summary")
    print("-" * 40)
    print(f"Created at: {metadata.get('created_at_utc', 'n/a')}")
    source = metadata.get("source", {})
    auth_fragment = ""
    username = source.get("username")
    if username not in (None, ""):
        auth_fragment = f", username={username}"
    print(
        "Source: host={host}, port={port}, db={db}{auth}".format(
            host=source.get("host", "n/a"),
            port=source.get("port", "n/a"),
            db=source.get("db", "n/a"),
            auth=auth_fragment,
        )
    )
    print(f"Number of keys: {metadata.get('key_count', 0)}")
    type_summary = metadata.get("type_summary") or {}
    if type_summary:
        print("Type distribution:")
        for redis_type, count in sorted(type_summary.items()):
            print(f"  - {redis_type}: {count}")
    else:
        print("No keys found.")
    print("-" * 40)


def get_database_overview(config: RedisConfig) -> Dict[str, Any]:
    """Return basic information about a Redis instance."""
    client = build_redis_client(config)
    info = client.info(section="server")
    return {
        "redis_version": info.get("redis_version", "?"),
        "redis_mode": info.get("redis_mode", "?"),
        "db": config.get("db", 0),
        "key_count": client.dbsize(),
    }


def restore_redis_backup(
    backup: BackupPayload, config: RedisConfig, *, flush_target: bool = False
) -> int:
    """Restore a backup payload into a Redis instance."""
    client = build_redis_client(config)
    if flush_target:
        client.flushdb()

    restored = 0
    for entry in backup.get("entries", []):
        key = decode_bytes(entry["key"])
        ttl_field = entry.get("pttl")
        ttl_ms = int(ttl_field) if ttl_field is not None and int(ttl_field) > 0 else None
        value_info: SerializedValue = entry.get("value", {})  # type: ignore[assignment]
        encoding = value_info.get("encoding")
        data = value_info.get("data")

        if encoding is None:
            encoding = "dump"
        if encoding != "dump":
            raise ValueError(f"Unsupported encoding '{encoding}' in backup entry")
        if data is None:
            raise ValueError("Missing raw dump payload for backup entry")

        raw_value = decode_bytes(data)
        ttl_for_restore = ttl_ms if ttl_ms is not None else 0
        try:
            client.restore(key, ttl_for_restore, raw_value, replace=True)
        except ResponseError as exc:
            message = str(exc)
            if "BUSYKEY" in message or "Target key name is busy" in message:
                client.delete(key)
                client.restore(key, ttl_for_restore, raw_value, replace=True)
            else:
                raise

        restored += 1
    return restored


def restore_multi_database_backup(
    backups: Mapping[int, BackupPayload],
    base_config: RedisConfig,
    *,
    flush_each: bool = False,
) -> Dict[int, int]:
    """Restore multiple Redis databases from backup payloads.

    Args:
        backups: Mapping from database number to backup payload.
        base_config: Common Redis connection parameters (host, port, auth, etc.).
        flush_each: Whether to flush each database before restoring into it.

    Returns:
        Dictionary mapping database numbers to the number of restored keys.
    """

    restored_counts: Dict[int, int] = {}
    shared_config = dict(base_config)

    for db, payload in backups.items():
        db_config = dict(shared_config)
        db_config["db"] = db
        restored_counts[db] = restore_redis_backup(payload, db_config, flush_target=flush_each)

    return restored_counts


__all__ = [
    "BackupEntry",
    "BackupPayload",
    "RedisConfig",
    "build_redis_client",
    "create_redis_backup",
    "decode_bytes",
    "display_backup_summary",
    "create_multi_database_backup",
    "encode_bytes",
    "get_database_overview",
    "load_backup_from_file",
    "restore_multi_database_backup",
    "restore_redis_backup",
    "save_backup_to_file",
    "utc_now_iso",
]
