import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest
import redis
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import redis_backup

REDIS_HOST = os.environ.get("ICDE_TEST_REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("ICDE_TEST_REDIS_PORT", "6379"))
REDIS_DBS = tuple(redis_backup.DEFAULT_REDIS_DATABASES)
DEFAULT_RUN_SECONDS = int(os.environ.get("ICDE_TEST_WORKER_SECONDS", "120"))


def _redis_server_available() -> bool:
    try:
        redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0).ping()
        return True
    except redis.ConnectionError:
        return False


REDIS_SERVER_AVAILABLE = _redis_server_available()
REDIS_UNAVAILABLE_REASON = "Redis server not available on localhost:6379"
requires_redis = pytest.mark.skipif(not REDIS_SERVER_AVAILABLE, reason=REDIS_UNAVAILABLE_REASON)


@pytest.fixture(scope="module")
def redis_clients():
    if not REDIS_SERVER_AVAILABLE:
        pytest.skip(REDIS_UNAVAILABLE_REASON)
    clients = {db: redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=db) for db in REDIS_DBS}
    for client in clients.values():
        client.flushdb()
    yield clients
    for client in clients.values():
        client.flushdb()


def _collect_counts() -> Dict[int, int]:
    return {db: redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=db).dbsize() for db in REDIS_DBS}


def _run_python_script(script: Path, *args: str, cwd: Path, timeout: float | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    pythonpath_parts = [str(REPO_ROOT)]
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath_parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    command = [sys.executable, str(script), *args]
    return subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
    )


def _load_worker_pids(work_dir: Path) -> Dict[str, Dict[str, object]]:
    pid_file = work_dir / "workers" / "worker_pids.json"
    if not pid_file.exists():
        return {}
    return json.loads(pid_file.read_text())


def test_build_redis_client_requires_host():
    with pytest.raises(ValueError):
        redis_backup.build_redis_client({"port": REDIS_PORT})


def test_build_redis_client_supports_credentials():
    client = redis_backup.build_redis_client(
        {
            "host": REDIS_HOST,
            "port": REDIS_PORT,
            "db": 0,
            "username": "demo",
            "password": "secret",
        }
    )
    kwargs = client.connection_pool.connection_kwargs
    assert kwargs["username"] == "demo"
    assert kwargs["password"] == "secret"
    assert kwargs["decode_responses"] is False


def test_build_redis_client_ignores_blank_username():
    client = redis_backup.build_redis_client(
        {"host": REDIS_HOST, "port": REDIS_PORT, "db": 0, "username": "", "password": ""}
    )
    kwargs = client.connection_pool.connection_kwargs
    assert not kwargs.get("username")
    assert not kwargs.get("password")


@pytest.fixture()
def sample_backup_payload():
    return {
        "metadata": {
            "created_at_utc": "2024-01-01T12:00:00Z",
            "source": {"host": "redis", "port": 6380, "db": 3, "username": "demo"},
            "key_count": 3,
            "type_summary": {"hash": 1, "string": 2},
        },
        "entries": [],
    }


def test_format_backup_summary_includes_metadata(sample_backup_payload):
    summary = redis_backup.format_backup_summary(sample_backup_payload)

    assert "Backup summary" in summary
    assert "Created at: 2024-01-01T12:00:00Z" in summary
    assert "Source: host=redis, port=6380, db=3, username=demo" in summary
    assert "Number of keys: 3" in summary
    assert "Type distribution:" in summary
    assert "  - hash: 1" in summary
    assert "  - string: 2" in summary


def test_format_backup_summary_handles_missing_types(sample_backup_payload):
    payload = sample_backup_payload
    payload["metadata"]["type_summary"] = {}
    payload["metadata"]["key_count"] = 0

    summary = redis_backup.format_backup_summary(payload)

    assert "No keys found." in summary
    assert "Type distribution" not in summary.splitlines()


def test_display_backup_summary_matches_formatted_output(sample_backup_payload, capsys):
    summary = redis_backup.format_backup_summary(sample_backup_payload)

    redis_backup.display_backup_summary(sample_backup_payload)

    captured = capsys.readouterr().out
    assert captured == summary + "\n"


def test_save_and_load_backup_roundtrip(tmp_path, sample_backup_payload):
    backup_path = tmp_path / "backup.json"

    redis_backup.save_backup_to_file(sample_backup_payload, backup_path)
    assert backup_path.exists()

    loaded = redis_backup.load_backup_from_file(backup_path)
    assert loaded == sample_backup_payload


def test_save_and_load_multi_database_manifest(tmp_path, sample_backup_payload):
    backups = {1: sample_backup_payload, 2: sample_backup_payload}

    directory = tmp_path / "backups"
    mapping = redis_backup.save_multi_database_backup_to_directory(
        backups, directory, file_prefix="snapshot"
    )

    assert sorted(mapping) == [1, 2]

    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["file_prefix"] == "snapshot"
    assert sorted(manifest["databases"]) == [1, 2]

    loaded = redis_backup.load_multi_database_backup_from_directory(directory)
    assert loaded == backups


def test_load_multi_database_backup_missing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError):
        redis_backup.load_multi_database_backup_from_directory(tmp_path)


def test_load_multi_database_backup_empty_manifest(tmp_path):
    directory = tmp_path / "backups"
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps({"files": {}}))

    with pytest.raises(ValueError):
        redis_backup.load_multi_database_backup_from_directory(directory)


@requires_redis
def test_backup_and_restore_roundtrip(tmp_path, capsys):
    source_config = {"host": REDIS_HOST, "port": REDIS_PORT, "db": 13}
    target_config = {"host": REDIS_HOST, "port": REDIS_PORT, "db": 14}

    source_client = redis_backup.build_redis_client(source_config)
    target_client = redis_backup.build_redis_client(target_config)
    source_client.flushdb()
    target_client.flushdb()

    source_client.set(b"plain", b"value")
    source_client.psetex(b"ephemeral", 5000, b"temp-value")
    source_client.hset(b"hash", {b"field1": b"value1", b"field2": b"value2"})
    source_client.lpush(b"numbers", b"one", b"two")
    source_client.sadd(b"letters", b"a", b"b")
    source_client.zadd(b"ranking", {b"alice": 1.0, b"bob": 2.0})

    backup = redis_backup.create_redis_backup(source_config, scan_count=2)
    assert backup["metadata"]["key_count"] == 6

    backup_path = tmp_path / "backup.json"
    redis_backup.save_backup_to_file(backup, backup_path)
    assert backup_path.exists()

    loaded_backup = redis_backup.load_backup_from_file(backup_path)
    assert loaded_backup == backup

    capsys.readouterr()
    redis_backup.display_backup_summary(backup)
    summary_output = capsys.readouterr().out
    assert "Number of keys" in summary_output

    restored = redis_backup.restore_redis_backup(backup, target_config, flush_target=True)
    assert restored == 6
    assert target_client.dbsize() == 6
    assert target_client.get(b"plain") == b"value"
    assert target_client.get(b"ephemeral") == b"temp-value"
    ttl_ms = target_client.pttl(b"ephemeral")
    # Restoring a key with a TTL is subject to small timing drift depending on
    # how long the round trip to Redis takes.  Allow a modest tolerance so the
    # test remains robust while still ensuring the original expiry is
    # preserved.
    assert ttl_ms == pytest.approx(5000, abs=100)
    assert target_client.hgetall(b"hash") == {b"field1": b"value1", b"field2": b"value2"}
    assert target_client.lrange(b"numbers", 0, -1) == [b"two", b"one"]
    assert target_client.smembers(b"letters") == {b"a", b"b"}
    assert target_client.zrange(b"ranking", 0, -1, withscores=True) == [
        (b"alice", 1.0),
        (b"bob", 2.0),
    ]


def _backup_all_databases(tmp_path: Path, capsys, prefix: str) -> Tuple[Dict[int, dict], Path]:
    backups = redis_backup.create_multi_database_backup(
        {"host": REDIS_HOST, "port": REDIS_PORT}, redis_backup.DEFAULT_REDIS_DATABASES
    )

    backup_dir = tmp_path / prefix
    path_map = redis_backup.save_multi_database_backup_to_directory(
        backups, backup_dir, file_prefix=prefix
    )
    manifest_path = backup_dir / "manifest.json"
    assert manifest_path.exists()

    loaded_backups = redis_backup.load_multi_database_backup_from_directory(backup_dir)
    assert loaded_backups == backups

    for db, backup in sorted(backups.items()):
        backup_path = path_map[db]
        assert backup_path.exists()
        capsys.readouterr()
        redis_backup.display_backup_summary(backup)
        summary = capsys.readouterr().out
        assert "Number of keys" in summary

    return backups, backup_dir


def _write_backup_report(
    backups: Dict[int, dict], report_path: Path, cycle_label: str
) -> Dict[str, Any]:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    database_summaries: Dict[str, Dict[str, Any]] = {}
    for db, backup in sorted(backups.items()):
        metadata = dict(backup.get("metadata") or {})
        database_summaries[str(db)] = {
            "key_count": int(metadata.get("key_count", 0)),
            "type_summary": dict(metadata.get("type_summary") or {}),
            "created_at_utc": metadata.get("created_at_utc"),
            "source": dict(metadata.get("source") or {}),
        }

    total_keys = sum(summary["key_count"] for summary in database_summaries.values())
    payload: Dict[str, Any] = {
        "cycle": cycle_label,
        "generated_at_utc": redis_backup.utc_now_iso(),
        "total_keys": total_keys,
        "databases": database_summaries,
    }
    report_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _restore_all_databases(backup_dir: Path) -> None:
    backups = redis_backup.load_multi_database_backup_from_directory(backup_dir)
    redis_backup.restore_multi_database_backup(
        backups, {"host": REDIS_HOST, "port": REDIS_PORT}, flush_each=True
    )


@requires_redis
def test_save_and_load_multi_database_directory(redis_clients, tmp_path):
    redis_clients[0].set(b"multi:test", b"value")
    redis_clients[1].lpush(b"multi:list", b"item")

    backups = redis_backup.create_multi_database_backup(
        {"host": REDIS_HOST, "port": REDIS_PORT}, databases=[0, 1]
    )

    backup_dir = tmp_path / "multi"
    path_map = redis_backup.save_multi_database_backup_to_directory(
        backups, backup_dir, file_prefix="snapshot"
    )

    assert sorted(path_map) == [0, 1]
    assert (backup_dir / "manifest.json").exists()

    loaded_backups = redis_backup.load_multi_database_backup_from_directory(backup_dir)
    assert loaded_backups == backups

    redis_clients[0].flushdb()
    redis_clients[1].flushdb()


@requires_redis
def test_pipeline_init_worker_backup_restore(redis_clients, tmp_path, capsys):
    init_args = [
        "MelbournePedestrian",
        "--class-label",
        "1",
    ]

    _run_python_script(
        REPO_ROOT / "init_aeon_univariate.py",
        *init_args,
        cwd=REPO_ROOT,
        timeout=900,
    )

    data_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
    assert data_client.get("label") is not None

    config = {
        "redis": {"host": REDIS_HOST, "port": REDIS_PORT},
        "workers": {
            "default": {
                "script": str(REPO_ROOT / "worker_cache.py"),
                "count": 1,
                "args": [],
            }
        },
        "logging": {"directory": str(tmp_path / "logs"), "cleanup_days": 7},
    }

    config_path = tmp_path / "worker_config.yaml"
    # Keep the configuration under the default filename so the launcher can be
    # executed exactly as ``python enhanced_launch_workers.py clean-restart``
    # without extra flags, mirroring the expected manual invocation.
    config_path.write_text(yaml.safe_dump(config))

    _run_python_script(
        REPO_ROOT / "enhanced_launch_workers.py",
        "clean-restart",
        cwd=tmp_path,
        timeout=120,
    )

    pid_snapshot = _load_worker_pids(tmp_path)
    assert pid_snapshot, "Worker manager should record running processes"

    time.sleep(DEFAULT_RUN_SECONDS)

    _run_python_script(
        REPO_ROOT / "enhanced_launch_workers.py",
        "stop",
        cwd=tmp_path,
        timeout=120,
    )

    time.sleep(5)
    assert not _load_worker_pids(tmp_path)

    first_counts = _collect_counts()
    assert sum(first_counts.values()) > 0

    backups_cycle1, backup_dir1 = _backup_all_databases(tmp_path, capsys, "backup_cycle1")

    reports_dir = tmp_path / "reports"
    report_cycle1_path = reports_dir / "cycle1_report.json"
    report_cycle1 = _write_backup_report(backups_cycle1, report_cycle1_path, "cycle1")
    assert report_cycle1_path.exists()
    assert report_cycle1["cycle"] == "cycle1"
    assert report_cycle1["total_keys"] == sum(
        payload["metadata"]["key_count"] for payload in backups_cycle1.values()
    )

    for client in redis_clients.values():
        client.flushdb()

    assert sum(_collect_counts().values()) == 0

    _restore_all_databases(backup_dir1)

    restored_counts = _collect_counts()
    assert restored_counts == first_counts

    _run_python_script(
        REPO_ROOT / "enhanced_launch_workers.py",
        "clean-restart",
        cwd=tmp_path,
        timeout=120,
    )

    time.sleep(DEFAULT_RUN_SECONDS)

    _run_python_script(
        REPO_ROOT / "enhanced_launch_workers.py",
        "stop",
        cwd=tmp_path,
        timeout=120,
    )

    time.sleep(5)
    assert not _load_worker_pids(tmp_path)

    second_counts = _collect_counts()
    assert sum(second_counts.values()) >= sum(first_counts.values())

    backups_cycle2, _backup_dir2 = _backup_all_databases(tmp_path, capsys, "backup_cycle2")

    report_cycle2_path = reports_dir / "cycle2_report.json"
    report_cycle2 = _write_backup_report(backups_cycle2, report_cycle2_path, "cycle2")
    assert report_cycle2_path.exists()
    assert report_cycle2["cycle"] == "cycle2"
    assert report_cycle2["total_keys"] == sum(
        payload["metadata"]["key_count"] for payload in backups_cycle2.values()
    )

    assert report_cycle2["total_keys"] >= report_cycle1["total_keys"]
