import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict

import pytest
import redis
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import redis_backup

REDIS_HOST = os.environ.get("ICDE_TEST_REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("ICDE_TEST_REDIS_PORT", "6379"))
REDIS_DBS = tuple(range(0, 11))
DEFAULT_RUN_SECONDS = int(os.environ.get("ICDE_TEST_WORKER_SECONDS", "120"))


def _redis_server_available() -> bool:
    try:
        redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0).ping()
        return True
    except redis.ConnectionError:
        return False


pytestmark = pytest.mark.skipif(
    not _redis_server_available(), reason="Redis server not available on localhost:6379"
)


@pytest.fixture(scope="module")
def redis_clients():
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
    assert target_client.pttl(b"ephemeral") == 5000
    assert target_client.hgetall(b"hash") == {b"field1": b"value1", b"field2": b"value2"}
    assert target_client.lrange(b"numbers", 0, -1) == [b"two", b"one"]
    assert target_client.smembers(b"letters") == {b"a", b"b"}
    assert target_client.zrange(b"ranking", 0, -1, withscores=True) == [
        (b"alice", 1.0),
        (b"bob", 2.0),
    ]


def _backup_all_databases(tmp_path: Path, capsys, prefix: str) -> Dict[int, dict]:
    backups = redis_backup.create_multi_database_backup(
        {"host": REDIS_HOST, "port": REDIS_PORT}, REDIS_DBS
    )

    for db, backup in backups.items():
        backup_path = tmp_path / f"{prefix}_db{db}.json"
        redis_backup.save_backup_to_file(backup, backup_path)
        assert backup_path.exists()
        capsys.readouterr()
        redis_backup.display_backup_summary(backup)
        summary = capsys.readouterr().out
        assert "Number of keys" in summary

    return backups


def _restore_all_databases(backups: Dict[int, dict]) -> None:
    redis_backup.restore_multi_database_backup(
        backups, {"host": REDIS_HOST, "port": REDIS_PORT}, flush_each=True
    )


def test_pipeline_init_worker_backup_restore(redis_clients, tmp_path, capsys):
    init_args = [
        "MelbournePedestrian",
        "--class-label",
        "1",
        "--n-estimators",
        "1",
        "--max-depth",
        "2",
        "--random-state",
        "0",
        "--sample-percentage",
        "100",
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
    config_path.write_text(yaml.safe_dump(config))

    _run_python_script(
        REPO_ROOT / "enhanced_launch_workers.py",
        "--config",
        str(config_path),
        "clean-restart",
        cwd=tmp_path,
        timeout=120,
    )

    pid_snapshot = _load_worker_pids(tmp_path)
    assert pid_snapshot, "Worker manager should record running processes"

    time.sleep(DEFAULT_RUN_SECONDS)

    _run_python_script(
        REPO_ROOT / "enhanced_launch_workers.py",
        "--config",
        str(config_path),
        "stop",
        cwd=tmp_path,
        timeout=120,
    )

    time.sleep(5)
    assert not _load_worker_pids(tmp_path)

    first_counts = _collect_counts()
    assert sum(first_counts.values()) > 0

    backups_cycle1 = _backup_all_databases(tmp_path, capsys, "backup_cycle1")

    for client in redis_clients.values():
        client.flushdb()

    assert sum(_collect_counts().values()) == 0

    _restore_all_databases(backups_cycle1)

    restored_counts = _collect_counts()
    assert restored_counts == first_counts

    _run_python_script(
        REPO_ROOT / "enhanced_launch_workers.py",
        "--config",
        str(config_path),
        "clean-restart",
        cwd=tmp_path,
        timeout=120,
    )

    time.sleep(DEFAULT_RUN_SECONDS)

    _run_python_script(
        REPO_ROOT / "enhanced_launch_workers.py",
        "--config",
        str(config_path),
        "stop",
        cwd=tmp_path,
        timeout=120,
    )

    time.sleep(5)
    assert not _load_worker_pids(tmp_path)

    second_counts = _collect_counts()
    assert sum(second_counts.values()) >= sum(first_counts.values())

    backups_cycle2 = _backup_all_databases(tmp_path, capsys, "backup_cycle2")

    total_keys_cycle1 = sum(payload["metadata"]["key_count"] for payload in backups_cycle1.values())
    total_keys_cycle2 = sum(payload["metadata"]["key_count"] for payload in backups_cycle2.values())
    assert total_keys_cycle2 >= total_keys_cycle1
