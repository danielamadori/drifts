import base64
import errno
import importlib
import io
import itertools
import json
import os
import random
import signal
import sys
import threading
import time
import types
from contextlib import redirect_stderr, redirect_stdout
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np
import pytest
from redis.exceptions import ResponseError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import redis_backup
from tests import dummy_worker


def _to_bytes(value):
    if isinstance(value, bytes):
        return value
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, (int, float)):
        return str(value).encode("utf-8")
    raise TypeError(f"Unsupported type for bytes conversion: {type(value)!r}")


def _b64encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"))


class FakeRedis:
    def __init__(self) -> None:
        self._entries: Dict[bytes, Dict[str, object]] = {}
        self._ttls: Dict[bytes, int | None] = {}

    def ping(self) -> bool:
        return True

    # Mutation helpers -------------------------------------------------
    def flushdb(self) -> None:
        self._entries.clear()
        self._ttls.clear()

    def delete(self, *keys) -> int:
        removed = 0
        for key in keys:
            key_b = _to_bytes(key)
            removed += int(key_b in self._entries)
            self._entries.pop(key_b, None)
            self._ttls.pop(key_b, None)
        return removed

    def exists(self, key) -> int:
        return int(_to_bytes(key) in self._entries)

    def set(self, key, value) -> bool:
        key_b = _to_bytes(key)
        value_b = _to_bytes(value)
        self._entries[key_b] = {"type": "string", "value": value_b}
        self._ttls[key_b] = None
        return True

    def psetex(self, key, ttl_ms: int, value) -> bool:
        key_b = _to_bytes(key)
        value_b = _to_bytes(value)
        self._entries[key_b] = {"type": "string", "value": value_b}
        self._ttls[key_b] = int(ttl_ms)
        return True

    def pexpire(self, key, ttl_ms: int) -> int:
        key_b = _to_bytes(key)
        if key_b not in self._entries:
            return 0
        self._ttls[key_b] = int(ttl_ms)
        return 1

    def get(self, key):
        key_b = _to_bytes(key)
        entry = self._entries.get(key_b)
        if not entry or entry["type"] != "string":
            return None
        return entry["value"]

    def hset(self, key, mapping) -> int:
        key_b = _to_bytes(key)
        data = { _to_bytes(field): _to_bytes(val) for field, val in mapping.items() }
        self._entries[key_b] = {"type": "hash", "value": data}
        self._ttls[key_b] = None
        return len(data)

    def hgetall(self, key):
        key_b = _to_bytes(key)
        entry = self._entries.get(key_b)
        if entry and entry["type"] == "hash":
            return dict(entry["value"])  # type: ignore[return-value]
        return {}

    def lpush(self, key, *values) -> int:
        key_b = _to_bytes(key)
        entry = self._entries.get(key_b)
        if entry is None:
            entry = {"type": "list", "value": []}
            self._entries[key_b] = entry
            self._ttls[key_b] = None
        elif entry["type"] != "list":
            raise TypeError("Cannot LPUSH into a non-list key")
        list_values = entry.setdefault("value", [])
        for value in values:
            list_values.insert(0, _to_bytes(value))
        return len(list_values)

    def rpush(self, key, *values) -> int:
        key_b = _to_bytes(key)
        entry = self._entries.get(key_b)
        if entry is None:
            entry = {"type": "list", "value": []}
            self._entries[key_b] = entry
            self._ttls[key_b] = None
        elif entry["type"] != "list":
            raise TypeError("Cannot RPUSH into a non-list key")
        list_values: List[bytes] = entry.setdefault("value", [])  # type: ignore[assignment]
        for value in values:
            list_values.append(_to_bytes(value))
        return len(list_values)

    def lrange(self, key, start: int, end: int):
        key_b = _to_bytes(key)
        entry = self._entries.get(key_b)
        if not entry or entry["type"] != "list":
            return []
        values: List[bytes] = entry["value"]  # type: ignore[assignment]
        if end == -1:
            end = len(values) - 1
        return values[start : end + 1]

    def sadd(self, key, *values) -> int:
        key_b = _to_bytes(key)
        entry = self._entries.get(key_b)
        if entry is None:
            entry = {"type": "set", "value": set()}
            self._entries[key_b] = entry
            self._ttls[key_b] = None
        elif entry["type"] != "set":
            raise TypeError("Cannot SADD into a non-set key")
        value_set = entry.setdefault("value", set())
        before = len(value_set)
        value_set.update({_to_bytes(value) for value in values})
        return len(value_set) - before

    def smembers(self, key):
        key_b = _to_bytes(key)
        entry = self._entries.get(key_b)
        if entry and entry["type"] == "set":
            return set(entry["value"])  # type: ignore[return-value]
        return set()

    def zadd(self, key, mapping) -> int:
        key_b = _to_bytes(key)
        entry = self._entries.get(key_b)
        if entry is None:
            entry = {"type": "zset", "value": {}}
            self._entries[key_b] = entry
            self._ttls[key_b] = None
        elif entry["type"] != "zset":
            raise TypeError("Cannot ZADD into a non-zset key")
        zmap: Dict[bytes, float] = entry.setdefault("value", {})  # type: ignore[assignment]
        for member, score in mapping.items():
            zmap[_to_bytes(member)] = float(score)
        return len(zmap)

    def zrange(self, key, start: int, end: int, withscores: bool = False):
        key_b = _to_bytes(key)
        entry = self._entries.get(key_b)
        if not entry or entry["type"] != "zset":
            return []
        zmap: Dict[bytes, float] = entry["value"]  # type: ignore[assignment]
        items = sorted(zmap.items(), key=lambda item: (item[1], item[0]))
        if end == -1:
            end = len(items) - 1
        slice_items = items[start : end + 1]
        if withscores:
            return [(member, score) for member, score in slice_items]
        return [member for member, _ in slice_items]

    # Introspection helpers -------------------------------------------
    def dbsize(self) -> int:
        return len(self._entries)

    def info(self, section: str = "server") -> Dict[str, str]:
        return {"redis_version": "fake", "redis_mode": "standalone"}

    def scan(self, cursor: int = 0, match: str | None = None, count: int = 10) -> Tuple[int, Iterable[bytes]]:
        keys = sorted(self._entries.keys())
        if match:
            filtered = []
            for key in keys:
                key_text = key.decode() if isinstance(key, bytes) else key
                if fnmatch(key_text, match):
                    filtered.append(key)
            keys = filtered
        return 0, keys

    def randomkey(self):
        if not self._entries:
            return None
        return random.choice(list(self._entries.keys()))

    def type(self, key) -> str:
        key_b = _to_bytes(key)
        entry = self._entries.get(key_b)
        return entry["type"] if entry else "none"

    def pttl(self, key) -> int:
        key_b = _to_bytes(key)
        if key_b not in self._entries:
            return -2
        ttl = self._ttls.get(key_b)
        return ttl if ttl is not None else -1

    # Backup primitives ------------------------------------------------
    def dump(self, key) -> bytes | None:
        key_b = _to_bytes(key)
        entry = self._entries.get(key_b)
        if entry is None:
            return None
        payload = {
            "type": entry["type"],
            "value": self._encode_value(entry["type"], entry["value"]),
        }
        return json.dumps(payload, sort_keys=True).encode("utf-8")

    def restore(self, key, ttl_ms: int, value: bytes, replace: bool = False) -> bool:
        key_b = _to_bytes(key)
        if not replace and key_b in self._entries:
            raise ResponseError("BUSYKEY Target key name is busy")
        payload = json.loads(value.decode("utf-8"))
        value_type = payload["type"]
        decoded_value = self._decode_value(value_type, payload["value"])
        self._entries[key_b] = {"type": value_type, "value": decoded_value}
        ttl_value = int(ttl_ms) if ttl_ms and int(ttl_ms) > 0 else None
        self._ttls[key_b] = ttl_value
        return True

    # Internal serialisation helpers ----------------------------------
    def _encode_value(self, value_type: str, value) -> object:
        if value_type == "string":
            return _b64encode(value)
        if value_type == "hash":
            return { _b64encode(k): _b64encode(v) for k, v in value.items() }
        if value_type == "set":
            return sorted(_b64encode(item) for item in value)
        if value_type == "list":
            return [_b64encode(item) for item in value]
        if value_type == "zset":
            return [[_b64encode(member), score] for member, score in sorted(value.items())]
        raise ValueError(f"Unsupported value type: {value_type}")

    def _decode_value(self, value_type: str, payload) -> object:
        if value_type == "string":
            return _b64decode(payload)
        if value_type == "hash":
            return { _b64decode(k): _b64decode(v) for k, v in payload.items() }
        if value_type == "set":
            return { _b64decode(item) for item in payload }
        if value_type == "list":
            return [_b64decode(item) for item in payload]
        if value_type == "zset":
            return { _b64decode(member): float(score) for member, score in payload }
        raise ValueError(f"Unsupported value type: {value_type}")


class FakeRedisFactory:
    def __init__(self) -> None:
        self.instances: Dict[Tuple[str, int, int], FakeRedis] = {}
        self.last_kwargs: Dict[str, Any] | None = None

    def __call__(self, **kwargs) -> FakeRedis:
        self.last_kwargs = dict(kwargs)
        key = (
            kwargs.get("host", "localhost"),
            kwargs.get("port", 6379),
            kwargs.get("db", 0),
        )
        instance = self.instances.get(key)
        if instance is None:
            instance = FakeRedis()
            self.instances[key] = instance
        return instance


@pytest.fixture()
def fake_redis(monkeypatch):
    factory = FakeRedisFactory()
    import redis as redis_module

    monkeypatch.setattr(redis_backup.redis, "Redis", factory)
    monkeypatch.setattr(redis_module, "Redis", factory)
    return factory


def _run_cli(main_callable: Callable[[], int], argv: List[str]) -> None:
    original_argv = sys.argv[:]
    sys.argv = argv
    try:
        result = main_callable()
    finally:
        sys.argv = original_argv
    assert result == 0


def _install_aeon_stub(monkeypatch):
    datasets_module = types.ModuleType("aeon.datasets")

    def load_classification(dataset_name, split="train"):
        rng = np.random.default_rng(123 if split == "train" else 456)
        n_samples = 6 if split == "train" else 4
        data = rng.normal(size=(n_samples, 1, 4)).astype(np.float32)
        labels = np.array(["1" if i % 2 == 0 else "0" for i in range(n_samples)], dtype=object)
        return data, labels

    datasets_module.load_classification = load_classification
    aeon_module = types.ModuleType("aeon")
    aeon_module.datasets = datasets_module

    monkeypatch.setitem(sys.modules, "aeon", aeon_module)
    monkeypatch.setitem(sys.modules, "aeon.datasets", datasets_module)


class _FakeWorkerProcess:
    def __init__(self, pid: int, cmd: List[str], stdout_handle, controller: "_FakeWorkerController") -> None:
        self.pid = pid
        self._cmd = cmd
        self._stdout_handle = stdout_handle
        self._controller = controller
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_worker, name=f"fake-worker-{pid}", daemon=True)
        self._thread.start()

    def _run_worker(self) -> None:
        script_path = Path(self._cmd[1]) if len(self._cmd) > 1 else Path(dummy_worker.__file__)
        args = self._cmd[2:]

        previous_event = dummy_worker.SHUTDOWN_EVENT
        dummy_worker.SHUTDOWN_EVENT = self._stop_event

        previous_argv = sys.argv[:]
        sys.argv = [str(script_path)] + list(args)

        stream = self._stdout_handle or io.StringIO()

        try:
            with redirect_stdout(stream), redirect_stderr(stream):
                try:
                    dummy_worker.main()
                except SystemExit as exc:
                    if exc.code not in (0, None):
                        raise
        finally:
            if self._stdout_handle:
                self._stdout_handle.flush()
            dummy_worker.SHUTDOWN_EVENT = previous_event
            sys.argv = previous_argv
            self._controller.unregister(self.pid)

    def send_signal(self, sig: int) -> None:
        self._stop_event.set()
        # Allow the worker thread to react to the shutdown request.
        self._thread.join(timeout=2)

    def is_running(self) -> bool:
        return self._thread.is_alive()


class _FakeWorkerController:
    def __init__(self) -> None:
        self._pid_iter = itertools.count(1000)
        self._processes: Dict[int, _FakeWorkerProcess] = {}
        self._lock = threading.Lock()

    def popen(self, cmd: List[str], stdout=None, stderr=None, cwd=None):  # noqa: D401 - mimic subprocess.Popen signature
        with self._lock:
            pid = next(self._pid_iter)
            process = _FakeWorkerProcess(pid, cmd, stdout, self)
            self._processes[pid] = process
        return process

    def unregister(self, pid: int) -> None:
        with self._lock:
            self._processes.pop(pid, None)

    def is_running(self, pid: int) -> bool:
        with self._lock:
            process = self._processes.get(pid)
        return bool(process and process.is_running())

    def active_pids(self) -> List[int]:
        with self._lock:
            return [pid for pid, process in self._processes.items() if process.is_running()]

    def kill(self, pid: int, sig: int) -> None:
        if sig == 0:
            if not self.is_running(pid):
                raise OSError(errno.ESRCH, os.strerror(errno.ESRCH))
            return

        with self._lock:
            process = self._processes.get(pid)

        if process is None:
            raise ProcessLookupError()

        process.send_signal(sig)

    def shutdown_all(self) -> None:
        with self._lock:
            processes = list(self._processes.values())
        for process in processes:
            process.send_signal(signal.SIGTERM)


def _setup_fake_worker_runtime(monkeypatch, workers_module):
    controller = _FakeWorkerController()

    def fake_popen(cmd, stdout=None, stderr=None, cwd=None):
        return controller.popen(cmd, stdout, stderr, cwd)

    def fake_is_process_running(self, pid):
        return controller.is_running(pid)

    monkeypatch.setattr(workers_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(workers_module.os, "kill", controller.kill)
    monkeypatch.setattr(workers_module.WorkerManager, "is_process_running", fake_is_process_running)
    return controller


def test_backup_and_restore_roundtrip(fake_redis, tmp_path, capsys):
    source_config = {"host": "source", "port": 6379, "db": 0}
    target_config = {"host": "target", "port": 6380, "db": 1}

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
    assert backup["metadata"]["type_summary"] == {
        "hash": 1,
        "list": 1,
        "set": 1,
        "string": 2,
        "zset": 1,
    }

    entries_by_key = {
        redis_backup.decode_bytes(entry["key"]): entry for entry in backup["entries"]
    }
    plain_entry = entries_by_key[b"plain"]
    assert plain_entry["value"]["encoding"] == "dump"
    assert redis_backup.decode_bytes(plain_entry["value"]["data"]) == source_client.dump(b"plain")
    assert entries_by_key[b"ephemeral"]["pttl"] == 5000
    hash_entry = entries_by_key[b"hash"]
    assert hash_entry["value"]["encoding"] == "dump"
    assert redis_backup.decode_bytes(hash_entry["value"]["data"]) == source_client.dump(b"hash")

    numbers_entry = entries_by_key[b"numbers"]
    assert numbers_entry["value"]["encoding"] == "dump"
    assert redis_backup.decode_bytes(numbers_entry["value"]["data"]) == source_client.dump(b"numbers")

    letters_entry = entries_by_key[b"letters"]
    assert letters_entry["value"]["encoding"] == "dump"
    assert redis_backup.decode_bytes(letters_entry["value"]["data"]) == source_client.dump(b"letters")

    ranking_entry = entries_by_key[b"ranking"]
    assert ranking_entry["value"]["encoding"] == "dump"
    assert redis_backup.decode_bytes(ranking_entry["value"]["data"]) == source_client.dump(b"ranking")

    backup_path = tmp_path / "backup.json"
    saved_path = redis_backup.save_backup_to_file(backup, backup_path)
    assert saved_path == backup_path

    loaded_backup = redis_backup.load_backup_from_file(backup_path)
    assert loaded_backup == backup

    redis_backup.display_backup_summary(loaded_backup)
    summary_output = capsys.readouterr().out
    assert "Numero di chiavi: 6" in summary_output

    target_client.set(b"plain", b"outdated")
    target_client.sadd(b"extra", b"value")

    restored = redis_backup.restore_redis_backup(loaded_backup, target_config, flush_target=True)
    assert restored == 6
    assert target_client.dbsize() == 6
    assert target_client.get(b"plain") == b"value"
    assert target_client.get(b"ephemeral") == b"temp-value"
    assert target_client.pttl(b"ephemeral") == 5000
    assert target_client.hgetall(b"hash") == {
        b"field1": b"value1",
        b"field2": b"value2",
    }
    assert target_client.lrange(b"numbers", 0, -1) == [b"two", b"one"]
    assert target_client.smembers(b"letters") == {b"a", b"b"}
    assert target_client.zrange(b"ranking", 0, -1, withscores=True) == [
        (b"alice", 1.0),
        (b"bob", 2.0),
    ]
    assert target_client.pttl(b"ephemeral") == 5000
    assert target_client.scan()[1]

    overview = redis_backup.get_database_overview(target_config)
    assert overview == {
        "redis_version": "fake",
        "redis_mode": "standalone",
        "db": 1,
        "key_count": 6,
    }


def test_build_redis_client_requires_host():
    with pytest.raises(ValueError):
        redis_backup.build_redis_client({})


def test_build_redis_client_supports_credentials(fake_redis):
    client = redis_backup.build_redis_client(
        {
            "host": "auth-host",
            "port": 7000,
            "db": 2,
            "username": "demo",
            "password": "secret",
        }
    )
    client.set(b"key", b"value")
    assert fake_redis.last_kwargs is not None
    assert fake_redis.last_kwargs["host"] == "auth-host"
    assert fake_redis.last_kwargs["username"] == "demo"
    assert fake_redis.last_kwargs["password"] == "secret"


def test_build_redis_client_ignores_blank_username(fake_redis):
    redis_backup.build_redis_client(
        {"host": "localhost", "port": 6379, "db": 0, "username": "", "password": None}
    )
    assert fake_redis.last_kwargs is not None
    assert "username" not in fake_redis.last_kwargs
    assert "password" not in fake_redis.last_kwargs


def test_backup_metadata_includes_username(fake_redis):
    config = {
        "host": "source",
        "port": 6379,
        "db": 0,
        "username": "demo",
        "password": "secret",
    }
    client = redis_backup.build_redis_client(config)
    client.set(b"demo", b"payload")
    backup = redis_backup.create_redis_backup(config)
    assert backup["metadata"]["source"]["username"] == "demo"


def test_pipeline_init_worker_backup_restore(fake_redis, tmp_path, monkeypatch, capsys):
    _install_aeon_stub(monkeypatch)
    monkeypatch.chdir(tmp_path)

    for module_name in ["init_aeon_univariate", "enhanced_launch_workers"]:
        if module_name in sys.modules:
            del sys.modules[module_name]

    init_module = importlib.import_module("init_aeon_univariate")
    workers_module = importlib.import_module("enhanced_launch_workers")

    controller = _setup_fake_worker_runtime(monkeypatch, workers_module)

    try:
        dummy_worker_path = Path(__file__).parent / "dummy_worker.py"
        workers_module.DEFAULT_CONFIG = {
            "redis": {"host": "localhost", "port": 6379},
            "workers": {
                "dummy": {
                    "script": str(dummy_worker_path),
                    "count": 1,
                    "args": ["--iterations", "200", "--sleep-seconds", "0.01"],
                }
            },
            "logging": {"directory": str(tmp_path / "logs"), "cleanup_days": 7},
            "profiles": {},
        }

        _run_cli(
            init_module.main,
            [
                "init_aeon_univariate.py",
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
            ],
        )

        capsys.readouterr()

        import redis as redis_module

        data_client = redis_module.Redis(host="localhost", port=6379, db=0)
        assert data_client.get("label") is not None

        def fetch_run_keys() -> set[str]:
            _, keys = data_client.scan(match="worker:run:*")
            decoded = set()
            for key in keys:
                decoded.add(key.decode() if isinstance(key, bytes) else key)
            return decoded

        def wait_for_run_keys(min_count: int, timeout: float = 2.0) -> set[str]:
            deadline = time.time() + timeout
            last_keys: set[str] = set()
            while time.time() < deadline:
                last_keys = fetch_run_keys()
                if len(last_keys) >= min_count:
                    return last_keys
                time.sleep(0.05)
            raise AssertionError("Timed out waiting for worker activity")

        # First run and shutdown ---------------------------------------------
        _run_cli(workers_module.main, ["enhanced_launch_workers.py", "clean-restart"])
        capsys.readouterr()
        assert controller.active_pids()

        initial_run_keys = wait_for_run_keys(3)
        assert data_client.get("worker:last_iteration") is not None

        _run_cli(workers_module.main, ["enhanced_launch_workers.py", "stop"])
        capsys.readouterr()
        assert not controller.active_pids()

        post_stop_run_keys = fetch_run_keys()
        assert post_stop_run_keys.issuperset(initial_run_keys)

        first_backup = redis_backup.create_redis_backup({"host": "localhost", "port": 6379, "db": 0})
        first_backup_path = tmp_path / "backup_cycle1.json"
        redis_backup.save_backup_to_file(first_backup, first_backup_path)
        capsys.readouterr()
        redis_backup.display_backup_summary(first_backup)
        summary_first = capsys.readouterr().out
        assert "Numero di chiavi" in summary_first
        assert first_backup_path.exists()

        data_client.flushdb()
        assert not fetch_run_keys()

        restored_count = redis_backup.restore_redis_backup(
            first_backup, {"host": "localhost", "port": 6379, "db": 0}, flush_target=True
        )
        assert restored_count == first_backup["metadata"]["key_count"]
        assert data_client.get("label") is not None
        assert fetch_run_keys() == post_stop_run_keys

        # Second run after restore ------------------------------------------
        _run_cli(workers_module.main, ["enhanced_launch_workers.py", "clean-restart"])
        capsys.readouterr()
        assert controller.active_pids()

        wait_for_run_keys(len(post_stop_run_keys) + 3)
        _run_cli(workers_module.main, ["enhanced_launch_workers.py", "stop"])
        capsys.readouterr()
        assert not controller.active_pids()

        final_run_keys = fetch_run_keys()
        assert len(final_run_keys) > len(post_stop_run_keys)
        assert data_client.get("worker:last_iteration") is not None

        final_backup = redis_backup.create_redis_backup({"host": "localhost", "port": 6379, "db": 0})
        final_backup_path = tmp_path / "backup_cycle2.json"
        redis_backup.save_backup_to_file(final_backup, final_backup_path)
        capsys.readouterr()
        redis_backup.display_backup_summary(final_backup)
        summary_final = capsys.readouterr().out
        assert "Numero di chiavi" in summary_final
        assert final_backup_path.exists()

        def decode_backup_run_keys(backup):
            result = set()
            for entry in backup["entries"]:
                key_name = redis_backup.decode_bytes(entry["key"]).decode()
                if key_name.startswith("worker:run:"):
                    result.add(key_name)
            return result

        backup_run_keys = decode_backup_run_keys(final_backup)
        assert backup_run_keys == final_run_keys
        assert final_backup["metadata"]["key_count"] >= first_backup["metadata"]["key_count"]
    finally:
        controller.shutdown_all()
