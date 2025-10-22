"""Tests for analysing redis backup summary output."""
from __future__ import annotations

import json

import pytest

import redis_backup


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
