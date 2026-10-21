#!/usr/bin/env python3
"""Prepare the official GFPack++ Dental inference assets without changing upstream."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import os
import pickle
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
RESOURCES = ROOT / "resources"
PROVENANCE = ROOT / "asset_provenance.json"
SOURCE_URL = "https://drive.google.com/file/d/1Xq-pptfYVsve2wsMJxSgMdyLALxD4AXW/view?usp=sharing"
DOWNLOAD_URL = "https://drive.usercontent.google.com/download?id=1Xq-pptfYVsve2wsMJxSgMdyLALxD4AXW&export=download&confirm=t"
EXPECTED_ARCHIVE_BYTES = 358228192
ARCHIVE = RESOURCES / "_downloads" / "gfpp.zip"
ARCHIVE_SOURCE = RESOURCES / "_downloads" / "gfpp.source.json"
CHUNK = 1024 * 1024


def timestamp() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_parent(path: Path) -> None:
    """All writes stay in this experiment, and never follow a symlink."""
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Destination escapes experiment: {path}")
    for parent in (ROOT, *reversed(path.parent.relative_to(ROOT).parents)):
        candidate = parent if parent == ROOT else ROOT / parent
        if candidate.is_symlink():
            raise ValueError(f"Refusing symlink directory: {candidate}")
    current = ROOT
    for component in path.parent.relative_to(ROOT).parts:
        current /= component
        if current.is_symlink():
            raise ValueError(f"Refusing symlink directory: {current}")
        current.mkdir(exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"Refusing symlink destination: {path}")


def install_without_overwrite(temporary: Path, destination: Path) -> str:
    """Publish a completed file exclusively; identical existing files are reusable."""
    safe_parent(destination)
    digest = sha256(temporary)
    if destination.exists():
        if not destination.is_file() or sha256(destination) != digest:
            raise FileExistsError(f"Existing file has different contents: {destination}")
        temporary.unlink()
        return "reused_identical"
    # Hard-link publication is atomic and cannot overwrite a concurrent writer.
    try:
        os.link(temporary, destination)
    except FileExistsError:
        if not destination.is_file() or sha256(destination) != digest:
            raise FileExistsError(f"Concurrent file has different contents: {destination}")
        temporary.unlink()
        return "reused_identical"
    temporary.unlink()
    return "created"


def download_archive() -> dict:
    safe_parent(ARCHIVE)
    if ARCHIVE.exists():
        if not ARCHIVE.is_file() or ARCHIVE.stat().st_size != EXPECTED_ARCHIVE_BYTES:
            raise FileExistsError(f"Unexpected existing archive; refusing overwrite: {ARCHIVE}")
        previous_source = PROVENANCE if PROVENANCE.exists() else ARCHIVE_SOURCE
        if not previous_source.exists():
            raise FileExistsError("Existing archive lacks provenance; refusing to assume its source")
        previous = json.loads(previous_source.read_text(encoding="utf-8"))
        recorded = previous.get("archive", {})
        digest = sha256(ARCHIVE)
        if recorded.get("sha256") != digest or previous.get("source_url") != SOURCE_URL:
            raise FileExistsError("Existing archive differs from its recorded official source")
        return {**recorded, "status": "reused_identical"}

    print(f"Downloading official gfpp.zip ({EXPECTED_ARCHIVE_BYTES:,} bytes)...", flush=True)
    request = urllib.request.Request(DOWNLOAD_URL, headers={"User-Agent": "GFPackPP-asset-preparation/1.0"})
    temporary = None
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            media_type = response.headers.get_content_type()
            if response.status != 200 or media_type not in ("application/octet-stream", "application/zip"):
                raise RuntimeError(f"Expected archive response; received HTTP {response.status}, {media_type}")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != EXPECTED_ARCHIVE_BYTES:
                raise RuntimeError(f"Official archive size changed: {length}")
            metadata = {
                "original_name": "gfpp.zip",
                "download_url": DOWNLOAD_URL,
                "resolved_url": response.url,
                "downloaded_at_beijing": timestamp(),
                "http_last_modified": response.headers.get("Last-Modified"),
                "http_content_type": media_type,
            }
            digest = hashlib.sha256()
            written = 0
            with tempfile.NamedTemporaryFile(prefix="gfpp-", suffix=".part", dir=ARCHIVE.parent, delete=False) as stream:
                temporary = Path(stream.name)
                while block := response.read(CHUNK):
                    written += len(block)
                    if written > EXPECTED_ARCHIVE_BYTES:
                        raise RuntimeError("Archive exceeds expected size; download aborted")
                    stream.write(block)
                    digest.update(block)
            if written != EXPECTED_ARCHIVE_BYTES:
                raise RuntimeError(f"Incomplete download: {written}/{EXPECTED_ARCHIVE_BYTES} bytes")
            metadata.update(bytes=written, sha256=digest.hexdigest(), path=str(ARCHIVE.relative_to(ROOT)))
        metadata["status"] = install_without_overwrite(temporary, ARCHIVE)
        temporary = None
        with ARCHIVE_SOURCE.open("x", encoding="utf-8") as source_file:
            json.dump({"source_url": SOURCE_URL, "archive": metadata}, source_file, indent=2)
        return metadata
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def selected_members() -> dict[str, Path]:
    selected = {
        "data/ckpt/dental.pth": RESOURCES / "checkpoints" / "score_model.pth",
        "data/test/dental_test.pkl": RESOURCES / "data" / "datasets" / "dataset_dental_vali_128.pkl",
    }
    selected.update({f"data/polygons/dental/{index}.txt": RESOURCES / "data" / "polygons" / f"{index}.txt" for index in range(440)})
    return selected


def prepare_members() -> tuple[list[dict], dict]:
    selected = selected_members()
    result = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        infos = archive.infolist()
        names = Counter(info.filename for info in infos)
        if any(names[name] != 1 for name in selected):
            raise RuntimeError("Required ZIP member missing or duplicated")
        if len(infos) > 10000 or sum(info.file_size for info in infos) > 1024 ** 3:
            raise RuntimeError("Archive exceeds the expected safety limits")
        if any(info.flag_bits & 1 for info in infos):
            raise RuntimeError("Encrypted ZIP members are not supported")
        print(f"Checking CRC for all {len(infos)} ZIP entries...", flush=True)
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP CRC failed: {bad_member}")
        crc = {"status": "passed", "checked_at_beijing": timestamp(), "entry_count": len(infos), "uncompressed_bytes": sum(info.file_size for info in infos)}
        for name, destination in selected.items():
            info = archive.getinfo(name)
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Refusing non-file archive member: {name}")
            safe_parent(destination)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(prefix="asset-", suffix=".part", dir=destination.parent, delete=False) as output:
                    temporary = Path(output.name)
                    with archive.open(info) as source:
                        shutil.copyfileobj(source, output, CHUNK)
                digest = sha256(temporary)
                status = install_without_overwrite(temporary, destination)
                temporary = None
                result.append({"original_archive_member": name, "path": str(destination.relative_to(ROOT)), "bytes": info.file_size, "zip_crc32": f"{info.CRC:08x}", "sha256": digest, "status": status})
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
    return result, crc


def inspect_dataset() -> dict:
    """Read only the official-source test pickle, on CPU, without installing packages."""
    dataset = RESOURCES / "data" / "datasets" / "dataset_dental_vali_128.pkl"
    try:
        with dataset.open("rb") as stream:
            polygon_ids = pickle.load(stream)
            actions = pickle.load(stream)
            try:
                heights = pickle.load(stream)
            except EOFError:
                heights = None
            if stream.read(1):
                raise ValueError("Unexpected objects after optional height list")
    except ModuleNotFoundError as error:
        return {"status": "not_checked_missing_existing_dependency", "reason": str(error)}
    if len(polygon_ids) != len(actions) or len(polygon_ids) < 128:
        raise ValueError("Test IDs/actions do not contain the expected 128 paired samples")
    if heights is not None and (len(heights) != len(polygon_ids) or
            not all(isinstance(height, numbers.Real) and math.isfinite(float(height))
                    and height > 0 for height in heights)):
        raise ValueError("Optional height list must contain a finite positive height per sample")
    all_ids = []
    polygon_counts = []
    for sample_ids, sample_actions in zip(polygon_ids, actions):
        if len(sample_ids) != len(sample_actions):
            raise ValueError("Polygon/action counts differ")
        polygon_counts.append(len(sample_ids))
        for polygon_id, action in zip(sample_ids, sample_actions):
            if not isinstance(polygon_id, numbers.Integral) or not 0 <= int(polygon_id) < 440:
                raise ValueError(f"Polygon ID outside 0..439: {polygon_id}")
            all_ids.append(int(polygon_id))
            if len(action) != 4 or not all(isinstance(value, numbers.Real) and math.isfinite(float(value)) for value in action):
                raise ValueError("Expected four finite action values per polygon")
    return {"status": "passed", "checked_at_beijing": timestamp(), "sample_count": len(polygon_ids), "polygon_id_type": type(polygon_ids).__name__, "actions_type": type(actions).__name__, "polygon_count_min": min(polygon_counts), "polygon_count_max": max(polygon_counts), "polygon_id_min": min(all_ids), "polygon_id_max": max(all_ids), "action_values_per_polygon": 4, "actions_all_finite": True, "optional_height_count": len(heights) if heights is not None else 0, "height_range": [min(heights), max(heights)] if heights is not None else None, "checkpoint_compatibility": "pending_server_environment_validation"}


def write_provenance(value: dict) -> None:
    if PROVENANCE.exists():
        existing = json.loads(PROVENANCE.read_text(encoding="utf-8"))
        if existing.get("source_url") != SOURCE_URL or existing.get("archive", {}).get("sha256") != value["archive"]["sha256"]:
            raise FileExistsError("Refusing to overwrite different asset provenance")
        if existing.get("files"):
            expected = {item["path"]: item["sha256"] for item in existing["files"]}
            actual = {item["path"]: item["sha256"] for item in value["files"]}
            if expected != actual:
                raise FileExistsError("Refusing to overwrite provenance for different extracted assets")
        # Preserve the original preparation/download timestamps on repeat runs.
        return
    safe_parent(PROVENANCE)
    with PROVENANCE.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    archive = download_archive()
    files, crc = prepare_members()
    inspection = inspect_dataset()
    manifest = {
        "schema_version": 1,
        "prepared_at_beijing": timestamp(),
        "source_url": SOURCE_URL,
        "archive": archive,
        "zip_crc_validation": crc,
        "selection": {"checkpoint_files": 1, "dataset_files": 1, "polygon_files": 440, "training_dataset_extracted": False},
        "dataset_inspection": inspection,
        "files": files,
    }
    write_provenance(manifest)
    print(json.dumps({"archive_sha256": archive["sha256"], "archive_bytes": archive["bytes"], "zip_crc": crc, "prepared_file_count": len(files), "dataset": inspection, "checkpoint_sha256": files[0]["sha256"], "dataset_sha256": files[1]["sha256"], "provenance": str(PROVENANCE)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
