#!/usr/bin/env python3
"""Audit released GFPack++ assets without importing or modifying upstream code.

The JSON contains logical asset labels and ZIP member names, never input paths.
Only tensor weights are loaded by torch (weights_only=True). Optional training
windows use a small, non-executing pickle opcode reader that rejects references
and all object construction; unsupported pickle formats fail explicitly.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, deque
from collections.abc import Mapping
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import pickle
import pickletools
import re
import sys
import zipfile

sys.dont_write_bytecode = True
CHUNK = 1024 * 1024
TEST_MEMBER = "data/test/dental_test.pkl"
TRAIN_MEMBER = "data/train/dental_train.pkl"
WEIGHT_MEMBER = "data/ckpt/dental.pth"


def digest_stream(stream):
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: stream.read(CHUNK), b""):
        digest.update(block)
        size += len(block)
    return {"sha256": digest.hexdigest(), "bytes": size}


def fingerprint(path):
    with path.open("rb") as stream:
        return digest_stream(stream)


def canonical(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")


class BasicUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError("Object construction is forbidden")

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("Persistent references are forbidden")


class Window:
    """Retain at most two windows while counting an entire top-level list."""
    def __init__(self, size):
        self.size = size
        self.first = []
        self.last = deque(maxlen=size)
        self.count = 0

    def extend(self, values):
        for value in values:
            if len(self.first) < self.size:
                self.first.append(value)
            self.last.append(value)
            self.count += 1


def read_window(stream, size):
    """Interpret primitive list pickle opcodes; never invoke a pickle callable.

    MEMOIZE/BINPUT are ignorable only because every GET opcode is rejected.
    This supports the current released independent numeric list records, rather
    than pretending to be a general-purpose pickle implementation.
    """
    stack = []
    mark = object()
    counts = Counter()
    stopped = False
    for op, arg, _ in pickletools.genops(stream):
        name = op.name
        counts[name] += 1
        if name in ("PROTO", "FRAME", "MEMOIZE", "BINPUT", "LONG_BINPUT", "PUT"):
            continue
        if name == "EMPTY_LIST":
            stack.append(Window(size) if not stack else [])
        elif name == "MARK":
            stack.append(mark)
        elif name in ("BININT", "BININT1", "BININT2", "LONG1", "LONG4", "INT", "LONG", "FLOAT", "BINFLOAT"):
            if not isinstance(arg, (int, float)):
                raise ValueError("Expected a numeric primitive")
            stack.append(arg)
        elif name == "APPENDS":
            index = len(stack) - 1
            while index >= 0 and stack[index] is not mark:
                index -= 1
            if index < 1 or not isinstance(stack[index - 1], (list, Window)):
                raise ValueError("Malformed numeric-list APPENDS")
            stack[index - 1].extend(stack[index + 1:])
            del stack[index:]
        elif name == "APPEND":
            if len(stack) < 2 or not isinstance(stack[-2], (list, Window)):
                raise ValueError("Malformed numeric-list APPEND")
            value = stack.pop()
            stack[-1].extend([value])
        elif name == "STOP":
            stopped = True
            break
        else:
            raise ValueError(f"Unsupported pickle opcode: {name}")
    if not stopped or len(stack) != 1 or not isinstance(stack[0], Window):
        raise ValueError("Expected one complete primitive top-level list")
    return stack[0], dict(sorted(counts.items()))


def basic_test_objects(payload):
    stream = io.BytesIO(payload)
    result = []
    while stream.tell() < len(payload):
        result.append(BasicUnpickler(stream).load())
    if len(result) not in (2, 3) or any(not isinstance(item, list) for item in result):
        raise ValueError("Expected test IDs/actions and optional height lists")
    if len({len(item) for item in result}) != 1:
        raise ValueError("Test object lengths disagree")
    return result


def inspect_windows(archive, test_payload, window_size):
    test = basic_test_objects(test_payload)
    if window_size < len(test[0]):
        raise ValueError("Window must contain at least the complete test sample count")
    test_stream = io.BytesIO(test_payload)
    cross_checks = []
    for expected in test:
        observed, _ = read_window(test_stream, max(len(expected), window_size))
        cross_checks.append(observed.first == expected and observed.count == len(expected))
    if not all(cross_checks):
        raise ValueError("Opcode reader disagrees with restricted test unpickler")
    windows = []
    opcodes = []
    full_format_checked = False
    with archive.open(TRAIN_MEMBER) as zipped, io.BufferedReader(zipped, CHUNK) as stream:
        ids, counts = read_window(stream, window_size)
        windows.append(ids)
        opcodes.append(counts)
        # IDs alone disprove exact split equality when both windows differ.
        # Only parse the large action object if an ID window could match.
        possible_first = ids.first[:len(test[0])] == test[0]
        possible_last = list(ids.last)[-len(test[0]):] == test[0]
        if possible_first or possible_last:
            while stream.peek(1):
                value, counts = read_window(stream, window_size)
                windows.append(value)
                opcodes.append(counts)
            if len(windows) not in (2, 3) or len({item.count for item in windows}) != 1:
                raise ValueError("Training object count/lengths are inconsistent")
            if len(test) != len(windows):
                raise ValueError("Training/test object formats differ")
            full_format_checked = True
    labels = ["polygon_ids", "actions", "heights"][:len(windows)]
    prefix = {label: value.first for label, value in zip(labels, windows)}
    suffix = {label: list(value.last) for label, value in zip(labels, windows)}
    first_count = min(len(test[0]), windows[0].count, window_size)
    target = {label: values for label, values in zip(labels, test)}
    first_equal = {label: prefix[label][:first_count] == values[:first_count] for label, values in target.items()}
    last_equal = {label: suffix[label][-first_count:] == values[-first_count:] for label, values in target.items()}
    for label in ["polygon_ids", "actions", "heights"][:len(test)]:
        first_equal.setdefault(label, None)
        last_equal.setdefault(label, None)
    test_id_hashes = {hashlib.sha256(canonical(value)).hexdigest() for value in test[0]}
    return {
        "status": "checked_streaming_primitive_opcodes",
        "reader_cross_checked_against_restricted_test_unpickler": all(cross_checks),
        "window_size": window_size,
        "train_object_count": len(windows) if full_format_checked else None,
        "scanned_train_object_count": len(windows),
        "scanned_fields": labels,
        "train_sample_count": windows[0].count,
        "test_sample_count": len(test[0]),
        "test_equals_train_first_window_by_field": first_equal,
        "test_equals_train_last_window_by_field": last_equal,
        "test_equals_train_first_window": len(test[0]) == first_count and all(value is True for value in first_equal.values()),
        "test_equals_train_last_window": len(test[0]) == first_count and all(value is True for value in last_equal.values()),
        "unscanned_field_reason": None if full_format_checked else "Both ordered ID windows already differ from test; complete record equality is impossible, so actions/heights were not scanned.",
        "test_ordered_id_records_present_in_first_window": sum(hashlib.sha256(canonical(row)).hexdigest() in test_id_hashes for row in prefix["polygon_ids"]),
        "test_ordered_id_records_present_in_last_window": sum(hashlib.sha256(canonical(row)).hexdigest() in test_id_hashes for row in suffix["polygon_ids"]),
        "window_sha256": {"first": hashlib.sha256(canonical(prefix)).hexdigest(), "last": hashlib.sha256(canonical(suffix)).hexdigest(), "test": hashlib.sha256(canonical(target)).hexdigest()},
        "opcode_counts_by_object": opcodes,
        "scope": "Only the first/last window was compared; interior membership, split origin, and paper test identity are not established. Fields marked null were not checked.",
    }


def checkpoint_audit(path):
    import torch
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    wrapper = None
    if isinstance(loaded, Mapping) and "state_dict" in loaded:
        wrapper = sorted(str(key) for key in loaded)
        state = loaded["state_dict"]
    else:
        state = loaded
    if not isinstance(state, Mapping) or not state or any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError("Expected a nonempty tensor-only state_dict")
    shapes = []
    groups = {}
    for key, tensor in sorted(state.items()):
        elements = tensor.numel()
        shapes.append({"key": key, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "elements": elements})
        group = key.split(".")[0]
        item = groups.setdefault(group, {"tensor_count": 0, "elements": 0})
        item["tensor_count"] += 1
        item["elements"] += elements
    def shape(key):
        return list(state[key].shape) if key in state else None
    def layers(prefix):
        indices = sorted({int(match.group(1)) for key in state if (match := re.match(re.escape(prefix) + r"\.(\d+)\.", key))})
        return {"indices": indices, "count": len(indices), "contiguous_from_zero": indices == list(range(len(indices)))}
    metadata = getattr(state, "_metadata", {})
    safe_metadata = {}
    if isinstance(metadata, Mapping):
        for key, value in metadata.items():
            if not isinstance(key, str) or not isinstance(value, Mapping) or set(value) != {"version"} or not isinstance(value["version"], int):
                raise ValueError("Unexpected state_dict metadata; review before exporting")
            safe_metadata[key] = {"version": value["version"]}
    return {
        "load_mode": "torch.load(map_location=cpu, weights_only=True)",
        "audit_torch_version": str(torch.__version__),
        "container_type": type(loaded).__name__,
        "wrapper_keys": wrapper,
        "tensor_count": len(shapes),
        "state_dict_elements": sum(item["elements"] for item in shapes),
        "tensor_storage_bytes": sum(t.numel() * t.element_size() for t in state.values()),
        "group_counts": groups,
        "observed_shapes": {"action_first_weight": shape("action_encoder.0.weight"), "main_output_weight": shape("output_layer.weight"), "main_encoder_feedforward_weight": shape("transformer_model.encoder.layers.0.linear1.weight"), "geometry_feedforward_weight": shape("geo_feature.transformer_encoder.layers.0.linear1.weight"), "fourier_vector": shape("t_embed.0.W")},
        "observed_layer_indices": {"main_encoder": layers("transformer_model.encoder.layers"), "main_decoder": layers("transformer_model.decoder.layers"), "geometry_encoder": layers("geo_feature.transformer_encoder.layers")},
        "boundary_named_tensor_keys": [key for key in state if any(word in key.lower() for word in ("boundary", "container"))],
        "state_metadata": safe_metadata,
        "all_tensor_shapes": shapes,
        "cannot_infer_from_weights": ["training sigma_min/sigma_max or SDE kind", "action translation units or rotation feature scale", "attention head counts", "dropout probability or train/eval inference mode", "Fourier initialization distribution scale", "training epochs, optimizer state, learning rate or EMA use", "loss weighting or training/validation/test membership", "identity with the paper checkpoint or reported architecture", "requires_grad flags or a universal trainable-parameter count"],
        "interpretation": "Names and tensor shapes support compatibility checks. They do not establish the historical training configuration; state_dict elements can include frozen parameters and buffers.",
    }


def source_audit(upstream):
    result = {"files": {}, "source_facts_not_training_provenance": []}
    for name in ("src/score_model.py", "src/gnn_feature.py", "src/sde.py", "src/train.py", "src/evaluate.py", "scripts/train.sh", "README.md"):
        path = upstream / name
        if path.is_file():
            result["files"][name] = fingerprint(path)
    path = upstream / "src/score_model.py"
    if path.is_file():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "PolygonPackingTransformer":
                for function in node.body:
                    if isinstance(function, ast.FunctionDef) and function.name == "__init__":
                        names = [argument.arg for argument in function.args.args]
                        defaults = {key: ast.unparse(value) for key, value in zip(names[-len(function.args.defaults):], function.args.defaults)}
                        result["source_facts_not_training_provenance"].append({"file": "src/score_model.py", "line": function.lineno, "constructor_defaults": defaults})
                    if isinstance(function, ast.FunctionDef) and function.name == "forward":
                        result["source_facts_not_training_provenance"].append({"file": "src/score_model.py", "line": function.lineno, "forward_argument_names": [argument.arg for argument in function.args.args]})
    for name, names in (("src/sde.py", {"init_sde"}), ("src/train.py", {"padData", "collectData", "collectValiData"})):
        path = upstream / name
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for function in tree.body:
            if isinstance(function, ast.FunctionDef) and function.name in names:
                for node in ast.walk(function):
                    if isinstance(node, (ast.Assign, ast.AugAssign)):
                        expression = ast.unparse(node)
                        if any(token in expression for token in ("sigma_min", "sigma_max", "*= 500", "dirName")):
                            result["source_facts_not_training_provenance"].append({"file": name, "function": function.name, "line": node.lineno, "statement": expression})
    candidate_suffixes = {".pth", ".pt", ".ckpt", ".json", ".yaml", ".yml", ".log"}
    result["upstream_candidate_artifact_files"] = sorted(path.relative_to(upstream).as_posix() for path in upstream.rglob("*") if path.is_file() and ".git" not in path.relative_to(upstream).parts and (path.suffix.lower() in candidate_suffixes or "tfevents" in path.name))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--train-windows", action="store_true")
    parser.add_argument("--window-size", default=128, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        raise ValueError("Output must be new and its parent directory must exist")
    if not 1 <= args.window_size <= 1024:
        raise ValueError("window-size must be between 1 and 1024")
    inputs = {"checkpoint": args.checkpoint, "archive": args.archive, "used_dataset": args.dataset}
    if args.provenance:
        inputs["asset_provenance"] = args.provenance
    before = {label: fingerprint(path) for label, path in inputs.items()}
    report = {"schema_version": 1, "status": "audit_complete_not_reproduction_validation", "inputs": before, "checkpoint": checkpoint_audit(args.checkpoint)}
    with zipfile.ZipFile(args.archive) as archive:
        infos = archive.infolist()
        if len({item.filename for item in infos}) != len(infos):
            raise ValueError("Duplicate ZIP names")
        members = []
        for info in infos:
            relative = PurePosixPath(info.filename)
            if relative.is_absolute() or ".." in relative.parts or ":" in info.filename or "\\" in info.filename:
                raise ValueError("Non-relative ZIP member name")
            item = {"member": info.filename, "bytes": info.file_size, "directory": info.is_dir(), "crc32": f"{info.CRC:08x}"}
            if not info.is_dir():
                with archive.open(info) as stream:
                    item.update(digest_stream(stream))
            members.append(item)
        member_map = {item["member"]: item for item in members}
        test_payload = archive.read(TEST_MEMBER)
        test = basic_test_objects(test_payload)
        report["archive"] = {"entry_count": len(members), "uncompressed_bytes": sum(item["bytes"] for item in members), "all_file_reads_crc_checked": True, "all_members": members, "checkpoint_members": [item["member"] for item in members if not item["directory"] and Path(item["member"]).suffix.lower() in {".pt", ".pth", ".ckpt"}], "config_log_members": [item["member"] for item in members if not item["directory"] and (Path(item["member"]).suffix.lower() in {".json", ".yaml", ".yml", ".log", ".py", ".sh"} or "tfevents" in item["member"])], "dataset_members": [item["member"] for item in members if item["member"].endswith(".pkl")]}
        report["asset_identity"] = {"checkpoint_archive_member": WEIGHT_MEMBER, "checkpoint_bytes_equal_archive_member": member_map[WEIGHT_MEMBER]["sha256"] == before["checkpoint"]["sha256"], "used_dataset_archive_member": TEST_MEMBER, "used_dataset_bytes_equal_archive_test": member_map[TEST_MEMBER]["sha256"] == before["used_dataset"]["sha256"], "dataset_role_proved": "Published ZIP test member, regardless of its local validation-like filename", "dataset_role_not_proved": "Paper test split identity, or derivation from the last 128 training samples", "test_object_count": len(test), "test_sample_count": len(test[0]), "test_piece_counts": sorted({len(row) for row in test[0]}), "test_height_values": sorted(set(test[2])) if len(test) == 3 else None}
        if args.train_windows:
            report["training_windows"] = inspect_windows(archive, test_payload, args.window_size)
    if args.provenance:
        value = json.loads(args.provenance.read_text(encoding="utf-8"))
        files = value.get("files", [])
        report["provenance"] = {"source_url": value.get("source_url"), "recorded_archive_sha256_matches": value.get("archive", {}).get("sha256") == before["archive"]["sha256"], "recorded_checkpoint_member": next((item.get("original_archive_member") for item in files if item.get("sha256") == before["checkpoint"]["sha256"]), None), "recorded_dataset_member": next((item.get("original_archive_member") for item in files if item.get("sha256") == before["used_dataset"]["sha256"]), None), "original_training_config_provided": False, "scope": "Local preparation manifest records retrieval/extraction; it does not establish checkpoint training provenance."}
    if args.upstream:
        report["source"] = source_audit(args.upstream)
    after = {label: fingerprint(path) for label, path in inputs.items()}
    report["input_files_unchanged"] = before == after
    if before != after:
        raise ValueError("Input changed during read-only audit")
    if not report["asset_identity"]["checkpoint_bytes_equal_archive_member"] or not report["asset_identity"]["used_dataset_bytes_equal_archive_test"]:
        raise ValueError("Used assets differ from published ZIP members")
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "state_dict_elements": report["checkpoint"]["state_dict_elements"], "tensor_count": report["checkpoint"]["tensor_count"], "asset_identity": report["asset_identity"], "training_windows": report.get("training_windows"), "input_files_unchanged": report["input_files_unchanged"]}, indent=2))


if __name__ == "__main__":
    main()
