#!/usr/bin/env python3
"""Expose the pinned, inactive rm_overlap in a separate CPU-only extension.

The copied algorithm is unchanged. Instrumentation only records iterations and
which return path was taken; every source change is saved in instrumentation.patch.
"""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import time


COMMIT = "ff75815e24ec50cd7fa8b2abdec1acfe0b55be95"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Pinned upstream checkout")
    parser.add_argument("--output", type=Path, required=True, help="New isolated build directory")
    args = parser.parse_args()
    source = args.source.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    if commit != COMMIT or dirty:
        raise ValueError("Expected the clean pinned upstream checkout")
    args.output.mkdir(parents=True, exist_ok=False)
    build = args.output / "source"
    build.mkdir()
    original_dir = source / "external/rmspacing_cpp"
    sources = {p.name: sha256(p) for p in sorted(original_dir.iterdir()) if p.is_file()}
    for name in sources:
        shutil.copyfile(original_dir / name, build / name)
    original = (build / "main.cpp").read_text()
    start, end = original.index("void rm_overlap("), original.index("\n/*", original.index("void rm_overlap("))
    function = original[start:end]
    loop = "for(int ep=0;ep<2000;ep++){"
    if function.count(loop) != 1 or function.count("return ;") != 1:
        raise ValueError("Unexpected pinned rm_overlap source")
    function = function.replace(loop, loop + "\n        probe_iterations = ep + 1;")
    function = function.replace("return ;", "probe_early_return = true;\n            return ;")
    patched = (original[:start] + "thread_local int probe_iterations = 0;\n"
               "thread_local bool probe_early_return = false;\n" + function + original[end:])
    patched = patched.replace("PYBIND11_MODULE(rmspacing, m)", "PYBIND11_MODULE(collision_probe, m)")
    needle = '    m.def("rm_spacing_all", &rm_spacing_all, "A function that removes spacing");'
    if patched.count(needle) != 1:
        raise ValueError("Unexpected module binding")
    binding = r'''
    m.def("overlap_probe", [](vector<int> ids, vector<double> rotates,
             vector<pair<double,double>> actions,
             vector<vector<pair<double,double>>> polys, double width, double height, double eps) {
        if (ids.empty() || ids.size() != rotates.size() || ids.size() != actions.size()
            || !std::isfinite(width) || !std::isfinite(height) || width <= 0 || height <= 0
            || !std::isfinite(eps) || eps < 0)
            throw std::invalid_argument("Invalid probe dimensions");
        for (size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] < 0 || (size_t)ids[i] >= polys.size() || polys[ids[i]].size() < 3
                || !std::isfinite(rotates[i]) || !std::isfinite(actions[i].first)
                || !std::isfinite(actions[i].second))
                throw std::invalid_argument("Invalid probe input");
            for (auto p : polys[ids[i]])
                if (!std::isfinite(p.first) || !std::isfinite(p.second))
                    throw std::invalid_argument("Nonfinite vertex");
        }
        probe_iterations = 0;
        probe_early_return = false;
        rm_overlap(ids, rotates, actions, polys, width, height, eps);
        py::dict result;
        result["actions_xy"] = actions;
        result["iterations"] = probe_iterations;
        result["return_path"] = probe_early_return ? "no_collision_before_boundary_update" : "iteration_limit";
        return result;
    });
'''
    patched = patched.replace(needle, needle + binding)
    (build / "main.cpp").write_text(patched)
    (args.output / "instrumentation.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(True), patched.splitlines(True), fromfile="pinned/main.cpp", tofile="probe/main.cpp")))
    includes = subprocess.check_output([sys.executable, "-m", "pybind11", "--includes"], text=True).split()
    extension = args.output / ("collision_probe" + sysconfig.get_config_var("EXT_SUFFIX"))
    command = [os.environ.get("CXX", "g++"), "-std=c++17", "-O3", "-fPIC", "-fopenmp", *includes,
               "main.cpp", "clipper.engine.cpp", "clipper.offset.cpp", "clipper.rectclip.cpp",
               "-shared", "-o", str(extension.resolve())]
    started = time.perf_counter()
    with (args.output / "build.log").open("w") as stream:
        result = subprocess.run(command, cwd=build, stdout=stream, stderr=subprocess.STDOUT, timeout=120)
    report = {"upstream_commit": commit, "source_sha256": sources,
              "instrumented_main_sha256": sha256(build / "main.cpp"),
              "patch_sha256": sha256(args.output / "instrumentation.patch"),
              "algorithm_modified": False, "iteration_limit": 2000,
              "instrumentation": "thread-local iteration count, return-path flag, isolated module binding",
              "compiler": subprocess.check_output([command[0], "--version"], text=True).splitlines()[0],
              "compile_flags": ["-std=c++17", "-O3", "-fPIC", "-fopenmp"],
              "python": sys.version, "elapsed_seconds": time.perf_counter() - started,
              "returncode": result.returncode}
    if result.returncode == 0:
        report["extension_sha256"] = sha256(extension)
    report["source_unchanged"] = all(sha256(original_dir / name) == digest for name, digest in sources.items())
    (args.output / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    if result.returncode or not report["source_unchanged"]:
        raise RuntimeError("Probe build failed; inspect the isolated build log")
    print(json.dumps({k: v for k, v in report.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
