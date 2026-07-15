"""Generate .pyi type stubs from the compiled nanobind extension.

This script is run by meson.build. It directly imports the built .so file
and generates a stub file using nanobind's StubGen.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import ModuleType

from nanobind.stubgen import StubGen


@contextmanager
def _windows_dll_dirs(module_path: Path):
    """Ensure Windows can resolve dependent DLLs for the extension import."""
    if os.name != "nt":
        yield
        return
    stack = ExitStack()
    try:
        parent = module_path.parent.resolve()
        candidates = [
            parent,
            Path(sys.base_prefix, "DLLs").resolve(),
            Path(sys.base_prefix).resolve(),
        ]
        seen: set[str] = set()
        abs_dirs: list[Path] = []
        for d in candidates:
            try:
                d = d.resolve()
            except Exception:
                continue
            if not d.exists() or not d.is_dir():
                continue
            if str(d) in seen:
                continue
            seen.add(str(d))
            abs_dirs.append(d)

        for d in abs_dirs:
            stack.enter_context(os.add_dll_directory(str(d)))
        yield
    finally:
        stack.close()


def load_module_from_filepath(name: str, filepath: str) -> ModuleType:
    """Load a compiled extension module directly from a file path."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module '{name}' from '{filepath}'")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def build_stub(module_path: Path, output_path: str) -> None:
    """Generate a .pyi stub file from a compiled extension module."""
    module_path = module_path.resolve()
    module_name = module_path.stem.split(".")[0]

    with _windows_dll_dirs(module_path):
        module = load_module_from_filepath(module_name, str(module_path))

    s = StubGen(module, include_docstrings=True, include_private=False)
    s.put(module)
    stub_txt = s.get()

    dest = Path(output_path)
    dest.write_text(stub_txt)

    ruff = Path(sys.executable).parent / "ruff"
    _ruff = str(ruff) if ruff.exists() else "ruff"
    subprocess.run([_ruff, "format", output_path], check=True)
    subprocess.run(
        [
            _ruff,
            "check",
            "--fix-only",
            "--unsafe-fixes",
            "--ignore=D",
            output_path,
        ]
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        module_path = Path(sys.argv[1])
    else:
        build_dir = Path(__file__).parent.parent / "builddir"
        module_path = next(
            x for x in build_dir.glob("_ihist.*") if x.is_file()
        )

    if len(sys.argv) > 2:
        output = Path(sys.argv[2])
    else:
        source_dir = Path(__file__).parent.parent / "python" / "src"
        output = source_dir / "ihist" / "_ihist.pyi"

    build_stub(module_path, str(output))
