#!/usr/bin/env python3
# Estimated runtime from last full canonical run (2026-03-09 15:52 UTC; full pipeline 32m18s): 1.7s.
from pathlib import Path

wrapper_path = Path(__file__).resolve()
archive_path = wrapper_path.parent / "_archive" / wrapper_path.name
namespace = {
    "__file__": str(wrapper_path),
    "__name__": "__main__",
    "__package__": None,
}
exec(compile(archive_path.read_text(), str(archive_path), "exec"), namespace)
