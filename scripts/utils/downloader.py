"""
TEP-JWST shared download utility.

Features:
- Skip if file already exists and size >= min_size_mb (never re-download)
- Resume interrupted downloads (HTTP Range: bytes=N-)
- Parallel chunk downloads for servers supporting Accept-Ranges
- Single-threaded streaming fallback for servers without Range support
- Atomic write via .tmp → rename
"""

import urllib.request
import urllib.error
import threading
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple


def _build_opener(auth: Optional[Tuple[str, str]], url: str):
    """Build urllib opener with optional Basic Auth handler."""
    if not auth:
        return urllib.request.build_opener()
    user, password = auth
    pm = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    pm.add_password(None, url, user, password)
    return urllib.request.build_opener(urllib.request.HTTPBasicAuthHandler(pm))


def _head(url: str, timeout: int = 15, auth=None) -> dict:
    """Return Content-Length and Accept-Ranges from HTTP HEAD."""
    req = urllib.request.Request(url, method="HEAD")
    opener = _build_opener(auth, url)
    try:
        with opener.open(req, timeout=timeout) as resp:
            headers = resp.headers
            return {
                "content_length": int(headers.get("Content-Length", 0) or 0),
                "accept_ranges": (headers.get("Accept-Ranges", "none") or "none").lower() == "bytes",
            }
    except Exception:
        return {"content_length": 0, "accept_ranges": False}


def _download_chunk(url: str, start: int, end: int, dest: Path, timeout: int = 120, auth=None) -> int:
    """Download bytes [start, end] from url into dest file. Returns bytes written."""
    req = urllib.request.Request(url)
    req.add_header("Range", f"bytes={start}-{end}")
    opener = _build_opener(auth, url)
    with opener.open(req, timeout=timeout) as resp:
        data = resp.read()
    with open(dest, "wb") as fh:
        fh.write(data)
    return len(data)


def _stream_download(url: str, dest: Path, resume_from: int = 0,
                     total_size: int = 0, logger=None, timeout: int = 300, auth=None) -> int:
    """Single-threaded streaming download with optional resume."""
    req = urllib.request.Request(url)
    if resume_from > 0:
        req.add_header("Range", f"bytes={resume_from}-")
    opener = _build_opener(auth, url)

    downloaded = resume_from
    mode = "ab" if resume_from > 0 else "wb"

    with opener.open(req, timeout=timeout) as resp, open(dest, mode) as fh:
        while True:
            chunk = resp.read(1 << 20)  # 1 MB
            if not chunk:
                break
            fh.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                print(f"\r  {pct}% ({downloaded/1e6:.1f}/{total_size/1e6:.1f} MB)", end="", flush=True)
    print()
    return downloaded


def smart_download(
    url: str,
    dest: Path,
    min_size_mb: float,
    fallback_urls: list = None,
    n_workers: int = 8,
    logger=None,
    timeout: int = 300,
    auth: Optional[Tuple[str, str]] = None,
) -> bool:
    """
    Download *url* to *dest*, skipping if already complete.

    Parameters
    ----------
    url : str
        Primary download URL.
    dest : Path
        Final destination path.
    min_size_mb : float
        Minimum acceptable file size in MB. Files larger than this are
        considered complete and will not be re-downloaded.
    fallback_urls : list[str], optional
        Tried in order if the primary URL fails.
    n_workers : int
        Number of parallel workers for chunked downloads (default 8).
    logger : optional
        Logger instance; falls back to print if None.
    timeout : int
        Per-connection timeout in seconds.

    Returns
    -------
    bool
        True if file is available and >= min_size_mb.
    """
    def _log(msg, level="INFO"):
        if logger is not None:
            getattr(logger, level.lower(), logger.info)(msg)
        else:
            print(f"[{level}] {msg}")

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    min_size_bytes = int(min_size_mb * 1e6)

    # --- Already complete? ---
    if dest.exists():
        size = dest.stat().st_size
        if size >= min_size_bytes:
            _log(f"Already present: {dest.name} ({size/1e6:.1f} MB) — skipping download.")
            return True
        _log(f"Existing file incomplete ({size/1e6:.1f} MB < {min_size_mb} MB) — re-downloading.")
        dest.unlink()

    all_urls = [url] + (fallback_urls or [])

    for attempt_url in all_urls:
        _log(f"Downloading: {attempt_url}")
        tmp = dest.with_suffix(dest.suffix + ".tmp")

        try:
            meta = _head(attempt_url, timeout=15, auth=auth)
            total = meta["content_length"]
            supports_ranges = meta["accept_ranges"] and total > 0

            # --- Resume check on .tmp file ---
            resume_from = 0
            if tmp.exists():
                partial = tmp.stat().st_size
                if supports_ranges and 0 < partial < total:
                    _log(f"  Resuming from {partial/1e6:.1f} MB …")
                    resume_from = partial
                else:
                    tmp.unlink()

            if supports_ranges and n_workers > 1 and total > 10 * 1024 * 1024:
                # --- Parallel chunked download ---
                chunk_size = max(1 * 1024 * 1024, total // n_workers)
                ranges = []
                pos = resume_from
                while pos < total:
                    end = min(pos + chunk_size - 1, total - 1)
                    ranges.append((pos, end))
                    pos = end + 1

                chunk_dir = tmp.parent / (tmp.name + "_chunks")
                chunk_dir.mkdir(exist_ok=True)
                chunk_paths = [chunk_dir / f"chunk_{i:04d}" for i in range(len(ranges))]

                _log(f"  {total/1e6:.1f} MB — parallel download ({len(ranges)} chunks × {n_workers} workers)")

                lock = threading.Lock()
                done = [0]

                def _task(args):
                    i, (start, end) = args
                    n = _download_chunk(attempt_url, start, end, chunk_paths[i], timeout=timeout, auth=auth)
                    with lock:
                        done[0] += n
                        pct = min(100, done[0] * 100 // total)
                        print(f"\r  {pct}% ({done[0]/1e6:.1f}/{total/1e6:.1f} MB)", end="", flush=True)
                    return n

                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    futures = list(pool.map(_task, enumerate(ranges)))
                print()

                # Assemble chunks
                with open(tmp, "ab" if resume_from else "wb") as out:
                    for cp in chunk_paths:
                        with open(cp, "rb") as inp:
                            shutil.copyfileobj(inp, out)
                        cp.unlink()
                chunk_dir.rmdir()

            else:
                # --- Single-threaded streaming ---
                if total > 0:
                    _log(f"  {total/1e6:.1f} MB — streaming download …")
                else:
                    _log("  Size unknown — streaming download …")
                _stream_download(attempt_url, tmp, resume_from=resume_from,
                                 total_size=total, logger=logger, timeout=timeout, auth=auth)

            # Verify and rename
            size = tmp.stat().st_size
            if size >= min_size_bytes:
                tmp.rename(dest)
                _log(f"  Download complete: {size/1e6:.1f} MB → {dest.name}")
                return True
            else:
                _log(f"  Downloaded file too small ({size/1e6:.1f} MB) — trying next URL.", "WARNING")
                tmp.unlink(missing_ok=True)

        except Exception as exc:
            _log(f"  Failed ({exc}) — trying next URL.", "WARNING")
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            chunk_dir = tmp.parent / (tmp.name + "_chunks")
            if chunk_dir.exists():
                shutil.rmtree(chunk_dir, ignore_errors=True)

    _log(f"ERROR: All download URLs failed. Save manually to: {dest}", "ERROR")
    return False
