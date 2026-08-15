"""
Direct download of HuggingFace files using requests.
Retries on transient errors such as connection resets and incomplete reads.
"""
import sys
import time
import requests
from pathlib import Path
from tqdm import tqdm

PROXY = None
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "moltbook_hf"
MAX_RETRIES = 200
RETRY_WAIT = 30  # seconds between retries

FILES = {
    "lnajt": {
        "base_url": "https://huggingface.co/datasets/lnajt/moltbook/resolve/main",
        "dest": DATA_DIR / "lnajt",
        "files": ["posts.parquet", "comments.parquet"],
    },
    "moltnet": {
        "base_url": "https://huggingface.co/datasets/iNLP-Lab/Moltbook-MoltNet/resolve/main",
        "dest": DATA_DIR / "moltnet",
        "files": [
            "data/v2026-02-28/posts.parquet",
            "data/v2026-02-28/comments.parquet",
            "data/v2026-02-28/agents.parquet",
            "data/v2026-02-28/submolts.parquet",
            "data/posts_fully_connected.parquet",
        ],
    },
}


def download_file(url, dest_path, proxies):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"  Already exists: {dest_path} ({dest_path.stat().st_size / 1e6:.1f} MB)")
        return True

    tmp_path = dest_path.with_suffix(".tmp")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  [{attempt}/{MAX_RETRIES}] Downloading {dest_path.name} ...")
            resp = requests.get(url, stream=True, proxies=proxies, timeout=(30, 300))
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))

            with open(tmp_path, "wb") as f:
                with tqdm(total=total, unit="B", unit_scale=True, desc=dest_path.name) as pbar:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                        pbar.update(len(chunk))

            # Verify size
            if total > 0 and tmp_path.stat().st_size < total:
                print(f"  Incomplete ({tmp_path.stat().st_size}/{total}), retrying...")
                tmp_path.unlink()
                time.sleep(RETRY_WAIT)
                continue

            tmp_path.rename(dest_path)
            print(f"  Saved: {dest_path} ({dest_path.stat().st_size / 1e6:.1f} MB)")
            return True
        except Exception as e:
            ename = type(e).__name__
            print(f"  Error (attempt {attempt}): {ename}: {str(e)[:120]}. Retrying in {RETRY_WAIT}s...")
            if tmp_path.exists():
                tmp_path.unlink()
            time.sleep(RETRY_WAIT)

    print(f"  FAILED after {MAX_RETRIES} attempts: {dest_path.name}")
    return False


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "all"
    targets = FILES if dataset == "all" else {dataset: FILES[dataset]}

    failed = []
    for name, cfg in targets.items():
        print(f"\n=== {name} ===")
        for fname in cfg["files"]:
            url = f"{cfg['base_url']}/{fname}"
            dest = cfg["dest"] / fname
            if not download_file(url, dest, PROXY):
                failed.append(fname)

    if failed:
        print(f"\nFailed downloads: {failed}")
        sys.exit(1)
    else:
        print("\nAll downloads complete!")


if __name__ == "__main__":
    main()
