from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import upload


def test_local_files_skip_images_and_sdk_caches(tmp_path):
    data = ("README.md", "data/chunk-000/file-000.parquet")
    for rel in (*data, "images/a.png", ".ms_upload_cache/x", ".cache/huggingface/y"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")
    assert sorted(upload.local_files(tmp_path)) == list(data)
