import os
import json
import tempfile
import shutil
import builtins
from io import StringIO
from chunkanon.utils import (
    log, log_to_file, ensure_folder,
    save_dict_to_json, get_progress_iter, format_time
)

def test_log(capsys):
    log("Hello")
    captured = capsys.readouterr()
    assert "[LOG] Hello" in captured.out

def test_log_to_file():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filepath = tmp.name
    try:
        log_to_file("Test message", filepath)
        with open(filepath, "r") as f:
            content = f.read()
            assert "Test message" in content
    finally:
        os.remove(filepath)

def test_ensure_folder():
    test_dir = tempfile.mkdtemp()
    test_subdir = os.path.join(test_dir, "subfolder")
    ensure_folder(test_subdir)
    assert os.path.exists(test_subdir)
    shutil.rmtree(test_dir)

def test_save_dict_to_json():
    data = {"a": 1, "b": 2}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        filepath = tmp.name
    try:
        save_dict_to_json(data, filepath)
        with open(filepath, "r") as f:
            loaded = json.load(f)
            assert loaded == data
    finally:
        os.remove(filepath)

def test_get_progress_iter():
    items = list(get_progress_iter(range(3), desc="Testing"))
    assert items == [0, 1, 2]

def test_format_time():
    assert format_time(3665) == (1, 1, 5)
    assert format_time(0) == (0, 0, 0)
    assert format_time(59) == (0, 0, 59)
