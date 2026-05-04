from pathlib import Path

import pytest

from src.parser import load_profile_csv


def write_csv(path: Path, rows: list[str]) -> None:
    header = "kernel_name,runtime_ms,flops,dram_read_bytes,dram_write_bytes,memory_access_pattern,notes"
    path.write_text("\n".join([header, *rows]), encoding="utf-8")


def test_load_profile_csv_normalizes_access_pattern(tmp_path: Path) -> None:
    csv_path = tmp_path / "profile.csv"
    write_csv(csv_path, ["k0,1.0,10,20,30, Irregular ,test"])

    profile = load_profile_csv(csv_path)

    assert profile.loc[0, "memory_access_pattern"] == "irregular"


def test_load_profile_csv_rejects_negative_bytes(tmp_path: Path) -> None:
    csv_path = tmp_path / "profile.csv"
    write_csv(csv_path, ["k0,1.0,10,-20,30,regular,test"])

    with pytest.raises(ValueError, match="DRAM byte columns"):
        load_profile_csv(csv_path)


def test_load_profile_csv_rejects_empty_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "profile.csv"
    csv_path.write_text(
        "kernel_name,runtime_ms,flops,dram_read_bytes,dram_write_bytes,memory_access_pattern,notes\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="empty"):
        load_profile_csv(csv_path)
