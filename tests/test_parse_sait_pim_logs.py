from pathlib import Path

import importlib.util


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "parse_sait_pim_logs.py"
spec = importlib.util.spec_from_file_location("parse_sait_pim_logs", SCRIPT_PATH)
parser_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(parser_module)


def test_parse_sait_pim_log_extracts_cycles_and_speedup(tmp_path: Path) -> None:
    log_path = tmp_path / "pimbench_add.log"
    log_path.write_text(
        """
  ADD (PIM disabled)
> Cycle : 6651
  ADD (PIM enabled)
> Cycle : 3349
> Speed-up : 1.98597
""",
        encoding="utf-8",
    )

    parsed = parser_module.parse_log(log_path)

    assert parsed["baseline_cycles"] == 6651
    assert parsed["pim_cycles"] == 3349
    assert parsed["speedup"] == 1.98597
