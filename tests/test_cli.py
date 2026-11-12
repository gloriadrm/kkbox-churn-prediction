import pytest

from kkbox_churn.cli import main


def test_cli_missing_input_exits_with_error(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["predict", "--input", str(tmp_path / "no_existe.csv")])
    assert exc.value.code == 1