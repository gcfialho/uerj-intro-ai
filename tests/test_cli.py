import json

from intro_ai.cli import main


def test_search_and_csp_write_results(tmp_path, capsys):
    args = ["--results-dir", str(tmp_path / "r"), "--img-dir", str(tmp_path / "img")]
    assert main(["search", *args]) == 0
    assert main(["csp", *args, "--instances", "3", "--budget", "5000"]) == 0
    out = capsys.readouterr().out
    assert "admissible=True consistent=True" in out
    search = json.loads((tmp_path / "r" / "search.json").read_text())
    assert {r["cost"] for r in search["results"] if r["strategy"] == "astar"} == {41}
    csp = json.loads((tmp_path / "r" / "csp.json").read_text())
    assert csp["australia"][0]["solutions"] == 18
    assert (tmp_path / "img" / "delivery-astar.png").stat().st_size > 0
    assert (tmp_path / "img" / "australia-colouring.png").stat().st_size > 0


def test_mlp_smoke(tmp_path):
    args = ["--results-dir", str(tmp_path / "r"), "--img-dir", str(tmp_path / "img")]
    assert main(["mlp", *args, "--max-epochs", "1"]) == 0
    runs = json.loads((tmp_path / "r" / "mlp.json").read_text())
    assert len(runs) == 5
    assert all(r["epochs"] == 1 for r in runs)
    assert (tmp_path / "img" / "mlp-curves.png").stat().st_size > 0


def test_no_files_writes_nothing(tmp_path):
    assert main(["search", "--no-files", "--results-dir", str(tmp_path / "r")]) == 0
    assert not (tmp_path / "r").exists()
