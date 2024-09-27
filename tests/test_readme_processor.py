"""Tests for the readme_processor module."""

from pyprobe.readme_processor import process_readme


def test_process_readme(titles_fixture, benchmark):
    """Test processing a readme file in yaml format."""

    def _process_readme():
        return process_readme("tests/sample_data/neware/README.yaml")

    readme = benchmark(_process_readme)

    assert list(readme.experiment_dict.keys()) == titles_fixture
    assert readme.experiment_dict["Break-in Cycles"]["Steps"] == [4, 5, 6, 7]
    assert readme.experiment_dict["Break-in Cycles"]["Step Descriptions"] == [
        "Discharge at 4 mA until 3 V",
        "Rest for 2 hours",
        "Charge at 4 mA until 4.2 V, Hold at 4.2 V until 0.04 A",
        "Rest for 2 hours",
    ]
