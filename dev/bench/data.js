window.BENCHMARK_DATA = {
  "lastUpdate": 1717583450248,
  "repoUrl": "https://github.com/ImperialCollegeLondon/PyProBE",
  "entries": {
    "Python Benchmark with pytest-benchmark": [
      {
        "commit": {
          "author": {
            "email": "137503955+tomjholland@users.noreply.github.com",
            "name": "Tom Holland",
            "username": "tomjholland"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3266a03bcb67ca213d33ceaa8fd62b90bf0be371",
          "message": "Merge pull request #46 from ImperialCollegeLondon/update-version-number\n\nUpdate version number",
          "timestamp": "2024-06-05T11:27:52+01:00",
          "tree_id": "e8aa7891d6ea912daa3144b68ecd13a936caad40",
          "url": "https://github.com/ImperialCollegeLondon/PyProBE/commit/3266a03bcb67ca213d33ceaa8fd62b90bf0be371"
        },
        "date": 1717583448273,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/cyclers/test_biologic.py::test_read_and_process",
            "value": 188.93309515199633,
            "unit": "iter/sec",
            "range": "stddev: 0.0003178483451748549",
            "extra": "mean: 5.292878937888049 msec\nrounds: 161"
          },
          {
            "name": "tests/cyclers/test_neware.py::test_read_and_process",
            "value": 0.19425392621979637,
            "unit": "iter/sec",
            "range": "stddev: 0.010085770421144915",
            "extra": "mean: 5.147901097600004 sec\nrounds: 5"
          },
          {
            "name": "tests/test_cell.py::test_add_procedure",
            "value": 703.001325019116,
            "unit": "iter/sec",
            "range": "stddev: 0.00007393803380607148",
            "extra": "mean: 1.4224724256000627 msec\nrounds: 625"
          },
          {
            "name": "tests/test_filter.py::test_step",
            "value": 13.152562534709052,
            "unit": "iter/sec",
            "range": "stddev: 0.0015387933410871157",
            "extra": "mean: 76.03081128571277 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_multi_step",
            "value": 12.951764543543565,
            "unit": "iter/sec",
            "range": "stddev: 0.0015661993610094904",
            "extra": "mean: 77.20955678571987 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_charge",
            "value": 12.694607936865559,
            "unit": "iter/sec",
            "range": "stddev: 0.0013738314229119258",
            "extra": "mean: 78.77360253844209 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_discharge",
            "value": 12.45631339731106,
            "unit": "iter/sec",
            "range": "stddev: 0.0020281556848905234",
            "extra": "mean: 80.28057484615549 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_chargeordischarge",
            "value": 11.7859952491079,
            "unit": "iter/sec",
            "range": "stddev: 0.007665772544060025",
            "extra": "mean: 84.8464621666712 msec\nrounds: 12"
          },
          {
            "name": "tests/test_filter.py::test_rest",
            "value": 12.551688795837912,
            "unit": "iter/sec",
            "range": "stddev: 0.0017157054318507282",
            "extra": "mean: 79.67055400000007 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_negative_cycle_index",
            "value": 12.47355893357423,
            "unit": "iter/sec",
            "range": "stddev: 0.0019200937544655954",
            "extra": "mean: 80.16958153846278 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_negative_step_index",
            "value": 12.81366591269017,
            "unit": "iter/sec",
            "range": "stddev: 0.0027809895089759657",
            "extra": "mean: 78.04167884614799 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_cycle",
            "value": 12.807630431987997,
            "unit": "iter/sec",
            "range": "stddev: 0.0012110679506894834",
            "extra": "mean: 78.07845528571988 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_all_steps",
            "value": 12.78876671248857,
            "unit": "iter/sec",
            "range": "stddev: 0.0015629904096892012",
            "extra": "mean: 78.19362276922867 msec\nrounds: 13"
          },
          {
            "name": "tests/test_procedure.py::test_experiment",
            "value": 20915.379581628742,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034362120700774376",
            "extra": "mean: 47.81170698323645 usec\nrounds: 5771"
          },
          {
            "name": "tests/test_procedure.py::test_process_readme",
            "value": 772.0749934813451,
            "unit": "iter/sec",
            "range": "stddev: 0.00007554126601443922",
            "extra": "mean: 1.2952109684202096 msec\nrounds: 665"
          },
          {
            "name": "tests/test_rawdata.py::test_set_SOC",
            "value": 1857.5684819213802,
            "unit": "iter/sec",
            "range": "stddev: 0.00012794177002254018",
            "extra": "mean: 538.3381607366894 usec\nrounds: 1574"
          }
        ]
      }
    ]
  }
}