window.BENCHMARK_DATA = {
  "lastUpdate": 1717338822297,
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
          "id": "00ee1a4ffc856c37c9a34f481c7e7f2639c30fd9",
          "message": "Merge pull request #36 from ImperialCollegeLondon/9-complete-docs\n\nAdd user guide for filtering data",
          "timestamp": "2024-06-02T15:30:40+01:00",
          "tree_id": "f104f555fee55451367fd978dd963849d0a71e22",
          "url": "https://github.com/ImperialCollegeLondon/PyProBE/commit/00ee1a4ffc856c37c9a34f481c7e7f2639c30fd9"
        },
        "date": 1717338821471,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/cyclers/test_biologic.py::test_read_and_process",
            "value": 193.32854755288412,
            "unit": "iter/sec",
            "range": "stddev: 0.0000576866733663054",
            "extra": "mean: 5.172541834394399 msec\nrounds: 157"
          },
          {
            "name": "tests/cyclers/test_neware.py::test_read_and_process",
            "value": 0.1863857225534372,
            "unit": "iter/sec",
            "range": "stddev: 0.06320664140933846",
            "extra": "mean: 5.3652178198 sec\nrounds: 5"
          },
          {
            "name": "tests/test_cell.py::test_add_procedure",
            "value": 522.9007307281387,
            "unit": "iter/sec",
            "range": "stddev: 0.00009120696847071487",
            "extra": "mean: 1.9124088784643714 msec\nrounds: 469"
          },
          {
            "name": "tests/test_filter.py::test_step",
            "value": 13.548390929657801,
            "unit": "iter/sec",
            "range": "stddev: 0.0020751315319283217",
            "extra": "mean: 73.80950292857084 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_multi_step",
            "value": 13.669525297734243,
            "unit": "iter/sec",
            "range": "stddev: 0.0006464258755579873",
            "extra": "mean: 73.15542992306781 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_charge",
            "value": 13.422348276223964,
            "unit": "iter/sec",
            "range": "stddev: 0.0017499243202680642",
            "extra": "mean: 74.50261157143245 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_discharge",
            "value": 13.324272789767713,
            "unit": "iter/sec",
            "range": "stddev: 0.002038692884993588",
            "extra": "mean: 75.05100021428136 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_chargeordischarge",
            "value": 12.965226415382785,
            "unit": "iter/sec",
            "range": "stddev: 0.0021686661776469615",
            "extra": "mean: 77.12938964286309 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_rest",
            "value": 12.604630822298903,
            "unit": "iter/sec",
            "range": "stddev: 0.0025750978525595253",
            "extra": "mean: 79.33592138461492 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_negative_cycle_index",
            "value": 12.764125916949837,
            "unit": "iter/sec",
            "range": "stddev: 0.0018278284906097805",
            "extra": "mean: 78.34457341666241 msec\nrounds: 12"
          },
          {
            "name": "tests/test_filter.py::test_negative_step_index",
            "value": 13.410015171746055,
            "unit": "iter/sec",
            "range": "stddev: 0.0021608463962360237",
            "extra": "mean: 74.5711311428587 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_cycle",
            "value": 13.920550660832676,
            "unit": "iter/sec",
            "range": "stddev: 0.0015869425521149237",
            "extra": "mean: 71.83623869231216 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_all_steps",
            "value": 13.277557400147199,
            "unit": "iter/sec",
            "range": "stddev: 0.0025305892585659588",
            "extra": "mean: 75.31505757142602 msec\nrounds: 14"
          },
          {
            "name": "tests/test_procedure.py::test_experiment",
            "value": 19863.778552130916,
            "unit": "iter/sec",
            "range": "stddev: 0.0000054695540189964275",
            "extra": "mean: 50.342889061896216 usec\nrounds: 5778"
          },
          {
            "name": "tests/test_procedure.py::test_process_readme",
            "value": 555.4450966902231,
            "unit": "iter/sec",
            "range": "stddev: 0.00013715957746071475",
            "extra": "mean: 1.8003579578949986 msec\nrounds: 475"
          },
          {
            "name": "tests/test_rawdata.py::test_set_SOC",
            "value": 1645.7205515187643,
            "unit": "iter/sec",
            "range": "stddev: 0.0001719383365141362",
            "extra": "mean: 607.6365754059176 usec\nrounds: 862"
          }
        ]
      }
    ]
  }
}