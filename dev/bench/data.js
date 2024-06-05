window.BENCHMARK_DATA = {
  "lastUpdate": 1717582180993,
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
          "id": "843a9641c11b617399994762a2ebe8c7aa702f57",
          "message": "Merge pull request #44 from ImperialCollegeLondon/improve-readme-step-import\n\nBiologic import bug fixes and README updates",
          "timestamp": "2024-06-05T11:06:44+01:00",
          "tree_id": "cbabcd0d8818be7fb62944405a7466900ab054ff",
          "url": "https://github.com/ImperialCollegeLondon/PyProBE/commit/843a9641c11b617399994762a2ebe8c7aa702f57"
        },
        "date": 1717582179931,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/cyclers/test_biologic.py::test_read_and_process",
            "value": 186.1736655933516,
            "unit": "iter/sec",
            "range": "stddev: 0.00006473398035097728",
            "extra": "mean: 5.371328951454619 msec\nrounds: 103"
          },
          {
            "name": "tests/cyclers/test_neware.py::test_read_and_process",
            "value": 0.1922691598746058,
            "unit": "iter/sec",
            "range": "stddev: 0.00831911311013592",
            "extra": "mean: 5.201042125800001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_cell.py::test_add_procedure",
            "value": 696.9983371952618,
            "unit": "iter/sec",
            "range": "stddev: 0.00007365057343538758",
            "extra": "mean: 1.4347236523174849 msec\nrounds: 604"
          },
          {
            "name": "tests/test_filter.py::test_step",
            "value": 13.634869338795284,
            "unit": "iter/sec",
            "range": "stddev: 0.0016110189067872708",
            "extra": "mean: 73.34137021428587 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_multi_step",
            "value": 13.408490360874913,
            "unit": "iter/sec",
            "range": "stddev: 0.0019034600553410707",
            "extra": "mean: 74.57961135713934 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_charge",
            "value": 12.682910397165598,
            "unit": "iter/sec",
            "range": "stddev: 0.009080615852325174",
            "extra": "mean: 78.84625600000155 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_discharge",
            "value": 13.064381295285607,
            "unit": "iter/sec",
            "range": "stddev: 0.0016202393260002547",
            "extra": "mean: 76.54399985714277 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_chargeordischarge",
            "value": 13.005868371365326,
            "unit": "iter/sec",
            "range": "stddev: 0.0021510709687479384",
            "extra": "mean: 76.88836849999754 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_rest",
            "value": 13.154640891908631,
            "unit": "iter/sec",
            "range": "stddev: 0.001511243926973646",
            "extra": "mean: 76.01879885714676 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_negative_cycle_index",
            "value": 12.932239076234824,
            "unit": "iter/sec",
            "range": "stddev: 0.0023443173334402557",
            "extra": "mean: 77.32612999999893 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_negative_step_index",
            "value": 13.102302664088631,
            "unit": "iter/sec",
            "range": "stddev: 0.0018543850090985553",
            "extra": "mean: 76.3224622142827 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_cycle",
            "value": 13.060763471499918,
            "unit": "iter/sec",
            "range": "stddev: 0.0012708081015676946",
            "extra": "mean: 76.56520250000044 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_all_steps",
            "value": 13.233071992953384,
            "unit": "iter/sec",
            "range": "stddev: 0.001976017264160307",
            "extra": "mean: 75.56824300000032 msec\nrounds: 14"
          },
          {
            "name": "tests/test_procedure.py::test_experiment",
            "value": 20373.29052181682,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036333974933804287",
            "extra": "mean: 49.08387277593406 usec\nrounds: 5227"
          },
          {
            "name": "tests/test_procedure.py::test_process_readme",
            "value": 758.2901708860404,
            "unit": "iter/sec",
            "range": "stddev: 0.00007078075614413263",
            "extra": "mean: 1.3187563790145513 msec\nrounds: 467"
          },
          {
            "name": "tests/test_rawdata.py::test_set_SOC",
            "value": 1709.3035139346507,
            "unit": "iter/sec",
            "range": "stddev: 0.00003019496510791731",
            "extra": "mean: 585.0336068742392 usec\nrounds: 931"
          }
        ]
      }
    ]
  }
}