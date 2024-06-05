window.BENCHMARK_DATA = {
  "lastUpdate": 1717583156191,
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
          "id": "13c2d317e48b415ceb7a166f3063895845fca418",
          "message": "Merge pull request #45 from ImperialCollegeLondon/9-complete-docs\n\nReorganise base method and experiment classes, add introduction section to docs",
          "timestamp": "2024-06-05T11:22:55+01:00",
          "tree_id": "772aabeccff5caa0d5186b32d12250a2ae9d93c8",
          "url": "https://github.com/ImperialCollegeLondon/PyProBE/commit/13c2d317e48b415ceb7a166f3063895845fca418"
        },
        "date": 1717583154445,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/cyclers/test_biologic.py::test_read_and_process",
            "value": 188.2265300061216,
            "unit": "iter/sec",
            "range": "stddev: 0.0006246301831544016",
            "extra": "mean: 5.3127473580237465 msec\nrounds: 162"
          },
          {
            "name": "tests/cyclers/test_neware.py::test_read_and_process",
            "value": 0.18848814228385743,
            "unit": "iter/sec",
            "range": "stddev: 0.010958559136428849",
            "extra": "mean: 5.305373525799996 sec\nrounds: 5"
          },
          {
            "name": "tests/test_cell.py::test_add_procedure",
            "value": 702.0982076158026,
            "unit": "iter/sec",
            "range": "stddev: 0.00005617898875901271",
            "extra": "mean: 1.424302169059536 msec\nrounds: 627"
          },
          {
            "name": "tests/test_filter.py::test_step",
            "value": 13.443143768431712,
            "unit": "iter/sec",
            "range": "stddev: 0.0022371253802698837",
            "extra": "mean: 74.38736185714845 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_multi_step",
            "value": 13.639909924372533,
            "unit": "iter/sec",
            "range": "stddev: 0.0014907929895045004",
            "extra": "mean: 73.31426714286034 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_charge",
            "value": 13.453103068393487,
            "unit": "iter/sec",
            "range": "stddev: 0.0021804304242273903",
            "extra": "mean: 74.33229307143156 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_discharge",
            "value": 13.236836918265311,
            "unit": "iter/sec",
            "range": "stddev: 0.00209141413165974",
            "extra": "mean: 75.54674928570851 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_chargeordischarge",
            "value": 12.715179968701392,
            "unit": "iter/sec",
            "range": "stddev: 0.0019041159139525712",
            "extra": "mean: 78.64615384615202 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_rest",
            "value": 12.948112173660585,
            "unit": "iter/sec",
            "range": "stddev: 0.0013733392873645264",
            "extra": "mean: 77.23133585714744 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_negative_cycle_index",
            "value": 12.534019135725378,
            "unit": "iter/sec",
            "range": "stddev: 0.00768203425031895",
            "extra": "mean: 79.78286846153975 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_negative_step_index",
            "value": 12.56238175938611,
            "unit": "iter/sec",
            "range": "stddev: 0.0018241181998588445",
            "extra": "mean: 79.60273928571227 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_cycle",
            "value": 12.975617245146292,
            "unit": "iter/sec",
            "range": "stddev: 0.002109073344151176",
            "extra": "mean: 77.06762469231002 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_all_steps",
            "value": 12.61030251005941,
            "unit": "iter/sec",
            "range": "stddev: 0.0042039215645395525",
            "extra": "mean: 79.30023876923542 msec\nrounds: 13"
          },
          {
            "name": "tests/test_procedure.py::test_experiment",
            "value": 20641.96078164935,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032587899127096856",
            "extra": "mean: 48.445010170206174 usec\nrounds: 5703"
          },
          {
            "name": "tests/test_procedure.py::test_process_readme",
            "value": 769.5658007183604,
            "unit": "iter/sec",
            "range": "stddev: 0.000022030392962303608",
            "extra": "mean: 1.2994340432832878 msec\nrounds: 670"
          },
          {
            "name": "tests/test_rawdata.py::test_set_SOC",
            "value": 1897.8285532048972,
            "unit": "iter/sec",
            "range": "stddev: 0.000055950071894016156",
            "extra": "mean: 526.9179865121546 usec\nrounds: 1038"
          }
        ]
      }
    ]
  }
}