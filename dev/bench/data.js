window.BENCHMARK_DATA = {
  "lastUpdate": 1717348665762,
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
          "id": "22a57a06db7cea870eb10c3244cff154861d09d6",
          "message": "Merge pull request #40 from ImperialCollegeLondon/9-complete-docs\n\nComplete basic user documentation",
          "timestamp": "2024-06-02T18:14:42+01:00",
          "tree_id": "588f7a596328c160423e537008d100600257df97",
          "url": "https://github.com/ImperialCollegeLondon/PyProBE/commit/22a57a06db7cea870eb10c3244cff154861d09d6"
        },
        "date": 1717348664792,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/cyclers/test_biologic.py::test_read_and_process",
            "value": 191.36658725456633,
            "unit": "iter/sec",
            "range": "stddev: 0.00034569410991690334",
            "extra": "mean: 5.22557262658264 msec\nrounds: 158"
          },
          {
            "name": "tests/cyclers/test_neware.py::test_read_and_process",
            "value": 0.1879163095196311,
            "unit": "iter/sec",
            "range": "stddev: 0.026627231671606598",
            "extra": "mean: 5.321517874400001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_cell.py::test_add_procedure",
            "value": 467.12493396480073,
            "unit": "iter/sec",
            "range": "stddev: 0.00042310753074354296",
            "extra": "mean: 2.140754918630297 msec\nrounds: 467"
          },
          {
            "name": "tests/test_filter.py::test_step",
            "value": 13.286379849017782,
            "unit": "iter/sec",
            "range": "stddev: 0.002528323970246089",
            "extra": "mean: 75.26504671428061 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_multi_step",
            "value": 13.505009828574494,
            "unit": "iter/sec",
            "range": "stddev: 0.002012103685669674",
            "extra": "mean: 74.04659550000149 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_charge",
            "value": 12.578672555021738,
            "unit": "iter/sec",
            "range": "stddev: 0.0022059580207512766",
            "extra": "mean: 79.499644785711 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_discharge",
            "value": 12.041867476077183,
            "unit": "iter/sec",
            "range": "stddev: 0.002838158508021368",
            "extra": "mean: 83.04359784615109 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_chargeordischarge",
            "value": 11.594222547116122,
            "unit": "iter/sec",
            "range": "stddev: 0.0029857443916837755",
            "extra": "mean: 86.24985383334167 msec\nrounds: 12"
          },
          {
            "name": "tests/test_filter.py::test_rest",
            "value": 11.858716953867791,
            "unit": "iter/sec",
            "range": "stddev: 0.0025991160089247523",
            "extra": "mean: 84.32615466666012 msec\nrounds: 12"
          },
          {
            "name": "tests/test_filter.py::test_negative_cycle_index",
            "value": 12.2472964335732,
            "unit": "iter/sec",
            "range": "stddev: 0.00224978397234361",
            "extra": "mean: 81.65067330768002 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_negative_step_index",
            "value": 12.570561632561772,
            "unit": "iter/sec",
            "range": "stddev: 0.003359686289618515",
            "extra": "mean: 79.55094046153678 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_cycle",
            "value": 12.732999984362749,
            "unit": "iter/sec",
            "range": "stddev: 0.0029397338174965205",
            "extra": "mean: 78.53608742857838 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_all_steps",
            "value": 12.079802232525951,
            "unit": "iter/sec",
            "range": "stddev: 0.007402845180758093",
            "extra": "mean: 82.78281223076735 msec\nrounds: 13"
          },
          {
            "name": "tests/test_procedure.py::test_experiment",
            "value": 19976.164915858764,
            "unit": "iter/sec",
            "range": "stddev: 0.000006085305601554246",
            "extra": "mean: 50.05965880898969 usec\nrounds: 5642"
          },
          {
            "name": "tests/test_procedure.py::test_process_readme",
            "value": 548.657160846488,
            "unit": "iter/sec",
            "range": "stddev: 0.00010314965874405437",
            "extra": "mean: 1.8226318206749805 msec\nrounds: 474"
          },
          {
            "name": "tests/test_rawdata.py::test_set_SOC",
            "value": 1778.693760699548,
            "unit": "iter/sec",
            "range": "stddev: 0.00018494661797310464",
            "extra": "mean: 562.2103265301313 usec\nrounds: 1127"
          }
        ]
      }
    ]
  }
}