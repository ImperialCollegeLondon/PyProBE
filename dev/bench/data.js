window.BENCHMARK_DATA = {
  "lastUpdate": 1717341580586,
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
          "id": "b40574a87daac367617bf5620945419ac7ef5e24",
          "message": "Merge pull request #37 from ImperialCollegeLondon/9-complete-docs\n\nAdd accessing data section to docs",
          "timestamp": "2024-06-02T16:16:26+01:00",
          "tree_id": "8e801d73ed540758ab235319ef42ffe28f103635",
          "url": "https://github.com/ImperialCollegeLondon/PyProBE/commit/b40574a87daac367617bf5620945419ac7ef5e24"
        },
        "date": 1717341578465,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/cyclers/test_biologic.py::test_read_and_process",
            "value": 191.0447986345156,
            "unit": "iter/sec",
            "range": "stddev: 0.0005250352042144614",
            "extra": "mean: 5.234374383115671 msec\nrounds: 154"
          },
          {
            "name": "tests/cyclers/test_neware.py::test_read_and_process",
            "value": 0.1835124648540507,
            "unit": "iter/sec",
            "range": "stddev: 0.04127113943416911",
            "extra": "mean: 5.449221123999996 sec\nrounds: 5"
          },
          {
            "name": "tests/test_cell.py::test_add_procedure",
            "value": 514.2564320008643,
            "unit": "iter/sec",
            "range": "stddev: 0.00018163414555758006",
            "extra": "mean: 1.9445551630909292 msec\nrounds: 466"
          },
          {
            "name": "tests/test_filter.py::test_step",
            "value": 13.632021903775325,
            "unit": "iter/sec",
            "range": "stddev: 0.0016449446744684151",
            "extra": "mean: 73.35668964286616 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_multi_step",
            "value": 13.574237945578677,
            "unit": "iter/sec",
            "range": "stddev: 0.0015486930509501673",
            "extra": "mean: 73.66896057142672 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_charge",
            "value": 13.336309286296819,
            "unit": "iter/sec",
            "range": "stddev: 0.001343321489134382",
            "extra": "mean: 74.98326399999655 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_discharge",
            "value": 13.105390158976682,
            "unit": "iter/sec",
            "range": "stddev: 0.002097509696960505",
            "extra": "mean: 76.30448142858525 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_chargeordischarge",
            "value": 13.140181870478662,
            "unit": "iter/sec",
            "range": "stddev: 0.001619796049358109",
            "extra": "mean: 76.10244742857374 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_rest",
            "value": 12.820553343070156,
            "unit": "iter/sec",
            "range": "stddev: 0.0017057980796407052",
            "extra": "mean: 77.99975346154042 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_negative_cycle_index",
            "value": 12.893736170203773,
            "unit": "iter/sec",
            "range": "stddev: 0.0021217477150950216",
            "extra": "mean: 77.5570390769207 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_negative_step_index",
            "value": 12.423124220982382,
            "unit": "iter/sec",
            "range": "stddev: 0.006878529409240388",
            "extra": "mean: 80.49504957142923 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_cycle",
            "value": 13.046115180564216,
            "unit": "iter/sec",
            "range": "stddev: 0.0014181209299215982",
            "extra": "mean: 76.65117057143382 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_all_steps",
            "value": 13.069308529668609,
            "unit": "iter/sec",
            "range": "stddev: 0.0034346712955478317",
            "extra": "mean: 76.51514215384097 msec\nrounds: 13"
          },
          {
            "name": "tests/test_procedure.py::test_experiment",
            "value": 20040.68424115531,
            "unit": "iter/sec",
            "range": "stddev: 0.000021847690801747463",
            "extra": "mean: 49.89849587802057 usec\nrounds: 5701"
          },
          {
            "name": "tests/test_procedure.py::test_process_readme",
            "value": 549.1451650304781,
            "unit": "iter/sec",
            "range": "stddev: 0.00010249367421704523",
            "extra": "mean: 1.8210121178878065 msec\nrounds: 492"
          },
          {
            "name": "tests/test_rawdata.py::test_set_SOC",
            "value": 1714.5403719076812,
            "unit": "iter/sec",
            "range": "stddev: 0.000019526667998384583",
            "extra": "mean: 583.2466918742492 usec\nrounds: 529"
          }
        ]
      }
    ]
  }
}