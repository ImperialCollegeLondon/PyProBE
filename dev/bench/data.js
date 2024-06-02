window.BENCHMARK_DATA = {
  "lastUpdate": 1717343424923,
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
          "id": "4c67b036db6a7863678bd3473d7d5b0eff03a401",
          "message": "Merge pull request #39 from ImperialCollegeLondon/9-complete-docs\n\nRevert sphinx build and deploy to happen on push to main",
          "timestamp": "2024-06-02T16:47:15+01:00",
          "tree_id": "6e1a761355b54ef8d08239f5652a9d55f75d4fe0",
          "url": "https://github.com/ImperialCollegeLondon/PyProBE/commit/4c67b036db6a7863678bd3473d7d5b0eff03a401"
        },
        "date": 1717343422689,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/cyclers/test_biologic.py::test_read_and_process",
            "value": 190.49168357014875,
            "unit": "iter/sec",
            "range": "stddev: 0.0005822747870953321",
            "extra": "mean: 5.2495730063289034 msec\nrounds: 158"
          },
          {
            "name": "tests/cyclers/test_neware.py::test_read_and_process",
            "value": 0.1902586291756363,
            "unit": "iter/sec",
            "range": "stddev: 0.02288436775468067",
            "extra": "mean: 5.2560033903999965 sec\nrounds: 5"
          },
          {
            "name": "tests/test_cell.py::test_add_procedure",
            "value": 514.2804032679302,
            "unit": "iter/sec",
            "range": "stddev: 0.00009633512030798272",
            "extra": "mean: 1.944464524888807 msec\nrounds: 442"
          },
          {
            "name": "tests/test_filter.py::test_step",
            "value": 12.846578731465565,
            "unit": "iter/sec",
            "range": "stddev: 0.0019744553164808814",
            "extra": "mean: 77.84173676923537 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_multi_step",
            "value": 12.80851765545716,
            "unit": "iter/sec",
            "range": "stddev: 0.00224632477887753",
            "extra": "mean: 78.0730469285759 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_charge",
            "value": 12.699077937827685,
            "unit": "iter/sec",
            "range": "stddev: 0.0019790812602105386",
            "extra": "mean: 78.74587469230548 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_discharge",
            "value": 12.862084124687202,
            "unit": "iter/sec",
            "range": "stddev: 0.0023386566976659477",
            "extra": "mean: 77.74789764285727 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_chargeordischarge",
            "value": 13.001603825843244,
            "unit": "iter/sec",
            "range": "stddev: 0.0026718405081731606",
            "extra": "mean: 76.91358799998991 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_rest",
            "value": 13.500279352453502,
            "unit": "iter/sec",
            "range": "stddev: 0.0016761292070550688",
            "extra": "mean: 74.07254130769249 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_negative_cycle_index",
            "value": 13.214300621033342,
            "unit": "iter/sec",
            "range": "stddev: 0.0029193005845805607",
            "extra": "mean: 75.67559030769206 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_negative_step_index",
            "value": 13.188337510181839,
            "unit": "iter/sec",
            "range": "stddev: 0.0014730097006004843",
            "extra": "mean: 75.82456842858066 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_cycle",
            "value": 13.963873991776852,
            "unit": "iter/sec",
            "range": "stddev: 0.0023719492256169877",
            "extra": "mean: 71.61336464285536 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_all_steps",
            "value": 13.943547984305251,
            "unit": "iter/sec",
            "range": "stddev: 0.001882719550979263",
            "extra": "mean: 71.71775800001494 msec\nrounds: 15"
          },
          {
            "name": "tests/test_procedure.py::test_experiment",
            "value": 20491.992668588853,
            "unit": "iter/sec",
            "range": "stddev: 0.000007394341053929172",
            "extra": "mean: 48.79954898348416 usec\nrounds: 5359"
          },
          {
            "name": "tests/test_procedure.py::test_process_readme",
            "value": 556.3932737074058,
            "unit": "iter/sec",
            "range": "stddev: 0.00009037505430479115",
            "extra": "mean: 1.7972898797584613 msec\nrounds: 499"
          },
          {
            "name": "tests/test_rawdata.py::test_set_SOC",
            "value": 1879.7349543367309,
            "unit": "iter/sec",
            "range": "stddev: 0.000060933662834641125",
            "extra": "mean: 531.9898944757628 usec\nrounds: 1611"
          }
        ]
      }
    ]
  }
}