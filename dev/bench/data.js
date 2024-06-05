window.BENCHMARK_DATA = {
  "lastUpdate": 1717594200795,
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
          "id": "bf3a21ccc6191b15a56aa5768a70fcacbe9b1724",
          "message": "Merge pull request #47 from ImperialCollegeLondon/fix-benchmark-plot\n\nAdd keep_files command for dev/bench",
          "timestamp": "2024-06-05T14:26:58+01:00",
          "tree_id": "94fbf6a834a4678ca7186aaf4549fc413d1cc517",
          "url": "https://github.com/ImperialCollegeLondon/PyProBE/commit/bf3a21ccc6191b15a56aa5768a70fcacbe9b1724"
        },
        "date": 1717594198750,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/cyclers/test_biologic.py::test_read_and_process",
            "value": 183.84372074966544,
            "unit": "iter/sec",
            "range": "stddev: 0.0006613811574584529",
            "extra": "mean: 5.439402531249193 msec\nrounds: 160"
          },
          {
            "name": "tests/cyclers/test_neware.py::test_read_and_process",
            "value": 0.19291453629311547,
            "unit": "iter/sec",
            "range": "stddev: 0.013093911868787455",
            "extra": "mean: 5.183642556000001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_cell.py::test_add_procedure",
            "value": 685.0744738491777,
            "unit": "iter/sec",
            "range": "stddev: 0.0000762090136498428",
            "extra": "mean: 1.459695315140518 msec\nrounds: 568"
          },
          {
            "name": "tests/test_filter.py::test_step",
            "value": 12.33545966054264,
            "unit": "iter/sec",
            "range": "stddev: 0.0019582190292230317",
            "extra": "mean: 81.06710471428106 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_multi_step",
            "value": 12.266686849455507,
            "unit": "iter/sec",
            "range": "stddev: 0.0014747208923606876",
            "extra": "mean: 81.52160499999948 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_charge",
            "value": 12.036665806202295,
            "unit": "iter/sec",
            "range": "stddev: 0.0019043124610129683",
            "extra": "mean: 83.07948530769347 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_discharge",
            "value": 12.462545002340628,
            "unit": "iter/sec",
            "range": "stddev: 0.003755532495095942",
            "extra": "mean: 80.24043241666827 msec\nrounds: 12"
          },
          {
            "name": "tests/test_filter.py::test_chargeordischarge",
            "value": 12.911813995402115,
            "unit": "iter/sec",
            "range": "stddev: 0.0032482014495708628",
            "extra": "mean: 77.44845150000604 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_rest",
            "value": 13.393141443558886,
            "unit": "iter/sec",
            "range": "stddev: 0.0016172368201342903",
            "extra": "mean: 74.66508169231098 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_negative_cycle_index",
            "value": 12.947023318198346,
            "unit": "iter/sec",
            "range": "stddev: 0.003112061252651953",
            "extra": "mean: 77.23783107692401 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_negative_step_index",
            "value": 12.525706145403085,
            "unit": "iter/sec",
            "range": "stddev: 0.008903435093548224",
            "extra": "mean: 79.83581830769666 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_cycle",
            "value": 13.465075555626235,
            "unit": "iter/sec",
            "range": "stddev: 0.003671168609355533",
            "extra": "mean: 74.26620042857175 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_all_steps",
            "value": 13.48695514724224,
            "unit": "iter/sec",
            "range": "stddev: 0.0028053407012203144",
            "extra": "mean: 74.14572000000135 msec\nrounds: 15"
          },
          {
            "name": "tests/test_procedure.py::test_experiment",
            "value": 20774.65889769019,
            "unit": "iter/sec",
            "range": "stddev: 0.0000042140408087990905",
            "extra": "mean: 48.13556770894486 usec\nrounds: 5723"
          },
          {
            "name": "tests/test_procedure.py::test_process_readme",
            "value": 757.5337654057321,
            "unit": "iter/sec",
            "range": "stddev: 0.00006925856176812249",
            "extra": "mean: 1.3200731712129081 msec\nrounds: 660"
          },
          {
            "name": "tests/test_rawdata.py::test_set_SOC",
            "value": 1610.2639698175233,
            "unit": "iter/sec",
            "range": "stddev: 0.00016249729349012543",
            "extra": "mean: 621.0161928378246 usec\nrounds: 726"
          }
        ]
      }
    ]
  }
}