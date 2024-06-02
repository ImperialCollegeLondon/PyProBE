window.BENCHMARK_DATA = {
  "lastUpdate": 1717335733732,
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
          "id": "a96863dd9c569dbe6acece42ac47b28d74d817c8",
          "message": "Merge pull request #35 from ImperialCollegeLondon/9-complete-docs\n\nAdd setup documentation and change user input to make_cell_list to require full filepath for experiment records",
          "timestamp": "2024-06-02T14:39:08+01:00",
          "tree_id": "0b4193873f3e60226fff7e6da0813de4618d490e",
          "url": "https://github.com/ImperialCollegeLondon/PyProBE/commit/a96863dd9c569dbe6acece42ac47b28d74d817c8"
        },
        "date": 1717335732599,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/cyclers/test_biologic.py::test_read_and_process",
            "value": 192.51576204755526,
            "unit": "iter/sec",
            "range": "stddev: 0.0005645897709033862",
            "extra": "mean: 5.19437987499943 msec\nrounds: 136"
          },
          {
            "name": "tests/cyclers/test_neware.py::test_read_and_process",
            "value": 0.19257556928714809,
            "unit": "iter/sec",
            "range": "stddev: 0.028800852090706028",
            "extra": "mean: 5.192766682199999 sec\nrounds: 5"
          },
          {
            "name": "tests/test_cell.py::test_add_procedure",
            "value": 514.0433583100865,
            "unit": "iter/sec",
            "range": "stddev: 0.0001031824435718556",
            "extra": "mean: 1.9453611914907178 msec\nrounds: 470"
          },
          {
            "name": "tests/test_filter.py::test_step",
            "value": 13.66384548923461,
            "unit": "iter/sec",
            "range": "stddev: 0.0015555823583666714",
            "extra": "mean: 73.18583928571749 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_multi_step",
            "value": 13.457965488164922,
            "unit": "iter/sec",
            "range": "stddev: 0.002794106592703504",
            "extra": "mean: 74.30543649999777 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_charge",
            "value": 13.215904159531311,
            "unit": "iter/sec",
            "range": "stddev: 0.002135490036729898",
            "extra": "mean: 75.66640828571687 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_discharge",
            "value": 13.338632645049657,
            "unit": "iter/sec",
            "range": "stddev: 0.0016270662114059318",
            "extra": "mean: 74.97020321427986 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_chargeordischarge",
            "value": 12.44178769868634,
            "unit": "iter/sec",
            "range": "stddev: 0.0024999779687021875",
            "extra": "mean: 80.37430184615548 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_rest",
            "value": 12.917420039050832,
            "unit": "iter/sec",
            "range": "stddev: 0.003437378399561099",
            "extra": "mean: 77.41483957143812 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_negative_cycle_index",
            "value": 13.141365122680918,
            "unit": "iter/sec",
            "range": "stddev: 0.002266702582314997",
            "extra": "mean: 76.09559514285787 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_negative_step_index",
            "value": 13.42285186647013,
            "unit": "iter/sec",
            "range": "stddev: 0.002251648707547319",
            "extra": "mean: 74.4998164285765 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_cycle",
            "value": 12.744606005500007,
            "unit": "iter/sec",
            "range": "stddev: 0.009310714734154174",
            "extra": "mean: 78.46456764284783 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_all_steps",
            "value": 12.861092420893616,
            "unit": "iter/sec",
            "range": "stddev: 0.0028216380901835113",
            "extra": "mean: 77.75389269230661 msec\nrounds: 13"
          },
          {
            "name": "tests/test_procedure.py::test_experiment",
            "value": 20844.629794275777,
            "unit": "iter/sec",
            "range": "stddev: 0.0000037633173780156915",
            "extra": "mean: 47.973987058988875 usec\nrounds: 5718"
          },
          {
            "name": "tests/test_procedure.py::test_process_readme",
            "value": 557.508683564714,
            "unit": "iter/sec",
            "range": "stddev: 0.000029792238371187537",
            "extra": "mean: 1.7936940346937624 msec\nrounds: 490"
          },
          {
            "name": "tests/test_rawdata.py::test_set_SOC",
            "value": 1694.4881959333131,
            "unit": "iter/sec",
            "range": "stddev: 0.00011049171133999015",
            "extra": "mean: 590.1486964618284 usec\nrounds: 537"
          }
        ]
      }
    ]
  }
}