window.BENCHMARK_DATA = {
  "lastUpdate": 1717342519870,
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
          "id": "2eb5978a55138cdbdaeca66472f174da25dbea0a",
          "message": "Merge pull request #38 from ImperialCollegeLondon/9-complete-docs\n\nAdd plotting to user guide",
          "timestamp": "2024-06-02T16:32:15+01:00",
          "tree_id": "3ff9703f161647ac430a941dd75a6f10cdc4a6a4",
          "url": "https://github.com/ImperialCollegeLondon/PyProBE/commit/2eb5978a55138cdbdaeca66472f174da25dbea0a"
        },
        "date": 1717342517718,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/cyclers/test_biologic.py::test_read_and_process",
            "value": 194.34160113380244,
            "unit": "iter/sec",
            "range": "stddev: 0.00006984520087036069",
            "extra": "mean: 5.145578682927023 msec\nrounds: 164"
          },
          {
            "name": "tests/cyclers/test_neware.py::test_read_and_process",
            "value": 0.19158211154922336,
            "unit": "iter/sec",
            "range": "stddev: 0.030457067576690647",
            "extra": "mean: 5.2196940095999995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_cell.py::test_add_procedure",
            "value": 525.6292801102741,
            "unit": "iter/sec",
            "range": "stddev: 0.00009870391986827517",
            "extra": "mean: 1.9024815356370663 msec\nrounds: 463"
          },
          {
            "name": "tests/test_filter.py::test_step",
            "value": 13.584384413999494,
            "unit": "iter/sec",
            "range": "stddev: 0.0015429163167593196",
            "extra": "mean: 73.61393564285785 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_multi_step",
            "value": 13.634359460778763,
            "unit": "iter/sec",
            "range": "stddev: 0.0012153265567338932",
            "extra": "mean: 73.34411292856454 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_charge",
            "value": 13.011602211483488,
            "unit": "iter/sec",
            "range": "stddev: 0.001970688007410576",
            "extra": "mean: 76.85448599999793 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_discharge",
            "value": 13.12981717810985,
            "unit": "iter/sec",
            "range": "stddev: 0.0017824513570730905",
            "extra": "mean: 76.16252278571015 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_chargeordischarge",
            "value": 12.9029200250849,
            "unit": "iter/sec",
            "range": "stddev: 0.0018375660017455728",
            "extra": "mean: 77.50183664285869 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_rest",
            "value": 12.921003555540425,
            "unit": "iter/sec",
            "range": "stddev: 0.00260862962290332",
            "extra": "mean: 77.39336930769653 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_negative_cycle_index",
            "value": 12.980690848703434,
            "unit": "iter/sec",
            "range": "stddev: 0.0023652622571147313",
            "extra": "mean: 77.03750221428963 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_negative_step_index",
            "value": 13.073194396887008,
            "unit": "iter/sec",
            "range": "stddev: 0.0020516638670901753",
            "extra": "mean: 76.49239884615501 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_cycle",
            "value": 13.335490878982917,
            "unit": "iter/sec",
            "range": "stddev: 0.001648887264801002",
            "extra": "mean: 74.98786576923285 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_all_steps",
            "value": 13.153662125730252,
            "unit": "iter/sec",
            "range": "stddev: 0.0019125435385774308",
            "extra": "mean: 76.02445542856628 msec\nrounds: 14"
          },
          {
            "name": "tests/test_procedure.py::test_experiment",
            "value": 20510.803489745518,
            "unit": "iter/sec",
            "range": "stddev: 0.00000586854795066176",
            "extra": "mean: 48.75479405280027 usec\nrounds: 5448"
          },
          {
            "name": "tests/test_procedure.py::test_process_readme",
            "value": 558.5523039230859,
            "unit": "iter/sec",
            "range": "stddev: 0.00009713960238799453",
            "extra": "mean: 1.7903426285709183 msec\nrounds: 490"
          },
          {
            "name": "tests/test_rawdata.py::test_set_SOC",
            "value": 1663.5772569776518,
            "unit": "iter/sec",
            "range": "stddev: 0.0001373566299313356",
            "extra": "mean: 601.1142529183024 usec\nrounds: 514"
          }
        ]
      }
    ]
  }
}