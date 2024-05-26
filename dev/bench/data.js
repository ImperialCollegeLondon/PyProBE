window.BENCHMARK_DATA = {
  "lastUpdate": 1716720628937,
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
          "id": "409ab3c5e3f87265eb790ee2312118e6fe0e7691",
          "message": "Merge pull request #30 from ImperialCollegeLondon/tomjholland-patch-1\n\nUpdate deploy_benchmark.yml",
          "timestamp": "2024-05-26T11:47:22+01:00",
          "tree_id": "703943aa20f7b9405ab6d1021894e0da7e52c875",
          "url": "https://github.com/ImperialCollegeLondon/PyProBE/commit/409ab3c5e3f87265eb790ee2312118e6fe0e7691"
        },
        "date": 1716720627817,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/cyclers/test_biologic.py::test_read_and_process",
            "value": 191.28751267680198,
            "unit": "iter/sec",
            "range": "stddev: 0.00033007692864619933",
            "extra": "mean: 5.2277327777772555 msec\nrounds: 171"
          },
          {
            "name": "tests/cyclers/test_neware.py::test_read_and_process",
            "value": 0.19294715847045232,
            "unit": "iter/sec",
            "range": "stddev: 0.011881191276018412",
            "extra": "mean: 5.1827661413999975 sec\nrounds: 5"
          },
          {
            "name": "tests/test_cell.py::test_add_procedure",
            "value": 517.8508510403105,
            "unit": "iter/sec",
            "range": "stddev: 0.00009945512059000542",
            "extra": "mean: 1.9310579445628022 msec\nrounds: 469"
          },
          {
            "name": "tests/test_filter.py::test_step",
            "value": 10.883610250244878,
            "unit": "iter/sec",
            "range": "stddev: 0.001968148125279104",
            "extra": "mean: 91.88127624999254 msec\nrounds: 12"
          },
          {
            "name": "tests/test_filter.py::test_multi_step",
            "value": 10.325082654716187,
            "unit": "iter/sec",
            "range": "stddev: 0.013221801476535658",
            "extra": "mean: 96.85152491667755 msec\nrounds: 12"
          },
          {
            "name": "tests/test_filter.py::test_charge",
            "value": 10.86070918906531,
            "unit": "iter/sec",
            "range": "stddev: 0.0034373117745214975",
            "extra": "mean: 92.07501854545667 msec\nrounds: 11"
          },
          {
            "name": "tests/test_filter.py::test_discharge",
            "value": 11.07025738201778,
            "unit": "iter/sec",
            "range": "stddev: 0.0018149106838515276",
            "extra": "mean: 90.33213641666293 msec\nrounds: 12"
          },
          {
            "name": "tests/test_filter.py::test_chargeordischarge",
            "value": 10.492376585897741,
            "unit": "iter/sec",
            "range": "stddev: 0.0019563769997643517",
            "extra": "mean: 95.3072920909118 msec\nrounds: 11"
          },
          {
            "name": "tests/test_filter.py::test_rest",
            "value": 11.022534169410337,
            "unit": "iter/sec",
            "range": "stddev: 0.0018423782399008546",
            "extra": "mean: 90.72323883333411 msec\nrounds: 12"
          },
          {
            "name": "tests/test_filter.py::test_negative_cycle_index",
            "value": 10.82887857032282,
            "unit": "iter/sec",
            "range": "stddev: 0.0018733607790356824",
            "extra": "mean: 92.34566566667013 msec\nrounds: 12"
          },
          {
            "name": "tests/test_filter.py::test_negative_step_index",
            "value": 10.152920416810696,
            "unit": "iter/sec",
            "range": "stddev: 0.0018779744881259614",
            "extra": "mean: 98.49382827272538 msec\nrounds: 11"
          },
          {
            "name": "tests/test_filter.py::test_cycle",
            "value": 10.634193042971985,
            "unit": "iter/sec",
            "range": "stddev: 0.0018097716302277462",
            "extra": "mean: 94.03628427272989 msec\nrounds: 11"
          },
          {
            "name": "tests/test_filter.py::test_all_steps",
            "value": 10.420934414306567,
            "unit": "iter/sec",
            "range": "stddev: 0.00190291109204914",
            "extra": "mean: 95.96068454544077 msec\nrounds: 11"
          },
          {
            "name": "tests/test_procedure.py::test_experiment",
            "value": 20636.12899096516,
            "unit": "iter/sec",
            "range": "stddev: 0.000003301443041689338",
            "extra": "mean: 48.458700778514064 usec\nrounds: 5912"
          },
          {
            "name": "tests/test_procedure.py::test_process_readme",
            "value": 551.3788606855313,
            "unit": "iter/sec",
            "range": "stddev: 0.000119305511463585",
            "extra": "mean: 1.8136350000010815 msec\nrounds: 477"
          },
          {
            "name": "tests/test_rawdata.py::test_set_SOC",
            "value": 1688.9983348569472,
            "unit": "iter/sec",
            "range": "stddev: 0.00010400843951990623",
            "extra": "mean: 592.0668951308924 usec\nrounds: 1335"
          }
        ]
      }
    ]
  }
}