window.BENCHMARK_DATA = {
  "lastUpdate": 1716905503150,
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
      },
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
          "id": "14cf3ae2e055cd240306b3f4bf84d8d9c350bebc",
          "message": "Merge pull request #31 from ImperialCollegeLondon/25-make-results-callable\n\nMake result objects callable",
          "timestamp": "2024-05-26T12:53:00+01:00",
          "tree_id": "fa37df9530ac01a92628a490ad28fcf6e94e9e32",
          "url": "https://github.com/ImperialCollegeLondon/PyProBE/commit/14cf3ae2e055cd240306b3f4bf84d8d9c350bebc"
        },
        "date": 1716724568508,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/cyclers/test_biologic.py::test_read_and_process",
            "value": 182.86102375796654,
            "unit": "iter/sec",
            "range": "stddev: 0.0004725208386118459",
            "extra": "mean: 5.4686339354831155 msec\nrounds: 155"
          },
          {
            "name": "tests/cyclers/test_neware.py::test_read_and_process",
            "value": 0.19230697606035996,
            "unit": "iter/sec",
            "range": "stddev: 0.022112695479638896",
            "extra": "mean: 5.2000193674 sec\nrounds: 5"
          },
          {
            "name": "tests/test_cell.py::test_add_procedure",
            "value": 504.8735790823153,
            "unit": "iter/sec",
            "range": "stddev: 0.00014986915476627324",
            "extra": "mean: 1.9806938636354323 msec\nrounds: 418"
          },
          {
            "name": "tests/test_filter.py::test_step",
            "value": 10.336365838580253,
            "unit": "iter/sec",
            "range": "stddev: 0.005225695737503513",
            "extra": "mean: 96.74580172728818 msec\nrounds: 11"
          },
          {
            "name": "tests/test_filter.py::test_multi_step",
            "value": 10.37399055495931,
            "unit": "iter/sec",
            "range": "stddev: 0.0038455893221927494",
            "extra": "mean: 96.39492099999529 msec\nrounds: 10"
          },
          {
            "name": "tests/test_filter.py::test_charge",
            "value": 10.511943769418943,
            "unit": "iter/sec",
            "range": "stddev: 0.0036371422955658135",
            "extra": "mean: 95.12988481817915 msec\nrounds: 11"
          },
          {
            "name": "tests/test_filter.py::test_discharge",
            "value": 10.382787758028579,
            "unit": "iter/sec",
            "range": "stddev: 0.004394745920276508",
            "extra": "mean: 96.31324681820078 msec\nrounds: 11"
          },
          {
            "name": "tests/test_filter.py::test_chargeordischarge",
            "value": 9.941030848811923,
            "unit": "iter/sec",
            "range": "stddev: 0.0034211273225577556",
            "extra": "mean: 100.5931895000117 msec\nrounds: 10"
          },
          {
            "name": "tests/test_filter.py::test_rest",
            "value": 10.710876905029416,
            "unit": "iter/sec",
            "range": "stddev: 0.00353539248872688",
            "extra": "mean: 93.36303730000282 msec\nrounds: 10"
          },
          {
            "name": "tests/test_filter.py::test_negative_cycle_index",
            "value": 10.78663836238309,
            "unit": "iter/sec",
            "range": "stddev: 0.002302401530908723",
            "extra": "mean: 92.70728900000594 msec\nrounds: 11"
          },
          {
            "name": "tests/test_filter.py::test_negative_step_index",
            "value": 10.066135682767829,
            "unit": "iter/sec",
            "range": "stddev: 0.0034469623070816436",
            "extra": "mean: 99.34298836363743 msec\nrounds: 11"
          },
          {
            "name": "tests/test_filter.py::test_cycle",
            "value": 10.6035183977979,
            "unit": "iter/sec",
            "range": "stddev: 0.004559404223528857",
            "extra": "mean: 94.30831941666422 msec\nrounds: 12"
          },
          {
            "name": "tests/test_filter.py::test_all_steps",
            "value": 9.689242475381121,
            "unit": "iter/sec",
            "range": "stddev: 0.005080740904116822",
            "extra": "mean: 103.20724272726652 msec\nrounds: 11"
          },
          {
            "name": "tests/test_procedure.py::test_experiment",
            "value": 20320.36262934205,
            "unit": "iter/sec",
            "range": "stddev: 0.000020752345895478486",
            "extra": "mean: 49.2117201961754 usec\nrounds: 4882"
          },
          {
            "name": "tests/test_procedure.py::test_process_readme",
            "value": 548.8865132258064,
            "unit": "iter/sec",
            "range": "stddev: 0.00011683804888641206",
            "extra": "mean: 1.8218702334713952 msec\nrounds: 484"
          },
          {
            "name": "tests/test_rawdata.py::test_set_SOC",
            "value": 1555.4756025167737,
            "unit": "iter/sec",
            "range": "stddev: 0.00002616504370522655",
            "extra": "mean: 642.8901863725737 usec\nrounds: 499"
          }
        ]
      },
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
          "id": "be2897b3cef1147e73e30842d6f21fd2521fa962",
          "message": "Merge pull request #33 from ImperialCollegeLondon/32-remove-cycle-dependence\n\nRemove dependence on cycle column in cycler data, add event column to PyProBE format, remove get_events method",
          "timestamp": "2024-05-28T15:08:30+01:00",
          "tree_id": "96c21e8559d3b126e686c674e71cecc9fbd00404",
          "url": "https://github.com/ImperialCollegeLondon/PyProBE/commit/be2897b3cef1147e73e30842d6f21fd2521fa962"
        },
        "date": 1716905501959,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/cyclers/test_biologic.py::test_read_and_process",
            "value": 192.6287667007818,
            "unit": "iter/sec",
            "range": "stddev: 0.00009540242952276275",
            "extra": "mean: 5.191332619355557 msec\nrounds: 155"
          },
          {
            "name": "tests/cyclers/test_neware.py::test_read_and_process",
            "value": 0.1905140072453521,
            "unit": "iter/sec",
            "range": "stddev: 0.01484717773139039",
            "extra": "mean: 5.248957882200005 sec\nrounds: 5"
          },
          {
            "name": "tests/test_cell.py::test_add_procedure",
            "value": 510.22294662701125,
            "unit": "iter/sec",
            "range": "stddev: 0.00009107487093920842",
            "extra": "mean: 1.9599275309172461 msec\nrounds: 469"
          },
          {
            "name": "tests/test_filter.py::test_step",
            "value": 12.641682936972249,
            "unit": "iter/sec",
            "range": "stddev: 0.002862244083032859",
            "extra": "mean: 79.10339192856749 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_multi_step",
            "value": 12.593109150449585,
            "unit": "iter/sec",
            "range": "stddev: 0.0022066995115828303",
            "extra": "mean: 79.40850730768891 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_charge",
            "value": 12.617259851075863,
            "unit": "iter/sec",
            "range": "stddev: 0.0021124190398801113",
            "extra": "mean: 79.25651146153821 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_discharge",
            "value": 12.406033126435535,
            "unit": "iter/sec",
            "range": "stddev: 0.0018021672489642352",
            "extra": "mean: 80.60594307693236 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_chargeordischarge",
            "value": 12.229438488888785,
            "unit": "iter/sec",
            "range": "stddev: 0.001562862527110604",
            "extra": "mean: 81.7699030833315 msec\nrounds: 12"
          },
          {
            "name": "tests/test_filter.py::test_rest",
            "value": 13.146979762947502,
            "unit": "iter/sec",
            "range": "stddev: 0.0021635261944051008",
            "extra": "mean: 76.06309723076684 msec\nrounds: 13"
          },
          {
            "name": "tests/test_filter.py::test_negative_cycle_index",
            "value": 13.301266639720076,
            "unit": "iter/sec",
            "range": "stddev: 0.0016561346798373792",
            "extra": "mean: 75.18080999998996 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_negative_step_index",
            "value": 13.320536547594829,
            "unit": "iter/sec",
            "range": "stddev: 0.0018988368979221055",
            "extra": "mean: 75.07205107143834 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_cycle",
            "value": 13.569122935830075,
            "unit": "iter/sec",
            "range": "stddev: 0.0017924573853036042",
            "extra": "mean: 73.6967307857047 msec\nrounds: 14"
          },
          {
            "name": "tests/test_filter.py::test_all_steps",
            "value": 13.402701786320069,
            "unit": "iter/sec",
            "range": "stddev: 0.001644117977463408",
            "extra": "mean: 74.61182200000037 msec\nrounds: 14"
          },
          {
            "name": "tests/test_procedure.py::test_experiment",
            "value": 20233.01430367735,
            "unit": "iter/sec",
            "range": "stddev: 0.000005416442151789242",
            "extra": "mean: 49.42417303675063 usec\nrounds: 4814"
          },
          {
            "name": "tests/test_procedure.py::test_process_readme",
            "value": 534.3485779319878,
            "unit": "iter/sec",
            "range": "stddev: 0.00006395206577164647",
            "extra": "mean: 1.871437562106286 msec\nrounds: 475"
          },
          {
            "name": "tests/test_rawdata.py::test_set_SOC",
            "value": 1576.7985925445498,
            "unit": "iter/sec",
            "range": "stddev: 0.00013202346456985364",
            "extra": "mean: 634.1964057605198 usec\nrounds: 764"
          }
        ]
      }
    ]
  }
}