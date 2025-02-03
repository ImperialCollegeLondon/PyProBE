# CHANGELOG


## v2.0.0 (2025-02-03)

### Bug Fixes

- Allow units module to deal with percentage symbols
  ([`14f6176`](https://github.com/ImperialCollegeLondon/PyProBE/commit/14f6176ceeb52d1b058fd62bcdc8ccbbc3f8088d))

- Move critical plot functionality for dashboard into dashboard script
  ([`ff88c3d`](https://github.com/ImperialCollegeLondon/PyProBE/commit/ff88c3d922e9524178162244724753c618baffe6))

- Remove blank title from plot
  ([`df3ff12`](https://github.com/ImperialCollegeLondon/PyProBE/commit/df3ff123a8f4e8b46baf1cbe716a97c9f4600c36))

### Chores

- Add pybamm installation to ci workflow
  ([`f8f21cc`](https://github.com/ImperialCollegeLondon/PyProBE/commit/f8f21cca89d3e7c8c17659384aafcf3b3225a9bc))

- Release candidate 2.0.0
  ([`0918db3`](https://github.com/ImperialCollegeLondon/PyProBE/commit/0918db38cb297273f34baee22a435e042f96ee31))

- Run pytest with uv
  ([`db65402`](https://github.com/ImperialCollegeLondon/PyProBE/commit/db65402bdef007d52cb36a84bced4f7b4927967a))

- **ci**: Update ci workflow to install pybamm before running tests
  ([`eaf42e1`](https://github.com/ImperialCollegeLondon/PyProBE/commit/eaf42e1c35241ee9e925449a0698b3dd03889f03))

### Refactoring

- Remove deprecated analysis classes
  ([`600c04a`](https://github.com/ImperialCollegeLondon/PyProBE/commit/600c04a41fc49acf09d986f2de209a9160e99dae))

BREAKING CHANGE: class-based structure for the analysis module has been replaced with single
  functions within the same module

- Remove deprecated Plot class
  ([`b03ab52`](https://github.com/ImperialCollegeLondon/PyProBE/commit/b03ab520ebeacf9be220729188ad6c813c313cf9))

BREAKING CHANGE: removal of deprecated class due to maintenance overhead. Use Result.plot(),
  Result.hvplot() or seaborn wrapper instead

- Reorganise dashboard into class and add tests for full coverage
  ([`a06ec46`](https://github.com/ImperialCollegeLondon/PyProBE/commit/a06ec4650dcf795c90741c87806cf6ed9f907ec0))

- Reorganise data processing methods
  ([`63a5bd7`](https://github.com/ImperialCollegeLondon/PyProBE/commit/63a5bd7c1fc13c0eedd62bee02ae7d1e71af4e75))

- Split dashboard into functions and add tests
  ([`bf38f5f`](https://github.com/ImperialCollegeLondon/PyProBE/commit/bf38f5fc49b4d1332f6c53ebf67bc98a2311dd84))


## v1.4.0 (2025-01-31)

### Chores

- Add a .zenodo.json file
  ([`29be1c1`](https://github.com/ImperialCollegeLondon/PyProBE/commit/29be1c1e275561eebebbfcddb4f144ee9fa45cf6))

- Add isort setting to ruff
  ([`24408a1`](https://github.com/ImperialCollegeLondon/PyProBE/commit/24408a133e4d2265f703d7d335ace9aed3ddba38))

- Add JOSS status badge
  ([`43a94a9`](https://github.com/ImperialCollegeLondon/PyProBE/commit/43a94a9b5db654e90424105ab0e11e623f0a17bd))

- Add path filters for sphinx workflow
  ([`480dcca`](https://github.com/ImperialCollegeLondon/PyProBE/commit/480dccaf4e6080f0356275520498d322460911be))

- Add pre-commit.ci badge
  ([`247e5bc`](https://github.com/ImperialCollegeLondon/PyProBE/commit/247e5bce085899dfdb3c7db7de9e8be95053bc02))

- Add status badges to readme
  ([`87de879`](https://github.com/ImperialCollegeLondon/PyProBE/commit/87de879e1b330b3c55dbaaa35e8e1a9c58bb5b54))

- Release candidate 1.4.0
  ([`a1e1a22`](https://github.com/ImperialCollegeLondon/PyProBE/commit/a1e1a22ad3b30ccaaadf9de70f4e58f03723be72))

- Release candidate 1.4.0
  ([`0d3181d`](https://github.com/ImperialCollegeLondon/PyProBE/commit/0d3181d2ecda0c41cc6568748e0682026d557ad4))

- Release candidate 1.4.0
  ([`e62a02d`](https://github.com/ImperialCollegeLondon/PyProBE/commit/e62a02d2554a7b6c3fb6412ffedd6f504bb87bc1))

- Replace pre-commit action with pre-commit.ci
  ([`d360ec4`](https://github.com/ImperialCollegeLondon/PyProBE/commit/d360ec45aa98d5b11a75b8ed65fa8b359b484c61))

- Specify paths for ci workflow
  ([`24f7119`](https://github.com/ImperialCollegeLondon/PyProBE/commit/24f7119f7e23ed6eac0da5af6e7a7af5fcc82d2c))

### Features

- Add capability to export any Result object to a .mat file
  ([`d5b94a6`](https://github.com/ImperialCollegeLondon/PyProBE/commit/d5b94a651aa1ce1352dabc5eccd733ed53428966))


## v1.3.2 (2025-01-22)

### Bug Fixes

- Fix uv version in pre-commit and workflows
  ([`246d7b1`](https://github.com/ImperialCollegeLondon/PyProBE/commit/246d7b1e46317d0b153158f515b4128d5c4395a5))

### Chores

- Release candidate 1.3.2
  ([`7e77220`](https://github.com/ImperialCollegeLondon/PyProBE/commit/7e77220ada092584510746b40cb225754389f4db))


## v1.3.1 (2025-01-22)

### Bug Fixes

- Concat biologic MB files diagonally to prevent schema conflict errors
  ([`95e97fc`](https://github.com/ImperialCollegeLondon/PyProBE/commit/95e97fc6341669b172e5f5eab09c4757283a5cb1))

- Paper figure typo
  ([`8cfe8c1`](https://github.com/ImperialCollegeLondon/PyProBE/commit/8cfe8c1cd4f8230dfc4ede316a1d24eeb1daf3d2))

### Chores

- Add --frozen flag to uv sync commands in workflows
  ([`b158397`](https://github.com/ImperialCollegeLondon/PyProBE/commit/b1583979908bac0a8fa9b92444c9e82e34d69545))

- Release candidate 1.3.1 [skip ci]
  ([`7764404`](https://github.com/ImperialCollegeLondon/PyProBE/commit/7764404aaf6fb0253d6a3fafd25917c51859b913))

- Replace warnings with logger for missing columns in data
  ([`dbd16c9`](https://github.com/ImperialCollegeLondon/PyProBE/commit/dbd16c9eeb85e9620f47e43f061e5cd32c100176))

- Unfreeze uv sync for release candidate workflow
  ([`3692bc5`](https://github.com/ImperialCollegeLondon/PyProBE/commit/3692bc535b20bfc82b6260e544da0d5e743944af))


## v1.3.0 (2025-01-12)

### Bug Fixes

- **cell**: Correct incompatible data/lazyframes in pybamm_experiment property
  ([`3a2ca00`](https://github.com/ImperialCollegeLondon/PyProBE/commit/3a2ca001fd6dc425ff01a5f18efbb0e18fbd319d))

### Chores

- Add --seed creation of uv venv during workflows
  ([`45a5fbe`](https://github.com/ImperialCollegeLondon/PyProBE/commit/45a5fbea8ac4dc3d978ac12164231e2ab5968728))

- Add codecov coverage upload
  ([`8c7a8ce`](https://github.com/ImperialCollegeLondon/PyProBE/commit/8c7a8cef59e5edf4a0c407d788e71087dba4dbec))

- Add condition to release candidate workflow
  ([`80cf916`](https://github.com/ImperialCollegeLondon/PyProBE/commit/80cf9164c924c9ac72f25dee61540186b2a710ad))

Prevents running the workflow when a merge is made from a branch created by the release candidate
  workflow itself

- Add readme info to pyproject.toml
  ([`0e4a25f`](https://github.com/ImperialCollegeLondon/PyProBE/commit/0e4a25f16144bb10be9a1d2ac3c57ad6205136d8))

- Add two step coverage upload using artifacts
  ([`b17ad0c`](https://github.com/ImperialCollegeLondon/PyProBE/commit/b17ad0ccf5a1ce8e02a60bf2ab6d89b0f97e0fb8))

- Add urls and classifiers to pyproject.toml
  ([`ef08ea5`](https://github.com/ImperialCollegeLondon/PyProBE/commit/ef08ea57ceeec7cd8d8c15136199f33a8143020d))

- Fix coverage workflow
  ([`3ff3028`](https://github.com/ImperialCollegeLondon/PyProBE/commit/3ff30281e2d047fac72280006c7be7cc5bb46845))

- Fix ruff linting errors
  ([`e32a067`](https://github.com/ImperialCollegeLondon/PyProBE/commit/e32a067dd96666e25cc39bb69e0acb1be1740d32))

- Ignore D103 (docstring in public function) in examples
  ([`f2a79c2`](https://github.com/ImperialCollegeLondon/PyProBE/commit/f2a79c2f17a661ff3c1f6fc684bd07edb03d823a))

- Ignore performance example from pytest on notebook examples
  ([`62adee0`](https://github.com/ImperialCollegeLondon/PyProBE/commit/62adee0a2106fae4ec6ed58ec25aa299139f0f26))

- Make ruff a dev dependency
  ([`2ffc0a2`](https://github.com/ImperialCollegeLondon/PyProBE/commit/2ffc0a2645e90142f8e91fda57d668fdfcb8b008))

- Move dev dependencies into optional dependency group
  ([`2202c22`](https://github.com/ImperialCollegeLondon/PyProBE/commit/2202c2239b7558d3b2d33ba52e706ddf9270b064))

Dependency groups are currently not supported for backwards compatibility with pip. Wait for
  resolution of PEP 735

- Recategorise xlsxwriter as dev dependency
  ([`699a850`](https://github.com/ImperialCollegeLondon/PyProBE/commit/699a8506cd5b58399885d3cfd7c363becb9b7615))

- Release candidate 1.3.0 [skip ci]
  ([`f0384a3`](https://github.com/ImperialCollegeLondon/PyProBE/commit/f0384a390aa81e25c970f14c9cbae51412175817))

- Release candidate 1.3.0 [skip ci]
  ([`d14ffce`](https://github.com/ImperialCollegeLondon/PyProBE/commit/d14ffce0e621b62dd301a1af663fb0cecc4bd7cf))

- Remove lean differentiation example
  ([`1a07ae9`](https://github.com/ImperialCollegeLondon/PyProBE/commit/1a07ae99f2f3c3d81cf9e75262d299462ab53b60))

- Remove ordered-set and distinctipy dependencies
  ([`314f0e9`](https://github.com/ImperialCollegeLondon/PyProBE/commit/314f0e9c666df8eb732324b970bc3d70fbfbf796))

- Run ci on push to main, exclude notebook tests on push
  ([`43e30f8`](https://github.com/ImperialCollegeLondon/PyProBE/commit/43e30f84b06fe0610d788fe5e49447f024006289))

- Update nbstripout precommit hook to remove kernelspec
  ([`ea8e10b`](https://github.com/ImperialCollegeLondon/PyProBE/commit/ea8e10be48fd2b29d8ed033528fa08236d9d05c1))

- Update workflow to v7 of create-pull-request gh action [skip ci]
  ([`8b3e2d3`](https://github.com/ImperialCollegeLondon/PyProBE/commit/8b3e2d386615c80b59b29fb5b88f115357b06222))

- Update workflows for dependency reorganisation
  ([`a1a76d2`](https://github.com/ImperialCollegeLondon/PyProBE/commit/a1a76d2700d4cf14d676360b96b316fc47eed269))

- Use % magic to install pybamm and matplotlib in example notebooks
  ([`62154c3`](https://github.com/ImperialCollegeLondon/PyProBE/commit/62154c3229d6bca3bba5207f1b16ab6793598c96))

### Code Style

- Run ruff format on examples and docs config files
  ([`adb57cc`](https://github.com/ImperialCollegeLondon/PyProBE/commit/adb57ccd78cbbbeb0ec0eff3c2f7c7174e32aaf4))

- Run ruff format on pyprobe/ and tests/
  ([`e8a267c`](https://github.com/ImperialCollegeLondon/PyProBE/commit/e8a267c791dc2961135fad395172ab1484c93fb5))

### Documentation

- Add citations to pybamm example
  ([`0af6632`](https://github.com/ImperialCollegeLondon/PyProBE/commit/0af66320dfc67f5348427cd2aee1c94e720ae740))

- Add detail for creating a jupyter kernel from uv .venv
  ([`35b33bd`](https://github.com/ImperialCollegeLondon/PyProBE/commit/35b33bd5b647ff40b95d39b1ecf82c4e4b265cfc))

- Add inline matplotlib magic to show plots in examples
  ([`47ee154`](https://github.com/ImperialCollegeLondon/PyProBE/commit/47ee1542e12050d0b908483e6e2f93658a477d04))

- Add matplotlib inline to pybamm example
  ([`b43a809`](https://github.com/ImperialCollegeLondon/PyProBE/commit/b43a80986429ba286d9fc26cc8fc3958a4050286))

- Add mention of pybamm integration to paper
  ([`e8549d9`](https://github.com/ImperialCollegeLondon/PyProBE/commit/e8549d98307c00295f00fc5b4de69f5802871383))

- Add optional dependency detail for hvplot and seaborn
  ([`0045fd4`](https://github.com/ImperialCollegeLondon/PyProBE/commit/0045fd4ebec97d23da542e6d3277d134a1f5a666))

- Add plotting example to examples toc
  ([`f8e2026`](https://github.com/ImperialCollegeLondon/PyProBE/commit/f8e20267dd2b013de197df21188c949fff53857f))

- Add pybamm example to docs index
  ([`a5c08f6`](https://github.com/ImperialCollegeLondon/PyProBE/commit/a5c08f68e68b96a77a29a0b6fa9065ba323515ac))

- Add some missing citations and describe plotting integrations
  ([`cd4b853`](https://github.com/ImperialCollegeLondon/PyProBE/commit/cd4b8533d4b025bd46d93d6d61bc1af29b05bec7))

- Capture output for installing packages in notebook
  ([`7f870c3`](https://github.com/ImperialCollegeLondon/PyProBE/commit/7f870c35fa5ca28a8b46ac82ed59df86abc5a215))

- Create pybamm integration example
  ([`7ad73e3`](https://github.com/ImperialCollegeLondon/PyProBE/commit/7ad73e3038639b55432356661014d77d49f1fffe))

- Fix errors and typos in documentation
  ([`458e4c6`](https://github.com/ImperialCollegeLondon/PyProBE/commit/458e4c6f86bed1d4b38a4bdd495047b97c80fa51))

- Fix mistake with comparing parquet read times not overwriting exiting files
  ([`28c0526`](https://github.com/ImperialCollegeLondon/PyProBE/commit/28c0526d8dc17e22ec8129febe85b55e78f664e7))

- Fix performance example for different parquet settings
  ([`7ba7053`](https://github.com/ImperialCollegeLondon/PyProBE/commit/7ba7053e4c8ed249b3face9e320fb96895f26565))

- Fix syntax highlighting in examples
  ([`9a724c7`](https://github.com/ImperialCollegeLondon/PyProBE/commit/9a724c72369cd7b5d7d838912922aa8733e29a8e))

- Minor typo fix to paper
  ([`5017aec`](https://github.com/ImperialCollegeLondon/PyProBE/commit/5017aec38f2e4ffddd0200339fe0bc874d0b2445))

- Remove kernelspec from example notebooks
  ([`6e278f6`](https://github.com/ImperialCollegeLondon/PyProBE/commit/6e278f6e71c4fd057c9ecbdc0bdcb62e60f10829))

- Remove mention of the requirement to specify a "Name" for a cell
  ([`cc1313f`](https://github.com/ImperialCollegeLondon/PyProBE/commit/cc1313ff4c0007d20d69473a585cd924dcc84a35))

- Update dev install instructions
  ([`f3740bf`](https://github.com/ImperialCollegeLondon/PyProBE/commit/f3740bfee14ecddcf92e08491010c7007b8eccf5))

- Update readme and user manual to reflect plotting integrations
  ([`553350f`](https://github.com/ImperialCollegeLondon/PyProBE/commit/553350f85934d31c952964e6e93e9e5b5a06b280))

### Features

- Add a selectbox for a cell identifier to replace the "Name" field
  ([`5657285`](https://github.com/ImperialCollegeLondon/PyProBE/commit/5657285a35e5e06fbd47fa0ea2892eff8de160cc))

- Allow any dict values in info dictionary
  ([`c610126`](https://github.com/ImperialCollegeLondon/PyProBE/commit/c610126351ceae10446e3b9d82caf3614c8f60f5))

### Refactoring

- Move cell identifier to below plot
  ([`6a5e1c4`](https://github.com/ImperialCollegeLondon/PyProBE/commit/6a5e1c45e0fd8b30821c7b04968dafcacf240af7))

- Remove colour generation in make_cell_list method
  ([`d7704cc`](https://github.com/ImperialCollegeLondon/PyProBE/commit/d7704cc825a3004c25461ebd5438fdc0d339d62d))

- Remove OrderedSet use in dashboard
  ([`26b38e1`](https://github.com/ImperialCollegeLondon/PyProBE/commit/26b38e1f720521bef0e425b4c2b7b76267afa404))

- Remove search for color entry in info dict
  ([`9894e58`](https://github.com/ImperialCollegeLondon/PyProBE/commit/9894e5864af0615e3e71ee5429ceccc22e847abb))

This changes the default behaviour to cycle through the colours built-in to plotly

- Remove setting of a default name and colour assignment in cell
  ([`9219eaf`](https://github.com/ImperialCollegeLondon/PyProBE/commit/9219eaf00ffba786f04422457c4c408667ff982b))

- Replace black, flake8 and isort config with ruff
  ([`8b7c508`](https://github.com/ImperialCollegeLondon/PyProBE/commit/8b7c5088ca646562725d180c937a1abaa833ba5a))

### Testing

- Don't check column order in add_procedure test
  ([`5e0ce2d`](https://github.com/ImperialCollegeLondon/PyProBE/commit/5e0ce2dc1c137ab1dab3a24a8c3321970e6d48e1))

- Fix Plot class tests
  ([`f9df697`](https://github.com/ImperialCollegeLondon/PyProBE/commit/f9df697428fafd44e7809a0005fe181e61733b81))

- Move pybamm out of dev dependencies
  ([`fc720d1`](https://github.com/ImperialCollegeLondon/PyProBE/commit/fc720d13692096d3f08e74eb8d97b0c9175edbdd))

Use pytest.importorskip for skipping pybamm tests if it is not installed

- Skip seaborn tests if it is not installed
  ([`ed8b04b`](https://github.com/ImperialCollegeLondon/PyProBE/commit/ed8b04b72729bd7c5272f4863748176ac44bed72))


## v1.2.0 (2025-01-02)

### Bug Fixes

- Disallow minor prerelease from main branch
  ([`6b595f6`](https://github.com/ImperialCollegeLondon/PyProBE/commit/6b595f6a97d725ca1fcf381e97af03fb16749fb3))

- Error from rebase
  ([`62a0e5f`](https://github.com/ImperialCollegeLondon/PyProBE/commit/62a0e5fe06f49576223c0b088d51d372c47ea21b))

- Issue with missing experiment tuple on dashboard
  ([`21d22c8`](https://github.com/ImperialCollegeLondon/PyProBE/commit/21d22c84af595df621501cba177a590bf33414ee))

- Pass label argument to plot through as string
  ([`c8fdcc5`](https://github.com/ImperialCollegeLondon/PyProBE/commit/c8fdcc5d1d0dea6f3b3321a7952b5ee518d6683e))

Prevents collection of columns if label argument happens to match a column name

- **result**: Bug in combine_results method
  ([`f1c40a0`](https://github.com/ImperialCollegeLondon/PyProBE/commit/f1c40a0780245c34b1abadf6f9c993309d85271a))

### Chores

- Add hvplot and seaborn as optional dependencies
  ([`217598c`](https://github.com/ImperialCollegeLondon/PyProBE/commit/217598c7a809948adaf8beab4ea321479afaea4a))

- Add ipykernel dependency
  ([`ae27260`](https://github.com/ImperialCollegeLondon/PyProBE/commit/ae272601db1bcfd5f6209f5a3ac384eeff48a364))

- Add nbmake as developer requirement
  ([`82d6867`](https://github.com/ImperialCollegeLondon/PyProBE/commit/82d68678bdeb1077e035b897077a1ff6cccd23f6))

- Add python-semantic-release as dependency
  ([`9458f82`](https://github.com/ImperialCollegeLondon/PyProBE/commit/9458f82274ddb03481fa8325fc84a753163b3574))

- Add setup for semantic release
  ([`0523185`](https://github.com/ImperialCollegeLondon/PyProBE/commit/0523185db1bdcaf3d6027d2de8e1b1c990e8b9a7))

- Release candidate 1.2.0 [skip ci]
  ([`c29dddb`](https://github.com/ImperialCollegeLondon/PyProBE/commit/c29dddb5a621ec2f2a6166fbfbde0ddb39a1aec7))

- Remove pip-tools requirements files
  ([`5924f31`](https://github.com/ImperialCollegeLondon/PyProBE/commit/5924f311faa41540e13de85aa17df835f3331bc0))

- Remove show_image method and associated dependencies
  ([`b3011d5`](https://github.com/ImperialCollegeLondon/PyProBE/commit/b3011d570b3f54e34c1b591d5fecff4304492868))

kaleido and ipython removed as required dependencies. This is not a breaking change as the .show()
  method remains.

- Replace auto publishing with workflow to create an rc PR
  ([`57b9de6`](https://github.com/ImperialCollegeLondon/PyProBE/commit/57b9de60004474c7cb046ef83e97d5e4fa2ca564))

- Set up workflow for pre-release
  ([`be4aa88`](https://github.com/ImperialCollegeLondon/PyProBE/commit/be4aa88b2f4002b950a5b65684a3065bdb7e28c2))

- Use _version.py for version numbering
  ([`b621de0`](https://github.com/ImperialCollegeLondon/PyProBE/commit/b621de0f77ca0684456671b170a05d5bc94ae9aa))

### Documentation

- Add explanations for quick_add_procedure
  ([`0617e37`](https://github.com/ImperialCollegeLondon/PyProBE/commit/0617e37791f730f93aaa0e246cfc47170b0a81ca))

- Add matplotlib/pandas and hvplot to plotting example
  ([`6bc9649`](https://github.com/ImperialCollegeLondon/PyProBE/commit/6bc96494acf68d22e477fee5196606c799d4bd94))

- Remove show_image() from LEAN differentiation example
  ([`1867836`](https://github.com/ImperialCollegeLondon/PyProBE/commit/1867836eb36a88962d9e879e172f613d5778e0a7))

- Simplify developer installation instructions
  ([`860b663`](https://github.com/ImperialCollegeLondon/PyProBE/commit/860b663631f3f23fd993bdf32370df38a649d6c4))

Remove requirement to specify all groups independently

- Update contributing guidelines
  ([`fe47fc0`](https://github.com/ImperialCollegeLondon/PyProBE/commit/fe47fc05ed71f5c411449f793f07914db1922e0d))

- Update examples to use .plot() method
  ([`dd2cbf9`](https://github.com/ImperialCollegeLondon/PyProBE/commit/dd2cbf954eda03376eb2873a2822542e04413a10))

### Features

- Add a method for adding a procedure without a README file
  ([`4b9e80d`](https://github.com/ImperialCollegeLondon/PyProBE/commit/4b9e80dd39bf909e2741ec5fc27ed14c0b89804e))

- Add buffer to charge and discharge filters to exclude noise around zero current
  ([`2b1c62d`](https://github.com/ImperialCollegeLondon/PyProBE/commit/2b1c62ddfcc8476e858b8fe2457f8ba78c9ca963))

- Add capability to read biologic mpt files that have no header
  ([`0a3f82d`](https://github.com/ImperialCollegeLondon/PyProBE/commit/0a3f82d256b3b34d25341a0e862288792edd7011))

- Add full-word Seconds unit
  ([`70743b7`](https://github.com/ImperialCollegeLondon/PyProBE/commit/70743b7b357b846a1f061c33db8d3c6a4f3668fe))

- Add method to combine multiple result objects
  ([`744f496`](https://github.com/ImperialCollegeLondon/PyProBE/commit/744f4965e12589f15f5ddcd84f4293cf86ecfb06))

This will integrate their info dicts into the dataframe

- Add user control of the header row index and date format when importing a generic file
  ([`2866a91`](https://github.com/ImperialCollegeLondon/PyProBE/commit/2866a917cb5a2ba7982b08f677a014f7973e1640))

- Add utility methods for plot and hvplot to result objects
  ([`f0c1f10`](https://github.com/ImperialCollegeLondon/PyProBE/commit/f0c1f106243cc3c7610f1a9bfe6f4e5d9004ae85))

- Allow specification of header rows in experiment log file
  ([`28b7a48`](https://github.com/ImperialCollegeLondon/PyProBE/commit/28b7a48fea9153f877a07e11c4993a62ff94f3ef))

- Create seaborn wrapper
  ([`5b9eb18`](https://github.com/ImperialCollegeLondon/PyProBE/commit/5b9eb1825c1c4f1992db938617083c8a43fd1a76))

### Refactoring

- Add cache for collected columns of the base dataframe
  ([`1d7ae55`](https://github.com/ImperialCollegeLondon/PyProBE/commit/1d7ae55a3fd85a7058017244755e2e6fd4959b6c))

- Remove unit information from column definitions
  ([`cf5f627`](https://github.com/ImperialCollegeLondon/PyProBE/commit/cf5f6274baaa0a28d70826c7adfd80f37f4a7204))


## v1.1.4 (2024-12-31)


## v1.1.3 (2024-12-07)


## v1.1.2 (2024-12-07)


## v1.1.1 (2024-12-03)


## v1.1.0 (2024-11-29)

### Documentation

- **contributor**: Contrib-readme-action has updated readme
  ([`70da06d`](https://github.com/ImperialCollegeLondon/PyProBE/commit/70da06d55a8254b6b2a4a5deb67d950b68782f68))


## v1.0.3 (2024-10-04)


## v1.0.2 (2024-09-16)


## v1.0.1 (2024-09-16)


## v1.0.0 (2024-09-16)


## v0.1.4 (2024-06-18)


## v0.1.3 (2024-06-17)


## v0.1.2 (2024-06-11)


## v0.1.1 (2024-06-05)


## v0.1.0 (2024-06-02)

### Documentation

- **contributor**: Contrib-readme-action has updated readme
  ([`ba1a272`](https://github.com/ImperialCollegeLondon/PyProBE/commit/ba1a272a78893fd73aa18ee44630fd7356fefcef))
