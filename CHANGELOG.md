# CHANGELOG


## v1.2.0-rc.1 (2025-01-02)

### Bug Fixes

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
