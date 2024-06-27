[![GitHub release; latest by date](https://img.shields.io/github/v/release/SETI/rms-vax)](https://github.com/SETI/rms-vax/releases)
[![GitHub Release Date](https://img.shields.io/github/release-date/SETI/rms-vax)](https://github.com/SETI/rms-vax/releases)
[![Test Status](https://img.shields.io/github/actions/workflow/status/SETI/rms-vax/run-tests.yml?branch=main)](https://github.com/SETI/rms-vax/actions)
[![Documentation Status](https://readthedocs.org/projects/rms-vax/badge/?version=latest)](https://rms-vax.readthedocs.io/en/latest/?badge=latest)
[![Code coverage](https://img.shields.io/codecov/c/github/SETI/rms-vax/main?logo=codecov)](https://codecov.io/gh/SETI/rms-vax)
<br />
[![PyPI - Version](https://img.shields.io/pypi/v/rms-vax)](https://pypi.org/project/rms-vax)
[![PyPI - Format](https://img.shields.io/pypi/format/rms-vax)](https://pypi.org/project/rms-vax)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/rms-vax)](https://pypi.org/project/rms-vax)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/rms-vax)](https://pypi.org/project/rms-vax)
<br />
[![GitHub commits since latest release](https://img.shields.io/github/commits-since/SETI/rms-vax/latest)](https://github.com/SETI/rms-vax/commits/main/)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/SETI/rms-vax)](https://github.com/SETI/rms-vax/commits/main/)
[![GitHub last commit](https://img.shields.io/github/last-commit/SETI/rms-vax)](https://github.com/SETI/rms-vax/commits/main/)
<br />
[![Number of GitHub open issues](https://img.shields.io/github/issues-raw/SETI/rms-vax)](https://github.com/SETI/rms-vax/issues)
[![Number of GitHub closed issues](https://img.shields.io/github/issues-closed-raw/SETI/rms-vax)](https://github.com/SETI/rms-vax/issues)
[![Number of GitHub open pull requests](https://img.shields.io/github/issues-pr-raw/SETI/rms-vax)](https://github.com/SETI/rms-vax/pulls)
[![Number of GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed-raw/SETI/rms-vax)](https://github.com/SETI/rms-vax/pulls)
<br />
![GitHub License](https://img.shields.io/github/license/SETI/rms-vax)
[![Number of GitHub stars](https://img.shields.io/github/stars/SETI/rms-vax)](https://github.com/SETI/rms-vax/stargazers)
![GitHub forks](https://img.shields.io/github/forks/SETI/rms-vax)

# Introduction

`vax` is a set of routines for converting between NumPy floating point and complex
scalars/arrays and VAX-format single- and double-precision floats.

`vax` is a product of the [PDS Ring-Moon Systems Node](https://pds-rings.seti.org).

# Installation

The `vax` module is available via the `rms-vax` package on PyPI and can be installed with:

```sh
pip install rms-vax
```

# Getting Started

The `vax` module provides two functions for converting *from* VAX-format floats:

- [`from_vax32`](https://rms-vax.readthedocs.io/en/latest/module.html#vax.from_vax32):
  Interpret a series of bytes or NumPy array as one or more VAX single-precision floats
  and convert them to a NumPy float or complex scalar or array.
- [`from_vax64`](https://rms-vax.readthedocs.io/en/latest/module.html#vax.from_vax64):
  Interpret a series of bytes NumPy array as one or more VAX double-precision floats and
  convert them to a NumPy float or complex scalar or array.

and two functions for converting *to* VAX-format floats::

- [`to_vax32`](https://rms-vax.readthedocs.io/en/latest/module.html#vax.to_vax32):
  Convert a NumPy float or complex scalar or array to a NumPy array containing the
  binary representation of VAX single-precision floats. Such an array can not be
  used for arithmetic operations since it is not in IEEE 754 format.
- [`to_vax32_bytes`](https://rms-vax.readthedocs.io/en/latest/module.html#vax.to_vax32_bytes):
  Convert a NumPy float or complex scalar or array to a Python `bytes` object containing
  the binary representation of VAX single-precision floats.

Note that there are no functions to convert a NumPy array to VAX double-precision format.

Details of each function are available in the [module documentation](https://rms-vax.readthedocs.io/en/latest/module.html).

Basic operation is as follows:

```python
import vax
b = vax.to_vax32([1., 2., 3.])
print(f'b = {b!r}')
ba = vax.to_vax32_bytes([1., 2., 3.])
print(f'ba = {ba!r}')
v = vax.from_vax32(b)
print(f'v = {v!r}')
va = vax.from_vax32(ba)
print(f'va = {va!r}')
```

yields:

```python
b = array([2.3138e-41, 2.3318e-41, 2.3407e-41], dtype=float32)
ba = b'\x80@\x00\x00\x00A\x00\x00@A\x00\x00'
v = array([1., 2., 3.], dtype=float32)
va = array([1., 2., 3.], dtype=float32)
```

As NASA data products stored as VAX-format floats are often provided in JPL's VICAR file
format, you may also be interested in the `rms-vicar` package
([documentation](https://rms-vicar.readthedocs.io/en/latest)).

# Contributing

Information on contributing to this package can be found in the
[Contributing Guide](https://github.com/SETI/rms-vax/blob/main/CONTRIBUTING.md).

# Links

- [Documentation](https://rms-vax.readthedocs.io)
- [Repository](https://github.com/SETI/rms-vax)
- [Issue tracker](https://github.com/SETI/rms-vax/issues)
- [PyPi](https://pypi.org/project/rms-vax)

# Licensing

This code is licensed under the [Apache License v2.0](https://github.com/SETI/rms-vax/blob/main/LICENSE).
