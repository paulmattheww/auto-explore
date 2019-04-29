========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |requires|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/auto-explore/badge/?style=flat
    :target: https://readthedocs.org/projects/auto-explore
    :alt: Documentation Status

.. |requires| image:: https://requires.io/github/paulmattheww/auto-explore/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/paulmattheww/auto-explore/requirements/?branch=master

.. |codecov| image:: https://codecov.io/github/paulmattheww/auto-explore/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/paulmattheww/auto-explore

.. |version| image:: https://img.shields.io/pypi/v/auto-explore.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/auto-explore

.. |commits-since| image:: https://img.shields.io/github/commits-since/paulmattheww/auto-explore/v0.1.1.svg
    :alt: Commits since latest release
    :target: https://github.com/paulmattheww/auto-explore/compare/v0.1.1...master

.. |wheel| image:: https://img.shields.io/pypi/wheel/auto-explore.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/auto-explore

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/auto-explore.svg
    :alt: Supported versions
    :target: https://pypi.org/project/auto-explore

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/auto-explore.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/auto-explore


.. end-badges

Semi-automated exploratory data analysis

* Free software: MIT license

Installation
============

::

    pip install auto-explore

Documentation
=============


https://auto-explore.readthedocs.io/


Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
