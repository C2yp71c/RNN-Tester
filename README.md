RNN Tester
==========
This is the implementation of the content of the paper
[Testing Cryptographically Secure Pseudo Random Number Generators with
Artificial Neural Networks](10.1109/TrustCom/BigDataSE.2018.00168).
The `python3` program `rnn_tester.py` evaluates cryptographically
secure pseudo random generators (CSPRNGs). The program uses a stream of
numbers from the CSRPNGs formatted
by [`weakrandom.py`](src/weakrandom/weakrandom.py).

## Build

No build instructions are necessary.

## Testing

Static analysis with `mypy src/rnn_tester.py` and
unit testing with `python3 src/rnn_tester_test.py`

## Usage

Use `python3 rnn_tester.py [options] -d <path to dir> -f <PRNG name>`
to test random numbers. In the directory `<path-to-dir>` all results from
the program `weakrandom.py` are stored. With `<PRNG name>` one of the
data sets could be chosen.

Weakrandom
=========

Look at [/src/weakrandom/](src/weakrandom/README.md).