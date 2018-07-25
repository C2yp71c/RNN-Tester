WEAKRADNOM
==========

The `python3` program `weakrandom.py` generates random number prepared
to analyze them with [`rnn_tester`](../rnn_tester.py).

## Build

Use `make` to build the dependencies.

## Usage

Use `python3 weakrandom.py [options]` to generate weak random numbers.

## Development

If you like to add a new PRNG extend `weakrandom.py` or create your new file.
All dependencies for a new PRNG are in `libweakrandom.py`.

### Testing

All unit tests could be executed with `python3 -m unittest`.
For static code analysis use `python3 -m mypy <file-name.py>` or run all tests with .
`python3 -m unittest discover -p "*_test.py"`.
