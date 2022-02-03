RNN Tester
==========
This is the implementation base on the content of the paper\
[Testing Cryptographically Secure Pseudo Random Number Generators with
Artificial Neural Networks](<https://ieeexplore.ieee.org/document/8456037>).\
The `python3` program `rnn_tester` evaluates (cryptographically
secure) pseudo random generators with AI.

## Build

To build the project the python package build is required

    python3 -m pip install --upgrade build

To build a package use

    python3 -m build

## Install

The tool requires the programm
[dieharder](https://webhome.phy.duke.edu/~rgb/General/dieharder.php) 
for statistical analysis of the results. Install the tool first.

- Debian/Ubuntu:

    `sudo apt-get install dieharder`
    
- Fedora:

    `sudo dnf install dieharder`
    

Install the tool with 

    pip install ./dist/rnn_tester-<version>-py3-none-any.whl

## Usage

The tool `rnn_tester` requires a binary file containing the result of an PRNG in `<path-to-rng-file>`.
Use 

    python3 rnn_tester [options] -i <path-to-rng-file>
    
to test your random numbers.

## Testing

The folder [test](./tests) contains all unit test.
All tests including statistical code analysis could be execute with 
    
    tox -r

