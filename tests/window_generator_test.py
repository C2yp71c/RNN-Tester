#!/usr/bin/env python3

# RNN Tester - Testing cryptographically secure pseudo random generator.
# Copyright (C) 2022 Tilo Fischer

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import random
from pathlib import Path

import numpy as np

import rnn_tester.window_generator as wg

import tensorflow as tf


class TestClass:
    _input_size = 6
    _label_size = 1
    _batch_size = 1

    def _bytesarray2window(self, abytes):
        window_size = self._input_size + self._label_size
        # Vector converted to bit list
        test_vector_bit = np.unpackbits(abytes.ravel()).ravel()
        # Index creating a window_size long window sliding with self._label_size
        indexer = [list(range(i, i + window_size))
                   for i in range(0, len(test_vector_bit) - window_size, self._label_size)]
        return test_vector_bit[indexer]

    def test_window_gen_data(self, tmp_path):
        zero_file = tmp_path / "zero.bin"
        rep = 1000
        with open(zero_file, 'wb') as file:
            file.write(bytes([1] * rep))
        w = wg.WindowGenerator(24, 1, 0, 10, zero_file)
        assert all(w._data == [[0], [0], [0], [0], [0], [0], [0], [1]] * rep)

    def test_window_gen_real(self):
        values = Path("./data/weakrandom/data/urand.bin")
        w = wg.WindowGenerator(24, 1, 0, 10, values)
        assert w is not None

    def test_window_gen_me_zero(self, tmp_path):
        zero_file = tmp_path / "zero.bin"
        with open(zero_file, 'wb') as file:
            file.write(bytes([0] * 1000))
        w = wg.WindowGenerator(24, 1, 0, 10, zero_file)
        assert w.get_mean_error() == 0

    def test_window_gen_me_max(self, tmp_path):
        one_file = tmp_path / "one.bin"
        with open(one_file, 'wb') as file:
            file.write(bytes([255] * 1000))
        w = wg.WindowGenerator(24, 1, 0, 10, one_file)
        assert w.get_mean_error() == 1

    def test_split(self, tmp_path):
        one_file = tmp_path / "one.bin"
        test_vector = [[1], [2], [3], [4], [5], [6], [7]]
        with open(one_file, 'wb') as file:
            file.write(bytes([1] * 1000))
        w = wg.WindowGenerator(self._input_size, self._label_size, 0, self._batch_size, one_file)
        inputs, labels = w._split_window(tf.constant([test_vector] * self._batch_size))
        assert inputs.shape == [self._batch_size, self._input_size, 1]
        assert labels.shape == [self._batch_size, self._label_size, 1]
        for i, l in zip(inputs, labels):
            assert all(i == test_vector[:self._input_size])
            assert all(l == test_vector[self._input_size:])

    def test_dataset(self, tmp_path):
        one_file = tmp_path / "one.bin"

        test_vector = np.tile(np.array([[random.randint(0, 255)]
                                        for _ in range(10)], dtype=np.uint8), (self._batch_size, 1))
        bit_windows = self._bytesarray2window(test_vector)

        with open(one_file, 'wb') as file:
            file.write(bytes(test_vector.ravel()))
        w = wg.WindowGenerator(self._input_size, self._label_size, 0, self._batch_size, one_file)
        ds = list(w._make_dataset(0, len(test_vector) * 8 - 1).as_numpy_iterator())

        assert len(ds) == len(bit_windows)
        for d, v in zip(ds, bit_windows):
            assert len(d) == 2
            assert all((d[0].ravel() == v[:self._input_size]))
            assert all((d[1].ravel() == v[self._input_size:]))

    def test_all_ds(self, tmp_path):
        one_file = tmp_path / "one.bin"
        test_vector = np.tile(np.array([[random.randint(0, 255)] for _ in range(100)],
                                       dtype=np.uint8), (self._batch_size, 1))
        bit_windows = self._bytesarray2window(test_vector)

        with open(one_file, 'wb') as file:
            file.write(bytes(test_vector.ravel()))
        w = wg.WindowGenerator(self._input_size, self._label_size, 0, self._batch_size, one_file)

        train = list(list(w.train.as_numpy_iterator()))
        val = list(list(w.val.as_numpy_iterator()))
        test = list(w.test.as_numpy_iterator())

        assert len(train) < len(bit_windows)
        assert len(val) == len(bit_windows[len(train):len(train) + len(val)])
        assert len(test) == len(bit_windows[len(train) + len(val):])

        for d, v in zip(train + val + test, bit_windows):
            assert len(d) == 2
            assert all(d[0].ravel() == v[:self._input_size])
            assert all(d[1].ravel() == v[self._input_size:])
