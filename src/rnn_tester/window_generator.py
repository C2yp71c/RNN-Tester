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

import logging
from pathlib import Path

import numpy as np

import tensorflow as tf


class WindowGenerator():
    def __init__(self, input_width: int, label_width: int, shift: int, batch_size: int,
                 ptrain: Path):
        # Store the raw data
        self._batch_size = batch_size

        # Work out the window parameters.
        self._input_width = input_width
        self._label_width = label_width

        self._total_window_size = input_width + shift + label_width

        self._input_slice = slice(0, input_width)
        self._input_indices = np.arange(self._total_window_size)[self._input_slice]

        label_start = self._total_window_size - self._label_width
        self._labels_slice = slice(label_start, None)
        self._label_indices = np.arange(self._total_window_size)[self._labels_slice]

        if not ptrain.is_file():
            raise Exception(str(ptrain) + "is not a file")
        self._data = np.unpackbits(np.fromfile(ptrain, dtype=np.uint8)).reshape((-1, 1))

        # File should be larger then 250MB
        if(len(self._data) < 250000000 * 8):
            logging.warning("The file %s contains too few value dieharder "
                            "will fail to eval them later (min. 250MB)",
                            str(ptrain))

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self._total_window_size}',
            f'Input indices: {self._input_indices}',
            f'Label indices: {self._label_indices}'])

    def get_mean_error(self) -> float:
        return np.mean(self._data)

    def _split_window(self, features: tf.Tensor):
        inputs = features[:, self._input_slice, :]
        labels = features[:, self._labels_slice, :]

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self._input_width, None])
        labels.set_shape([None, self._label_width, None])
        return tf.cast(inputs, float), tf.cast(labels, float)

    def _make_dataset(self, start_index: int, end_index: int) -> tf.data.Dataset:

        if(start_index >= end_index):
            raise ValueError('start_index muste smaller then end_index')
        elif(end_index >= len(self._data)):
            raise ValueError('end_index must be smaller then full data set')

        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=self._data,
            targets=None,
            sequence_length=self._total_window_size,
            sequence_stride=self._label_width,
            shuffle=False,
            start_index=start_index,
            end_index=end_index,
            batch_size=self._batch_size)

        ds = ds.map(self._split_window)
        return ds

    @property
    def train(self) -> tf.data.Dataset:
        # byte align bit size
        bit_end = int(len(self._data) * 0.7) - 1
        if bit_end < self._total_window_size:
            return None
        return self._make_dataset(0, bit_end)

    @property
    def val(self):
        # byte align bit size
        bit_begin = int(len(self._data) * 0.7) - self._total_window_size
        bit_end = int(len(self._data) * 0.8) - 1
        if (bit_end - bit_begin) < self._total_window_size:
            return None
        return self._make_dataset(bit_begin, bit_end)

    @property
    def test(self):
        bit_begin = int(len(self._data) * 0.8) - self._total_window_size
        if (self._data.size - bit_begin) < self._total_window_size:
            return None
        return self._make_dataset(bit_begin, len(self._data) - 1)
