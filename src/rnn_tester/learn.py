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

import math
from typing import Tuple

import numpy as np

import tensorflow as tf


class learn:
    def __init__(self, args, input_lines: int, target_lines: int, mean_error: float):
        # Get meta data for input and target
        self._exp_mean_error = mean_error
        self._c = args.c
        self._CELLTYPE = args.CELLTYPE
        self._CELLS = args.CELLS
        self._target_lines = target_lines

    def _relu_bool(self, x):
        return tf.keras.backend.relu(x, max_value=1)

    def custom_loss(self, y_true, y_pred):
        mean = tf.keras.backend.mean(y_true - y_pred)
        abs_mean = self._exp_mean_error - mean
        return -1 * tf.keras.backend.tanh(abs_mean * self._c) + math.pi / 2

    def make_model(self):
        # Define the cell type
        if self._CELLTYPE == "lstm":
            cell = tf.keras.layers.LSTM
        elif self._CELLTYPE == "gru":
            cell = tf.keras.layers.GRU

        model = tf.keras.Sequential()

        model.add(cell(units=self._CELLS, return_sequences=True, time_major=False))

        model.add(cell(units=int(self._CELLS / 1.2),
                       return_sequences=True, time_major=False))

        model.add(cell(units=int(self._CELLS / 1.4),
                       return_sequences=True, time_major=False))

        model.add(cell(units=int(self._CELLS / 1.6),
                       return_sequences=True, time_major=False))

        model.add(cell(units=int(self._CELLS / 1.8), time_major=False))
        model.add(tf.keras.layers.Dropout(0.25))

        model.add(tf.keras.layers.Dense(self._target_lines))

        return model

    def prediction(self, data, model) -> Tuple[bytes, bytes]:
        listlearnd = b''
        labels = b''

        pred = model.predict(data.test).flatten()
        listlearnd = bytes(np.packbits(np.around(pred).astype(bool), axis=-1))

        for _, label in data.test.as_numpy_iterator():
            labels += bytes(np.packbits(label.flatten().astype(bool), axis=-1))

        return listlearnd, labels
