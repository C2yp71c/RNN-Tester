#!/usr/bin/env python3

# RNN Tester - Testing cryptographically secure pseudo random generator.
# Copyright (C) 2017-2018 Tilo Fischer <tilo.fischer@aisec.fraunhofer.de>
# (employee of Fraunhofer Institute for Applied and Integrated Security)
# All rights reserved

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

from typing import List, Tuple, Dict, BinaryIO, Type
from sys import getsizeof
from pathlib import Path
import numpy as np
import os


class Weakrandom(object):
    def __init__(self, num_of_mbytes: int, data: Path, prng_name: str) -> None:
        """!@brief

        @param num_of_mbytes number of Mb generated (Min. 40)
        @param data directory that contains the results
        """
        if(num_of_mbytes < 20):
            raise ValueError("num_of_mbytes must be greater then 40")

        self.data_dir = data
        self.data_dir.mkdir(parents=True,exist_ok=True)

        self.word_size = 1
        self.train_number_of_lines = 10000000
        self.eval_number_of_lines = 100000
        self.test_number_of_lines = 1000
        self._round_size = 10000000  #  10MB
        self._rounds = 0
        self._train_lines = self.train_number_of_lines
        self._eval_lines = self.eval_number_of_lines
        self._num_of_bytes = 0
        self._src_bytes = 0
        self._rounds, self._round_size, self._num_of_bytes, self._src_bytes, self._targ_bytes = self._adjust_num_elements((num_of_mbytes * 1000000 * 3.7) // 3)
        self._prng_name = prng_name
        self._train_target_mean: List[int] = []
        self._eval_target_mean: List[int] = []
        self._train_target_count: Dict[str,int] = dict()
        self._eval_target_count: Dict[str,int] = dict()
        self._handles: Dict[str,BinaryIO] = dict()
        self._saved_init = [num_of_mbytes, data]

        INPUT_FILENAME_PATTERN = '%s_inputs_%s.bin'
        TARGET_FILENAME_PATTERN = '%s_targets_%s.bin'

        # train
        self.train_inputs_path = os.path.join(
                str(self.data_dir), INPUT_FILENAME_PATTERN %
                ("train", self._prng_name))

        try:
            os.remove(self.train_inputs_path)
        except OSError:
            pass

        self.train_targets_path = os.path.join(
                str(self.data_dir), TARGET_FILENAME_PATTERN %
                ("train", self._prng_name))

        try:
            os.remove(self.train_targets_path)
        except OSError:
            pass

        # eval
        self.eval_inputs_path = os.path.join(
                str(self.data_dir), INPUT_FILENAME_PATTERN %
                ("eval", self._prng_name))

        try:
            os.remove(self.eval_inputs_path)
        except OSError:
            pass

        self.eval_targets_path = os.path.join(
                str(self.data_dir), TARGET_FILENAME_PATTERN %
                ("eval", self._prng_name))

        try:
            os.remove(self.eval_targets_path)
        except OSError:
            pass

        # test
        self.test_inputs_path = os.path.join(
                str(self.data_dir), INPUT_FILENAME_PATTERN %
                ("test", self._prng_name))

        try:
            os.remove(self.test_inputs_path)
        except OSError:
            pass

        self.test_targets_path = os.path.join(
                str(self.data_dir), TARGET_FILENAME_PATTERN %
                ("test", self._prng_name))

        try:
            os.remove(self.test_targets_path)
        except OSError:
            pass

    def _adjust_num_elements(self, num_of_bytes: int) -> Tuple[int,int,int,int,int]:
        # 80% 20% distribution
        number_of_lines = self.train_number_of_lines + self.eval_number_of_lines + self.test_number_of_lines
        src_bytes_per_line = int((num_of_bytes * 0.80) / number_of_lines)
        if(src_bytes_per_line == 0):
            src_bytes_per_line = 1
        # 20%
        targ_bytes_per_line = src_bytes_per_line // 4
        if(targ_bytes_per_line == 0):
            targ_bytes_per_line = 1
        if((src_bytes_per_line + targ_bytes_per_line) == 0):
            raise ValueError("num_of_mbytes is to small")
        num_of_bytes = (src_bytes_per_line + targ_bytes_per_line) * (
            self.train_number_of_lines + self.eval_number_of_lines +
            self.test_number_of_lines)
        adjust_round_size = self._round_size - self._round_size % (
            src_bytes_per_line + targ_bytes_per_line)
        # rounds must be multiple of round_size
        rounds = num_of_bytes // adjust_round_size
        return rounds,adjust_round_size, num_of_bytes, src_bytes_per_line, targ_bytes_per_line

    def get_round_size(self) -> int:
        return self._round_size

    def get_full_size(self) -> int:
        return self._num_of_bytes

    def save_split(self, data: str) -> bool:
        """
        Get data for test, eval and train and split them into two files

        @param data should be multiple of get_round_size
        @return False if more data are required True if writing is finished
        """
        n = self._src_bytes + self._targ_bytes
        ret = False

        if(len(self._prng_name) == 0):
            raise ValueError("The prng name is not set. Call set_prng() before!")

        data_chars = list(set(data))
        if(len(data_chars) > 2 or len(data_chars) < 1):
            raise ValueError("data must be element of [0,1]*")
        for c in data_chars:
            if(not (c == '0' or c =='1')):
                raise ValueError("data should only contains 0 and 1")
        if len(data) % n != 0:
            raise ValueError('data (' + str(len(data)) + ") does not have the multiple size of n (" + str(n) +") instead" + str(len(data) % n))

        # Looks complex only for seed up and memory efficiency
        for x in range(len(data) // (self._round_size)):
            source_list = []
            target_list = []
            for i in range(0, self._round_size, n):
                source_list.append([
                    int(data[i + j:i + j + self.word_size])
                    for j in range(0, self._src_bytes, self.word_size)
                ])
                target_list.append([
                    int(data[i + self._src_bytes + j:
                             i + self._src_bytes + j + self.word_size])
                    for j in range(0, self._targ_bytes, self.word_size)
                ])

            ret = self._write_bin(source_list, target_list)

            data = data[self._round_size:]

        if(not len(data) % (self._round_size) == 0):
            source_list = []
            target_list = []
            for i in range(0, len(data) % (self._round_size), n):
                source_list.append([
                    int(data[i + j:i + j + self.word_size])
                    for j in range(0, self._src_bytes, self.word_size)
                ])
                target_list.append([
                    int(data[i + self._src_bytes + j:
                             i + self._src_bytes + j + self.word_size])
                    for j in range(0, self._targ_bytes, self.word_size)
                ])
            ret = self._write_bin(source_list, target_list)
        return ret

    def _write_bin(self, source_list: List[List[int]], target_list: List[List[int]]) -> bool:

        INPUT_FILENAME_PATTERN = '%s_inputs_%s.bin'
        TARGET_FILENAME_PATTERN = '%s_targets_%s.bin'
        source_list = np.array(source_list)
        target_list = np.array(target_list)

        if len(source_list) != len(target_list):
            raise ValueError(
                'source_list and target_list must have equal sizes')

        # train
        if self.train_number_of_lines > 0 and len(target_list) > 0:
            # input
            with open(self.train_inputs_path,"ab+") as inputs_handle:
                    source_list[:self.train_number_of_lines].astype('bool').tofile(inputs_handle)
            source_list = source_list[self.train_number_of_lines:]
            # target
            with open(self.train_targets_path,"ab+") as target_handle:
                    target_list[:self.train_number_of_lines].astype('bool').tofile(target_handle)            # calculate metadata
            tmp_dict = dict(
                zip(*np.unique(
                    target_list[:self.train_number_of_lines],
                    return_counts=True)))
            self._train_target_count = {
                k: self._train_target_count.get(k, 0) + tmp_dict.get(k, 0)
                for k in set(self._train_target_count) | set(tmp_dict)
            }

            self._train_target_mean = np.append(
                self._train_target_mean,
                np.mean(
                    target_list[:self.train_number_of_lines],
                    dtype=np.float64)*len(target_list[:self.train_number_of_lines]))

            length = len(target_list[:self.train_number_of_lines])
            target_list = target_list[self.train_number_of_lines:]
            self.train_number_of_lines -= length


        # eval
        if self.eval_number_of_lines > 0 and len(target_list) > 0:
            # input
            with open(self.eval_inputs_path,"ab+") as input_handle:
                    source_list[:self.eval_number_of_lines].astype('bool').tofile(input_handle)
            source_list = source_list[self.eval_number_of_lines:]
            # target
            with open(self.eval_targets_path,"ab+") as target_handle:
                    target_list[:self.train_number_of_lines +
                                self.eval_number_of_lines].astype('bool').tofile(target_handle)
            # calculate metadata
            tmp_dict = dict(
                zip(*np.unique(
                    target_list[:self.train_number_of_lines +
                                self.eval_number_of_lines],
                    return_counts=True)))
            self._eval_target_count = {
                k: self._eval_target_count.get(k, 0) + tmp_dict.get(k, 0)
                for k in set(self._eval_target_count) | set(tmp_dict)
            }
            self._eval_target_mean = np.append(
                self._eval_target_mean,
                np.mean(
                    target_list[:self.train_number_of_lines +
                                self.eval_number_of_lines],
                    dtype=np.float64) * len(target_list[:self.train_number_of_lines +
                                self.eval_number_of_lines]))

            length = len(target_list[:self.eval_number_of_lines])
            target_list = target_list[self.eval_number_of_lines:]
            self.eval_number_of_lines -= length

        # test
        if self.test_number_of_lines > 0 and len(target_list) > 0:
            # input
            with open(self.test_inputs_path,"ab+") as input_handle:
                    source_list[:self.test_number_of_lines].astype('bool').tofile(input_handle)
            source_list = source_list[self.test_number_of_lines:]
            # target
            with open(self.test_targets_path,"ab+") as target_handle:
                    target_list[:self.test_number_of_lines].astype('bool').tofile(target_handle)
            length = len(target_list[:self.test_number_of_lines])
            target_list = target_list[self.test_number_of_lines:]
            self.test_number_of_lines -= length

        if self.test_number_of_lines == 0:
            mean = np.mean(self._train_target_mean, dtype=np.float64) / (self._train_lines/len(self._train_target_mean))
            train_mean_error = np.mean(
                np.abs(
                    sorted(
                        self._train_target_count,
                        key=self._train_target_count.get)[-1] - mean),
                dtype=np.float64)

            mean = np.mean(self._eval_target_mean, dtype=np.float64) / (self._eval_lines/len(self._eval_target_mean))
            eval_mean_error = np.mean(
                np.abs(
                    sorted(
                        self._eval_target_count,
                        key=self._eval_target_count.get)[-1] - mean),
                dtype=np.float64)

            np.save(
                str(self.data_dir) + "/" + self._prng_name + ".npy", [
                    train_mean_error, eval_mean_error, self._src_bytes,
                    self._targ_bytes
                ])
            return True
        return False
