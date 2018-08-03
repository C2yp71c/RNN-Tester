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


import unittest
import libweakrandom
import random
from pathlib import Path
import shutil

class TestWeakrandomMethods(unittest.TestCase):

    fail_path = Path("/test")
    pass_path = Path("./test")
    pass_prng = "test_prng"

    def test_init(self):
        with self.assertRaises(ValueError):
            libweakrandom.Weakrandom(1,self.pass_path, self.pass_prng)
        with self.assertRaises(ValueError):
            libweakrandom.Weakrandom(-1,self.pass_path, self.pass_prng)
        with self.assertRaises(ValueError):
            libweakrandom.Weakrandom(1,self.fail_path, self.pass_prng)
        with self.assertRaises(AttributeError):
            libweakrandom.Weakrandom(40,'fail', self.pass_prng)
        with self.assertRaises(TypeError):
            libweakrandom.Weakrandom('fail', self.pass_path, self.pass_prng)


        with self.assertRaises(PermissionError):
            libweakrandom.Weakrandom(40,self.fail_path, self.pass_prng)

        p = Path("./test_dir")
        test_object = libweakrandom.Weakrandom(40,p, self.pass_prng)

        self.assertTrue(test_object.data_dir != None)
        self.assertTrue(test_object._rounds != 0)
        self.assertEqual(test_object._round_size % (test_object._targ_bytes + test_object._src_bytes),0)
        self.assertTrue(p.is_dir())


    def test_save_split(self):
        p = Path("./test_dir")
        test_object = libweakrandom.Weakrandom(40, p, self.pass_prng)

        fail_size_str1 = "01"
        fail_size_str2 = "01010"
        fail_bin_str = "020002"
        fail_value_str = "0asdösadfkj"

        pass_str = "000111000111"

        with self.assertRaises(ValueError):
            test_object.save_split(fail_size_str1)
        with self.assertRaises(ValueError):
            test_object.save_split(fail_size_str2)
        with self.assertRaises(ValueError):
            test_object.save_split(fail_bin_str)
        with self.assertRaises(ValueError):
            test_object.save_split(fail_value_str)


        self.assertEqual(test_object.save_split(pass_str), False)
        files_p = [str(x.name) for x in p.iterdir()]
        self.assertTrue("train_inputs_test_prng.bin" in files_p)
        self.assertTrue("train_inputs_test_prng.bin" in files_p)

    def test_get_round_size(self):
        p = Path("./test_dir")
        test_object = libweakrandom.Weakrandom(40, p, self.pass_prng)

        self.assertTrue(test_object.get_round_size() > 2)

    def test_get_full_size(self):
        p = Path("./test_dir")
        test_object = libweakrandom.Weakrandom(40, p, self.pass_prng)

        self.assertTrue(test_object.get_full_size() >= (3*(test_object.train_number_of_lines + test_object.eval_number_of_lines + test_object.test_number_of_lines)))

    def __del__(self):
        try:
            shutil.rmtree("./test_dir")
            shutil.rmtree(self.pass_path)
        except:
            pass


if __name__ == '__main__':
    unittest.main()
