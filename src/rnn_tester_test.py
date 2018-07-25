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
import rnn_tester
import numpy as np
from pathlib import Path
import shutil

class TestWeakrandomMethods(unittest.TestCase):
    def setUp(self):
        self.pass_path = Path("./test_dir")
        self.pass_path.mkdir(parents=True, exist_ok=True)
        self.fail_path = Path("./234987324")
        self.s = rnn_tester.Stats(10,10)
    def test_get_meta(self):

        np.save(str(self.pass_path) + "/test.npy",[0.0,0.0,1,1])
        self.assertEqual(rnn_tester.get_meta(self.pass_path,"test"),(0.0,0.0,1,1))
        np.save(str(self.pass_path) + "/test.npy",[0.123,1.773,1234,12343])
        self.assertEqual(rnn_tester.get_meta(self.pass_path,"test"),(0.123,1.773,1234,12343))

        with self.assertRaises(ValueError):
            rnn_tester.get_meta(self.pass_path,"test777777")
        with self.assertRaises(ValueError):
            rnn_tester.get_meta(self.fail_path,"/test.npy")

    def test_stats_init(self):
        with self.assertRaises(ValueError):
            rnn_tester.Stats(0,1)
        with self.assertRaises(ValueError):
            rnn_tester.Stats(1,0)

        self.assertFalse(rnn_tester.Stats(1,1) == None)

    def test_get_result(self):
        x,y,zmean,zpval = self.s.get_result([0.0] * 5)

        for i,j in zip(x,([1, 1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3], [4, 4], [5])):
            np.testing.assert_array_equal(i,np.array(j))
        for i,j in zip(y,([0, 1, 2, 3, 4], [0, 1, 2, 3], [0, 1, 2], [0, 1], [0])):
            np.testing.assert_array_equal(i,np.array(j))
        for i,j in zip(zmean,([0., 0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0.], [0., 0.], [0.])):
            np.testing.assert_array_equal(i,np.array(j))
        for i,j in zip(zpval,([0.28904143, 0.28904143, 0.28904143, 0.28904143, 0.28904143], [0.0970269, 0.0970269, 0.0970269, 0.0970269], [0.03262165, 0.03262165, 0.03262165], [0.01106564, 0.01106564], [0.00378135])):
            np.testing.assert_allclose(i,np.array(j), rtol=1e-5)
    def __del__(self):
        try:
            shutil.rmtree("./test_dir")
        except:
            pass


if __name__ == '__main__':
    unittest.main()
