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
import weakrandom
import random
from pathlib import Path
import shutil

class TestWeakrandomMethods(unittest.TestCase):
    def setUp(self):
        self.wrap = weakrandom.intwraprand(random.betavariate)
        self.r = weakrandom.rand()
        self.p = Path("./test_dir")
    def test_randint(self):

        with self.assertRaises(ValueError):
            self.wrap.randint(99,22)

        for mini in range(256):
            for maxi in range(mini+1,256):
                ra = self.wrap.randint(mini,maxi)
                self.assertTrue(ra >= mini and ra <= maxi)

    def test_rand(self):
        self.assertEqual(self.r.mersen(40,self.p), None)
        self.assertEqual(self.r.sysrand(40,self.p), None)
        self.assertEqual(self.r.crand(40,self.p), None)
        self.assertEqual(self.r.javarand(40,self.p), None)
        self.assertEqual(self.r.gauss(40,self.p), None)
        self.assertEqual(self.r.triangular(40,self.p), None)
        self.assertEqual(self.r.betavariate(40,self.p), None)
        self.assertEqual(self.r.rc4rand(40,self.p), None)

if __name__ == '__main__':
    unittest.main()
