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
# along with this program.  If not, see <https://www.gnu.org/licenses/>

import unittest
import arc4random

class Testarc4randomMethods(unittest.TestCase):
    def test_init(self):
        fail_short_key ="234871024379832450"
        fail_long_key = "2"*257
        fail_alpha_key ="a"*256
        pass_key = "8" * 256

        with self.assertRaises(ValueError):
            arc4random.arc4random(fail_short_key)
        with self.assertRaises(ValueError):
            arc4random.arc4random(fail_long_key)
        with self.assertRaises(ValueError):
            arc4random.arc4random(fail_alpha_key)

        self.assertFalse(arc4random.arc4random(pass_key) == None)
        self.assertFalse(arc4random.arc4random() == None)

    def test_randbin(self):
        pass_key = "8" * 256
        prng = arc4random.arc4random(pass_key)

        for i in range(100):
           self.assertEqual(len(prng.randbin(i)),i)

        self.assertFalse(prng.randbin(100) == prng.randbin(100))

if __name__ == '__main__':
    unittest.main()
