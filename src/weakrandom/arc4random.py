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
# along with this program.  If not, see <https://www.gnu.org/licenses/

from typing import List

class arc4random:
    """Generate random numbers based on the
    RC4 pseudo random generation algorithm
    """
    def __init__(self, key :str = None) -> None:
        if (key is None):
            self.key = [0 for i in range(256)]
        elif (len(key) == 256 and int(max(key)) < 256 and int(min(key)) >= 0):
            self.key = [int(x) for x in key]
        else:
            raise ValueError('key is not a RC4 key')

        # Internale state of rc4
        self._sbox = self._gen_sbox()
        self._x = 0
        self._y = 0
        self._last_values = ''

    def randbin(self, size : int) -> str:
        sample = self._prga((size // 8) + 1)
        binlist = []
        for i in range(len(sample)):
            binlist.append(str(bin(sample[i])[2:].zfill(8)))
        binstr = self._last_values + ''.join(binlist)
        self._last_values = binstr[size:]
        return binstr[:size]

    def _gen_sbox(self) -> List[int]:
        sbox = list(range(256))
        x = 0
        keySize = len(self.key)
        for i in sbox:
            x = (x + i + self.key[i % keySize]) % 256
            self._swap(sbox, i, x)
        return sbox

    def _prga(self, num_of_bytes : int) -> List[int]:
        """Weak implementation ignore rfc4345"""
        seeds = []
        for i in range(num_of_bytes):
            self._x = (self._x + 1) % 256
            self._y = (self._y + self._sbox[self._x]) % 256
            self._swap(self._sbox, self._x, self._y)
            seeds.append(
                self._sbox[(self._sbox[self._x] + self._sbox[self._y]) % 256])
        return seeds

    def _swap(self, listy : List[int], n1: int, n2: int) -> None:
        listy[n1], listy[n2] = listy[n2], listy[n1]

