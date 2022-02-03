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

import rnn_tester.dieharder as dieharder


def test_xor_bool():
    assert dieharder.xor_bytes(b'\xFF', b'\xFF') == b'\x00'
    assert dieharder.xor_bytes(b'\xFF', b'\x00') == b'\xFF'
    assert dieharder.xor_bytes(b'\x00', b'\xFF') == b'\xFF'
    assert dieharder.xor_bytes(b'\x00', b'\x00') == b'\x00'


def test_xor_bytes():
    a = b'\x66\x50\x1C\x9A\xA6\x4F\x4F'
    b = b'\xBB\x5C\x2A\xA8\xC5\x65\x03'
    c = b'\xDD\x0C\x36\x32\x63\x2A\x4C'
    assert dieharder.xor_bytes(a, b) == c


def test_dieharder_thread():
    a = bytearray([random.randint(0, 255) for _ in range(50000000)])
    assert dieharder.dieharder_thread(a) is not None
