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

import re
import subprocess
import tempfile
from multiprocessing.pool import ThreadPool
from os import fsync
from typing import List, Optional, Tuple


def dieharder_eval(learnd: bytes, labels: bytes) -> Tuple[Optional[List[Tuple[str, str, str]]],
                                                          Optional[List[Tuple[str, str, str]]],
                                                          Optional[List[Tuple[str, str, str]]]]:

    if len(learnd) != len(labels):
        raise Exception("Both arrays must have the same size")

    pool = ThreadPool(processes=3)
    print(learnd)
    res_learnd = pool.apply_async(dieharder_thread, (learnd,))
    print(labels)
    res_lables = pool.apply_async(dieharder_thread, (labels,))
    print(xor_bytes(learnd, labels))
    res_xor = pool.apply_async(dieharder_thread, (xor_bytes(learnd, labels),))

    return res_learnd.get(), res_lables.get(), res_xor.get()


def xor_bytes(a: bytes, b: bytes) -> Optional[bytes]:
    if(len(a) != len(b)):
        return None
    return bytes(a ^ b for a, b in zip(a, b))


def dieharder_list() -> Optional[List[int]]:
    """Return a list of test numbers supported by dieharder."""
    res = subprocess.run(["dieharder", "-l"], capture_output=True, shell=False)
    if(res.returncode != 0):
        return None
    p = re.compile(r'-d (\d+)')
    return (p.findall(res.stdout.decode()))


def dieharder_test(file, test_num: int) -> Optional[str]:
    """Run a single dieharder test on a file handler."""
    res = subprocess.run(["dieharder", "-d", str(test_num), "-g",
                          "201", "-f", file.name], capture_output=True, shell=False)
    if res.returncode == 0:
        return res.stdout.decode().replace("\\n", '\\n')
    return None


def dieharder_thread(values: bytes) -> Optional[List[Tuple[str, str, str]]]:
    """Run all dieharder test on `values` and returns a list with (test name, p-value, result)."""
    fout = tempfile.NamedTemporaryFile()
    fout.write(values)
    fout.flush()
    fsync(fout)

    test_list = dieharder_list()
    if test_list is None:
        return None

    pool = ThreadPool()
    res_pool = list()
    for test in test_list:
        if test is not None:
            res_pool.append(pool.apply_async(dieharder_test, (fout, test)))

    res = str()
    for r in res_pool:
        v = r.get()
        if v is not None:
            res += v

    fout.close()

    p = re.compile(
        r'[\t ]*([a-z_\d]*)\|[ \t]*\d*\|[ \t]*\d*\|[ \t]*\d*\|([\.0-9]*)\|[\t ]*([A-Z]*)')
    return p.findall(res)
