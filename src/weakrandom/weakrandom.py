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

import random
from sys import getsizeof
from pathlib import Path
import numpy as np
import os
import argparse
import libweakrandom as wr
import arc4random as rc4
from multiprocessing import Process
from typing import Callable

class intwraprand:
    def __init__(self, prng: Callable[..., float]) -> None:
        self._prng = prng

    def randint(self, min_rand: int, max_rand: int) -> int:

        if(not min_rand < max_rand):
            raise ValueError("min_rand must be smaller than max_rand")
        result = 0
        if (self._prng == random.triangular):
            result = int(
                self._prng(min_rand, max_rand + 1, (min_rand + max_rand) / 2))
        elif (self._prng == random.betavariate):
            result = int(self._prng(2, 5) * (max_rand - min_rand) + min_rand)
        elif (self._prng == random.gauss):
            result = -1
            mean = (min_rand + max_rand) / 2
            while (result < min_rand or result > max_rand):
                result = int(self._prng(mean, mean / 2))
        return result


class rand:
    _word_size=1
    
    def mersen(self, num_of_mbytes: int, data: Path) -> None:
        """
        Generates random bits with function randint()

        Python uses the Mersenne Twister it is not a cryptographical PRNG.
        """
        form = wr.Weakrandom(num_of_mbytes, data, "mersen")
        r = random
        while(not form.save_split(self._pyrand(r.randint, form.get_round_size()))):
            pass


    def sysrand(self, num_of_mbytes: int, data: Path) -> None:
        """
        Linux this function use urand (very good random numbers)
        """
        form = wr.Weakrandom(num_of_mbytes, data, "sysrand")
        s = random.SystemRandom(0)
        while(not form.save_split(self._pyrand(s.randint, form.get_round_size()))):
            pass


    def crand(self, num_of_mbytes: int, data: Path) -> None:
        """"
        c stdlib rand() with srand(0)
        """
        import subprocess
        form = wr.Weakrandom(num_of_mbytes, data, "crand")
        bits = str(subprocess.check_output(['./crand', str(form.get_full_size())]))[2:-3]
        form.save_split(bits)

    def javarand(self, num_of_mbytes: int, data: Path) -> None:
        import subprocess
        form = wr.Weakrandom(num_of_mbytes, data, "javarand")
        bits = str(subprocess.check_output(['java', 'RandomInteger', str(form.get_full_size())]))[2:-1]
        form.save_split(bits)


    def gauss(self, num_of_mbytes: int, data: Path) -> None:
        """
        Good PRNG with gaussian distribution
        """
        form = wr.Weakrandom(num_of_mbytes, data, "gauss")
        g = intwraprand(random.gauss)
        while(not form.save_split(self._pyrand(g.randint, form.get_round_size()))):
            pass

    def triangular(self, num_of_mbytes: int, data: Path) -> None:
        form = wr.Weakrandom(num_of_mbytes, data, "triangular")
        g = intwraprand(random.triangular)
        while(not form.save_split(self._pyrand(g.randint, form.get_round_size()))):
            pass

    def betavariate(self, num_of_mbytes: int, data: Path) -> None:
        form = wr.Weakrandom(num_of_mbytes, data, "betavariate")
        g = intwraprand(random.betavariate)
        while(not form.save_split(self._pyrand(g.randint, form.get_round_size()))):
            pass

    def rc4rand(self, num_of_mbytes: int, data: Path) -> None:
        form = wr.Weakrandom(num_of_mbytes, data, "rc4rand")
        a = rc4.arc4random()
        while(not form.save_split(a.randbin(form.get_round_size()))):
            pass

    def _pyrand(self, randint: Callable[[int,int], int], num_of_bytes:int) -> str:
        random.seed(0)
        pyrand = "".join([
            bin(randint(0, (2**self._word_size) - 1))[2:].zfill(
                self._word_size) for j in range(num_of_bytes // self._word_size)
        ])
        return pyrand


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-r",
        "--prngs",
        nargs="+",
        choices=[
            "crand", "javarand", "betavariate", "triangular", "gauss",
            "mersen", "sysrand", "rc4rand"
        ],
        help="Choose one, or many PRNG functions.")
    parser.add_argument(
        "-s",
        "--size",
        action="store",
        type=int,
        dest="size",
        default="200",
        help="Data size in MB. This is only a guiding value.")
    parser.add_argument(
        "-d",
        "--dir",
        action="store",
        type=Path,
        dest="folder",
        default=Path("./data"),
        help="Folder that will contain the generated data.")
    options = parser.parse_args()

    w = rand()

    prngs = []
    process = []
    prngs_dict = {
        "crand": w.crand,
        "javarand": w.javarand,
        "betavariate": w.betavariate,
        "triangular": w.triangular,
        "gauss": w.gauss,
        "mersen": w.mersen,
        "sysrand": w.sysrand,
        "rc4rand": w.rc4rand
    }

    print("Start PRNG...")
    for prng in options.prngs:
        print("\t" + str(prng))
        prngs.append(prngs_dict.get(prng))

    for p in prngs:
        process.append(Process(target=p,args=(options.size, options.folder)))
    for p in process:
        p.start()
    for p in process:
        p.join()
    print("Finished")



if __name__ == "__main__":
    main()
