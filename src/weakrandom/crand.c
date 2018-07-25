/**
 * RNN Tester - Testing cryptographically secure pseudo random generator.
 * Copyright (C) 2017-2018 Tilo Fischer <tilo.fischer@aisec.fraunhofer.de>
 * (employee of Fraunhofer Institute for Applied and Integrated Security)
 * All rights reserved
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define INT_BYTES (sizeof(unsigned int) *8)

void int2bin(unsigned int intiger, char* bin)
{
	unsigned long mask = 1;
	for(unsigned int i = 0; i < sizeof(unsigned int) * 8; i++)
	{
		bin[i] = (intiger & mask) == 0 ? '0' : '1';
		mask *= 2;
	}
}

int main(int argc, char** argv)
{
	if(argc != 2)
		printf("Wrong number of args\n");
	else
	{
		char randstr[INT_BYTES + 1];
		long int num = strtol(argv[1], (char**) NULL,10);

		srand(0);
		unsigned int r = 0;
		for(unsigned int i = 0; i < (num)/INT_BYTES; i++)
		{
			r = rand();
			int2bin(r, randstr);
			randstr[INT_BYTES] = '\0';
			printf("%s",randstr);
		}
		r = rand();
		int2bin(r,randstr);
		randstr[num % INT_BYTES] = '\0';
		printf("%s\n", randstr);
	}

	return 0;
}
