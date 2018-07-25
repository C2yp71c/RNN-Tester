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
 *along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

import java.util.Random;
import java.security.SecureRandom;

public final class RandomInteger {

  public static final void main(String... args){
    //note a single Random object is reused here
    SecureRandom randomGenerator = new SecureRandom();
    byte[] by = new byte[1];
    for(int j = 0; j < Long.parseLong(args[0])/8L; j++)
    {
    	randomGenerator.nextBytes(by);
	System.out.print(Integer.toBinaryString(by[0] & 255 | 256).substring(1));
    }
    randomGenerator.nextBytes(by);
    System.out.print(Integer.toBinaryString(by[0] & 255 | 256).substring(1,(int)(Long.parseLong(args[0])%8L +1L)));
  }
}
