# RNN Tester - Testing cryptographically secure pseudo random generator.
# Copyright (C) 2022 Tilo Fischer
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

import argparse
import datetime
import logging
import warnings
from pathlib import Path
from typing import Optional

import rnn_tester.dieharder as dieharder

import rnn_tester.learn as learn

import rnn_tester.window_generator as wg

import tensorflow as tf


def eval_args() -> Optional[argparse.Namespace]:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        action="store",
        dest="path",
        nargs='?',
        help="Path to the file that contains the random numbers stored as binary", required=True)
    parser.add_argument(
        "-s",
        "--savedir",
        action="store",
        dest="path_save",
        default=".",
        nargs='?',
        help="Path to the folder that should contain the results", required=False)
    parser.add_argument(
        "-x",
        "--factor",
        action="store",
        type=int,
        dest="c",
        default=80,
        nargs='?',
        help="Set a factor for the loss function")
    parser.add_argument(
        "-b",
        "--batch",
        action="store",
        type=int,
        dest="BATCH_SIZE",
        default=32,
        nargs='?',
        help="Define the batch size (number of inputs per training)")
    parser.add_argument(
        "-e",
        "--epoch",
        action="store",
        type=int,
        dest="EPOCH",
        nargs='?',
        default=1000,
        help="Number of evaluating and testing repetitions")
    parser.add_argument(
        "-C",
        "--cells",
        action="store",
        type=int,
        dest="CELLS",
        nargs='?',
        default=512,
        help="Number of cells in a layer")
    parser.add_argument(
        "-l",
        "--learnrate",
        action="store",
        type=float,
        dest="LEARN_RATE",
        nargs='?',
        default=0.00001,
        help="Learning rate for optimizer")
    parser.add_argument(
        "-c",
        "--celltype",
        action="store",
        dest="CELLTYPE",
        default="lstm",
        nargs='?',
        choices=["gru", "lstm"],
        help="Define the used cells in the RNN model")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="Verbose",
        default=False,
        help="Enable additional information")
    parser.add_argument(
        "--debug",
        action="store_true",
        dest="Debug",
        default=False,
        help="Enable tfgdb debugging")
    options = parser.parse_args()

    options.path = Path(options.path)
    if not options.path.is_file():
        logging.error(
            "The path -i " + str(options.path) + " is not a file")
        return None

    if ((options.BATCH_SIZE % 8) != 0):
        logging.error("Batch size must be a multiple of 8")
        return None

    if options.BATCH_SIZE < 20 or options.BATCH_SIZE > 512:
        logging.warning("Unusual batch size. This could reduce performance.")

    if options.EPOCH < 1:
        logging.warning("Negative or zero epochs makes no sense")

    if (options.EPOCH * options.BATCH_SIZE) <= 200000:
        logging.warning("Untested for such small data sets.\n"
                        "\tIncrease epochs (-e) or batch size (-b)")
    if options.CELLS < 512:
        logging.warning("Untested for lesser than 512 cells.\n")

    return options


def main():
    # Make TF deterministic
    tf.random.set_seed(42)

    logging.basicConfig(level=logging.INFO)

    args = eval_args()
    if args is not None:
        if args.Verbose is not True:
            logging.getLogger("tensorflow").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Prevent of full memory allocation on the GPU handy for shred resources
        phys_devs = tf.config.experimental.list_physical_devices('GPU')
        for dev in phys_devs:
            tf.config.experimental.set_memory_growth(dev, True)

        tbCallBack = None
        if args.Verbose is True:
            save_dir = Path(args.path_save) / ('linear_log/'
                                               + (str(datetime.datetime.now())
                                                  .replace(' ', '-')
                                                  .replace(':', '')
                                                  .replace('.', ''))
                                               + "-"
                                               + str(args.path.stem)
                                               + "-"
                                               + str(args.CELLTYPE))
            save_dir.mkdir(parents=True, exist_ok=True)
            tbCallBack = [tf.keras.callbacks.TensorBoard(
                log_dir=save_dir, histogram_freq=1, write_graph=True, write_images=True)]

        multi_window = wg.WindowGenerator(100, 1, 0, args.BATCH_SIZE, args.path)

        data = learn.learn(args, 100, 1, multi_window.get_mean_error())

        model = data.make_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(args.LEARN_RATE),
                      loss=data.custom_loss,
                      metrics=['mae'])  # mean absolute error

        model.fit(multi_window.train, epochs=args.EPOCH,
                  steps_per_epoch=args.BATCH_SIZE,
                  callbacks=tbCallBack,
                  shuffle=False,
                  validation_data=multi_window.val,
                  validation_steps=1000,
                  batch_size=args.BATCH_SIZE)

        logging.info("Predict random values...")
        learnd, labels = data.prediction(multi_window, model)

        logging.info("Eval values with dieharder...")
        learnd_res, label_res, improved_res = dieharder.dieharder_eval(learnd, labels)

        logging.info("Analysis of the input data")
        x = 0
        for a, b, c in label_res:
            if(c == "PASSED"):
                x = x + 1

        logging.info("Diehader fails on", x, "from", str(len(label_res)), "tests")
        if x / len(label_res) >= 0.95:
            logging.info("dieharder: PASSED")
        else:
            logging.info("dieharder: FAILED")

        logging.info("Analysis of the learned values")
        x = 0
        for a, b, c in learnd_res:
            if(c == "PASSED"):
                x = x + 1
        if(x / learnd_res < 0.05):
            logging.error("The too few training the AI learnd nothing")
            exit()

        logging.info("Analysis of xor input and learned")
        x = 0
        for a1, b1, c1, a2, b2, c2 in zip(label_res, improved_res):
            logging.debug(a1, "|", b1, "->", b2, "|", c1, "->", c2)
            if(c2 == "PASSED"):
                x = x + 1

        if(x / len(improved_res) >= 0.95):
            logging.info("RNN_TEST: PASSED")
        else:
            logging.info("RNN_TEST: FAILED")
