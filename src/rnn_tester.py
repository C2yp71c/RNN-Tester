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

import math
from scipy import stats
from pathlib import Path
import progressbar
import datetime
import argparse
import random
import sys
from multiprocessing import Pool
from typing import Optional, Tuple, List, cast
import numpy as np
import tensorflow as tf

# random.SystemRandom does not allow multiple instances
prand = random.SystemRandom()


def print_error(text: str) -> None:
    print("[ERROR]", text, file=sys.stderr)


def print_warning(text: str) -> None:
    print("[WARNING]", text)


def print_info(text: str) -> None:
    print("[INFO]", text)


def get_meta(path: Path, function: str) -> Tuple[float, float, int, int]:
    """!@brief Loads random numbers from directory

    @param path to a directory with random numbers
    @param function random number function name
    """
    meta_path = Path(str(path) + "/" + function + ".npy")

    if meta_path.is_file():
        print("Load " + function + " metadata")
        train_guess_error, eval_guess_error, input_coloumns, target_coloumns = np.load(
            meta_path)
    else:
        raise ValueError(
            "No metadata for the PRNG " + str(function) + " in " + str(meta_path))

    return train_guess_error, eval_guess_error, np.int64(
        input_coloumns), np.int64(target_coloumns)


class Stats:
    def __init__(self, row_size: int, eval_rounds: int) -> None:
        """!@brief Get statistic form predicted values

        This function is nondeterministic and only valid for large eval_rounds

        @param row_size length of a row in the target data set
        @param eval_rounds number of eval and test rounds for prediction
        """
        if row_size <= 0:
            raise ValueError("row_size must be greater than zero")
        if eval_rounds <= 0:
            raise ValueError("eval_rounds must be greater than zero")

        self._normal = np.array([0.0] * row_size)
        for j in range(eval_rounds):
            rand1 = np.array([prand.getrandbits(1) for n in range(row_size)])
            rand2 = np.array([prand.getrandbits(1) for n in range(row_size)])
            self._normal += np.abs(rand1 - rand2)
        self._normal /= cast(float, eval_rounds)

    def get_result(self, accu: List[float]) -> Tuple[List[List[int]], List[
            List[int]], List[List[float]], List[List[float]]]:
        pool = Pool(20)
        self._accu = accu
        return zip(*pool.map(self._get_window, list(range(1, len(accu) + 1))))

    def get_valid_range(self) -> Tuple[int, int]:
        mean = np.mean(self._normal)
        std = np.std(self._normal)
        dist = np.random.normal(mean, std, 100000)
        lower_limit = np.percentile(dist, 1)
        upper_limit = np.percentile(dist, 99)
        return lower_limit, upper_limit

    def _get_window(self, window: int
                    ) -> Tuple[List[int], List[int], List[float], List[float]]:
        x = np.array([])
        y = np.array([])
        zmean: List[float] = list()
        zpval: List[float] = list()
        for pos in range(len(self._accu) - (window - 1)):
            x = np.append(x, window)
            y = np.append(y, pos)
            zmean = np.append(zmean, np.mean(self._accu[pos:pos + window]))
            zpval = np.append(
                zpval,
                stats.ks_2samp(self._accu[pos:pos + window],
                               self._normal[pos:pos + window])[1])
        return x.astype(int), y.astype(int), zmean, zpval


def prediction(options, eval_rounds: int,
               sess_path: Path) -> Tuple[float, float]:
    """!@brief Core of the RNN tester

    This function learns the dependencies between two data sets.
    The focus is to learn the dependencies in pseudo random number streams.
    """
    # FIFO queue from file
    with tf.variable_scope("queue"):
        select_q = tf.placeholder(tf.int32, [])
        # train ques
        train_target_queue = tf.train.string_input_producer(
            [
                str(options.path_string) +
                "/train_targets_" +
                str(options.FUNCTION) + ".bin"
            ],
            capacity=options.BATCH_SIZE * 100,
            shuffle=False)
        train_input_queue = tf.train.string_input_producer(
            [
                str(options.path_string) +
                "/train_inputs_" +
                str(options.FUNCTION) + ".bin"
            ],
            capacity=options.BATCH_SIZE * 100,
            shuffle=False)

        # eval queues
        eval_target_queue = tf.train.string_input_producer(
            [
                str(options.path_string) +
                "/eval_targets_" +
                str(options.FUNCTION) + ".bin"
            ],
            capacity=options.BATCH_SIZE * 100,
            shuffle=False)
        eval_input_queue = tf.train.string_input_producer(
            [
                str(options.path_string) +
                "/eval_inputs_" +
                str(options.FUNCTION) + ".bin"
            ],
            capacity=options.BATCH_SIZE * 100,
            shuffle=False)

        # Test queues
        test_target_queue = tf.train.string_input_producer(
            [
                str(options.path_string) +
                "/test_targets_" +
                str(options.FUNCTION) + ".bin"
            ],
            capacity=options.BATCH_SIZE * 100,
            shuffle=False)
        test_input_queue = tf.train.string_input_producer(
            [
                str(options.path_string) +
                "/test_inputs_" +
                str(options.FUNCTION) + ".bin"
            ],
            capacity=options.BATCH_SIZE * 100,
            shuffle=False)

        # Get metadata for input and target
        loss_guess_train, loss_guess_eval, input_lines, target_lines = get_meta(
            options.path_string, str(options.FUNCTION))

        # Select queue
        x_queue = tf.QueueBase.from_list(
            select_q, [train_input_queue, eval_input_queue, test_input_queue])
        y_queue = tf.QueueBase.from_list(
            select_q,
            [train_target_queue, eval_target_queue, test_target_queue])

        # Read from queue
        reader1 = tf.FixedLengthRecordReader(record_bytes=input_lines*options.BATCH_SIZE)
        reader2 = tf.FixedLengthRecordReader(record_bytes=target_lines*options.BATCH_SIZE)
        _, x_bin = reader1.read(x_queue)
        _, y_bin = reader2.read(y_queue)


        # Decode queue to list
        x_list = tf.decode_raw(x_bin, tf.uint8)
        y_list = tf.decode_raw(y_bin, tf.uint8)

        x_list = tf.slice(x_list,[0],[input_lines*options.BATCH_SIZE])
        y_list = tf.slice(y_list,[0],[target_lines*options.BATCH_SIZE])

        # Concatenate list elements to tensor
        x = tf.transpose(tf.reshape(
            tf.cast(x_list, tf.float32), [1, options.BATCH_SIZE, input_lines]))
        y_ = tf.transpose(tf.reshape(tf.cast(y_list, tf.float32),[options.BATCH_SIZE, target_lines]))

    # Modell
    with tf.name_scope('model'):
        # RNN output node weights and biases
        W = tf.Variable(
            tf.random_normal([options.CELLS,
                              int(y_.get_shape()[0])]),
            dtype=tf.float32)
        b = tf.Variable(
            tf.random_normal([int(y_.get_shape()[0])]), dtype=tf.float32)

        # Define the cell type
        if options.CELLTYPE == "lstm":
            cell = tf.contrib.rnn.LSTMCell
        elif options.CELLTYPE == "gru":
            cell = tf.contrib.rnn.GRUCell

        rnn_cell_fw = tf.contrib.rnn.MultiRNNCell(
            [cell(options.CELLS), cell(options.CELLS)])

        outputs, states = tf.nn.dynamic_rnn(
            rnn_cell_fw, x, dtype=tf.float32, time_major=True)
        y = tf.matmul(outputs[-1], W) + b

    # Training-Graph
    with tf.name_scope('train'):
        # Compare learned with test values
        loss_learn = tf.reduce_mean(
            tf.abs(y - tf.transpose(y_), name='loss_learn'))
        # Save the mean value
        tf.summary.scalar('loss learn', loss_learn)
        # Defines the error of guess
        tf.summary.scalar('loss guess', loss_guess_train)
        loss_tmp = loss_guess_train - loss_learn
        tf.summary.scalar('loss guess - loss_learn', loss_tmp)
        loss = (tf.negative(tf.atan(loss_tmp * options.c)) + (math.pi / 2))
        loss = tf.reduce_mean(loss, name='loss')
        tf.summary.scalar('optimized loss', loss)
        optimizer = tf.train.AdamOptimizer(options.LEARN_RATE)
        train = optimizer.minimize(loss)

    with tf.name_scope('eval'):
        # Forcecast will be compared with eval
        mean_loss = tf.reduce_mean(tf.abs(y - tf.transpose(y_)))
        loss_guess = loss_guess_eval
        # accuracy defines how much better is the algorithm then guess
        accuracy = tf.abs(loss_guess - mean_loss)

    with tf.name_scope("test"):
        # Forecast compare with correct result
        test_accuracy = tf.reduce_mean(tf.abs(y - tf.transpose(y_)))

    summary_op = tf.summary.merge_all()

    # Learn
    with tf.Session(
            config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)) as sess:
        # Init data
        init = tf.global_variables_initializer()
        sess.run(init)

        # Session Path
        sess_path = Path(
            './linear_log/' + str(datetime.datetime.now()).replace(' ', '-') +
            "-" + options.FUNCTION + "-" + options.CELLTYPE + '/')
        sess_path.parent.mkdir(parents=True, exist_ok=True)

        # Open log file
        writer = tf.summary.FileWriter(str(sess_path), sess.graph)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Create a saver object that will save all the variables
        saver = tf.train.Saver()

        # Training
        bar = progressbar.ProgressBar()
        print("Train Neuronal Network...")
        for step in bar(range(options.EPOCH)):
            # Run one step of the model.
            summary, _ = sess.run([summary_op, train], {select_q: 0})
            writer.add_summary(summary, step)

            if options.Verbose is True:
                if step > int(options.EPOCH * 0.3) and step % 10000 == 0:
                    print("\tEval Neuronal Network...")
                    # Save a checkpoint periodically.
                    saver.save(sess, str(sess_path) + "/model.ckpt")
                    with tf.Session(
                            config=tf.ConfigProto(
                                allow_soft_placement=True,
                                log_device_placement=False)) as sess1:
                        saver.restore(sess1, str(sess_path) + "/model.ckpt")

                        # Start input enqueue threads.
                        coord1 = tf.train.Coordinator()
                        threads1 = tf.train.start_queue_runners(
                            sess=sess1, coord=coord1)

                        np.set_printoptions(threshold=np.nan)
                        accu = 0.0
                        for i in range(eval_rounds):
                            accu += sess1.run(accuracy, {select_q: 1})

                        print("\t\t" + str(accu / eval_rounds) +
                              "% better the guess")

                        coord1.request_stop()
                        # Wait for threads to finish.
                        coord1.join(threads1)

        coord.request_stop()
        # Wait for threads to finish.
        coord.join(threads)

        print('Saving Results...')
        saver.save(sess, str(sess_path) + "/model.ckpt")

        writer.close()

    # Test
    with tf.Session(
            config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, str(sess_path) + "/model.ckpt")

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        accu = np.array([0.0] * target_lines, dtype=np.float64)
        zvar = 0

        bar = progressbar.ProgressBar()
        if options.Verbose == True:
            verbose_rounds = 0
            print("Test result:")
            for j in bar(range(eval_rounds)):
                listlearn, listright, listinput = sess.run([y, y_, x], {
                    select_q: 2
                })
                for _, i, l, r in zip(
                        range(verbose_rounds, 2),
                        listinput.transpose()[0], listlearn,
                        listright.transpose()):
                    print("Train: " + str(i) + "\nPrediction:\t" + str(l) +
                          "\nTarget:\t\t" + str(r) + "\nMean error: " +
                          str(np.mean(l - r)))

                    print('#' * 80)
                    verbose_rounds += 1
                h = np.abs(listlearn - listright.transpose())
                zvar += np.mean(np.var(listlearn, 1)) / eval_rounds
                accu += (np.sum(h, 0, dtype=np.float64) / listlearn.shape[0]
                         ) / eval_rounds
        else:
            for j in bar(range(eval_rounds)):
                listlearn, listright = sess.run([y, y_], {select_q: 2})
                h = np.abs(listlearn - listright.transpose())
                zvar += np.mean(np.var(listlearn, 1)) / eval_rounds
                accu += (np.sum(h, 0, dtype=np.float64) / listlearn.shape[0]
                         ) / eval_rounds

        coord.request_stop()
        coord.join(threads)
    return accu, zvar

class resultst:
    def __init__(self,
                 var: float,
                 accu: List[float],
                 eval_rounds: int,
                 verbose=False) -> None:
        self._stat = Stats(len(accu), eval_rounds)
        x, y, zmean, zpval = self._stat.get_result(accu)
        self._x = np.concatenate(x)
        self._y = np.concatenate(y)
        self._z_mean = np.concatenate(zmean)
        self._z_pval = np.concatenate(zpval)
        self._var = var
        self._verbose = verbose

    def save(self, save_dir: Path) -> None:
        np.save(str(save_dir) + "/x.npy", self._x)
        np.save(str(save_dir) + "/y.npy", self._y)
        np.save(str(save_dir) + "/mean_error.npy", self._z_mean)
        np.save(str(save_dir) + "/var.npy", self._var)
        np.save(str(save_dir) + "/pval.npy", self._z_pval)

    def eval(self) -> int:
        low, up = self._stat.get_valid_range()
        indicator_mean = (self._z_mean[-1] > low and self._z_mean[-1] < up)
        indicator_var = (self._var <= 0.25)
        indicator_p = (self._z_pval[-1] > 0.01)

        if self._verbose is True:
            print(
                "H: " + str(self._z_mean[-1]) + " in [" + str(low) + "," +
                str(up) + "] => " + str(indicator_mean) + "\nvar: " + str(
                    self._var) + " <= 0.25 => " + str(indicator_var) + "\np: "
                + str(self._z_pval[-1]) + " > 0.01 => " + str(indicator_p))

        if indicator_mean:
            if indicator_var:
                if indicator_p:
                    self._print_result("PASSED")
                    return 0
                else:
                    print_warning(
                        "Unknown result.\n\t"
                        "This could happen if the eval_rounds are too few.")
                    return 1
            else:
                if indicator_p:
                    print_error("Unknown result.\n\t"
                                "This should never happen!!!\n\t"
                                "Could be an implantation error")
                    return 1
        self._print_result("FAILED")
        return 2

    def _print_result(self, text: str) -> None:
        left_buf = int((78 - len(text)) / 2)
        right_buf = 78 - left_buf - len(text)
        print('\n\n' + '#' * 80 + '\n#' + ' ' * 78 + '#\n#' + ' ' * left_buf +
              text + ' ' * right_buf + '#\n#' + ' ' * 78 + '#\n' + '#' * 80)


def eval_args() -> Optional[argparse.Namespace]:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--randomdir",
        action="store",
        dest="path_string",
        nargs='?',
        help="Path to the folder that contains the random numbers stored as binary")
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
        "-f",
        "--function",
        action="store",
        dest="FUNCTION",
        nargs='?',
        help="Define the PRNG to find the right file")
    parser.add_argument(
        "-b",
        "--batch",
        action="store",
        type=int,
        dest="BATCH_SIZE",
        default=50,
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
    options = parser.parse_args()

    options.path_string = Path(options.path_string)
    if not options.path_string.is_dir():
        print_error(
            "The path -d " + str(options.path_string) + "is not directory")
        return None

    if options.FUNCTION is None:
        print_error("You have to define a random number generator (-f)")
        return None

    if options.BATCH_SIZE < 20 or options.BATCH_SIZE > 512:
        print_warning("Unusual batch size. This could reduce performance.")

    if options.EPOCH < 1:
        print_warning("Negativ or zero epochs makes no sense")

    if (options.EPOCH * options.BATCH_SIZE) < 40000000:
        print_warning("Untested for such small data sets.\n"
                      "\tIncrease epochs (-e) or batch size (-b)")
    if options.CELLS < 512:
        print_warning("Untested for lesser than 512 cells.\n")

    return options


def main():
    args = eval_args()
    if not args is None:
        save_dir = Path('./linear_log/' + str(datetime.datetime.now()).replace(
            ' ', '-') + "-" + str(args.FUNCTION) + "-" + str(args.CELLTYPE) + '/')
        save_dir.mkdir()
        eval_rounds = 10 if (args.EPOCH // 100) < 10 else (args.EPOCH // 100)
        accu, var = prediction(args, eval_rounds, save_dir)
        res = resultst(var, accu, eval_rounds, args.Verbose)
        res.save(save_dir)
        res.eval()


if __name__ == "__main__":
    main()
