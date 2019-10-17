from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append('/ghome/fengrl/compare_gan/')

# pylint: disable=unused-import

from absl import app
from absl import flags
from absl import logging
import tensorflow_hub as hub
import gin
import gin.tf.external_configurables
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", None, "Where to store files.")
flags.DEFINE_string("bigbigan_dir", None, "Where to load BigBiGAN.")
flags.DEFINE_string(
    "schedule", "train",
    "Schedule to run. Options: train, continuous_eval.")
flags.DEFINE_multi_string(
    "gin_config", [],
    "List of paths to the config files.")
flags.DEFINE_multi_string(
    "gin_bindings", [],
    "Newline separated list of Gin parameter bindings.")
flags.DEFINE_string(
    "score_filename", "scores.csv",
    "Name of the CSV file with evaluation results model_dir.")

flags.DEFINE_integer(
    "num_eval_averaging_runs", 3,
    "How many times to average FID and IS")
flags.DEFINE_integer(
    "eval_every_steps", 5000,
    "Evaluate only checkpoints whose step is divisible by this integer")

flags.DEFINE_bool("use_tpu", None, "Whether running on TPU or not.")


def main():
    logging.info("Gin config: %s\nGin bindings: %s",
                 FLAGS.gin_config, FLAGS.gin_bindings)
    gin.parse_config_files_and_bindings(FLAGS.gin_config, FLAGS.gin_bindings)


