# coding=utf-8
# Copyright 2018 Google LLC & Hwalsuk Lee.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data-related utility functions.

Includes:
- A helper class to hold images and Inception features for evaluation.
- A method to load a dataset as NumPy array.
- Sample from the generator and return the data as a NumPy array.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import logging

import numpy as np
from six.moves import range
import tensorflow as tf
import tensorflow_gan as tfgan


# Special value returned when fake image generated by GAN has nans.
NAN_DETECTED = 31337.0

INCEPTION_URL = "http://download.tensorflow.org/models/frozen_inception_v1_2015_12_05.tar.gz"
INCEPTION_FROZEN_GRAPH = "inceptionv1_for_inception_score.pb"


def get_inception_graph_def():
  return tfgan.eval.get_graph_def_from_url_tarball(  # pylint: disable=unreachable
      url=INCEPTION_URL,
      filename=INCEPTION_FROZEN_GRAPH,
      tar_filename=os.path.basename(INCEPTION_URL))


class NanFoundError(Exception):
  """Exception thrown, when the Nans are present in the output."""


class EvalDataSample(object):
  """Helper class to hold images and Inception features for evaluation.

  All properties are tensors. Images are in [0, 255].
  """

  def __init__(self, images):
    self.images = images
    self.activations = None
    self.logits = None

  def discard_images(self):
    logging.info("Deleting references to images: %s", self.images.shape)
    del self.images

  def set_inception_features(self, activations, logits):
    self.activations = activations
    self.logits = logits

  def set_num_examples(self, num_examples):
    if self.images is not None:
      assert self.images.shape[0] >= num_examples
      self.images = self.images[:num_examples]
    if self.activations is not None:
      assert self.activations.shape[0] >= num_examples
      self.activations = self.activations[:num_examples]
    if self.logits is not None:
      assert self.logits.shape[0] >= num_examples
      self.logits = self.logits[:num_examples]


def get_real_images(dataset,
                    num_examples,
                    split=None,
                    failure_on_insufficient_examples=True):
  """Get num_examples images from the given dataset/split.

  Args:
    dataset: `ImageDataset` object.
    num_examples: Number of images to read.
    split: Split of the dataset to use. If None will use the default split for
      eval defined by the dataset.
    failure_on_insufficient_examples: If True raise an exception if the
      dataset/split does not images. Otherwise will log to error and return
      fewer images.

  Returns:
    4-D NumPy array with images with values in [0, 256].

  Raises:
    ValueError: If the dataset/split does not of the number of requested number
        requested images and `failure_on_insufficient_examples` is True.
  """
  logging.info("Start loading real data.")
  with tf.Graph().as_default():
    ds = dataset.eval_input_fn(split=split)
    # Get real images from the dataset. In the case of a 1-channel
    # dataset (like MNIST) convert it to 3 channels.
    next_batch = ds.make_one_shot_iterator().get_next()[0]
    shape = [num_examples] + next_batch.shape.as_list()
    is_single_channel = shape[-1] == 1
    if is_single_channel:
      shape[-1] = 3
    real_images = np.empty(shape, dtype=np.float32)
    with tf.Session() as sess:
      for i in range(num_examples):
        try:
          b = sess.run(next_batch)
          b *= 255.0
          if is_single_channel:
            b = np.tile(b, [1, 1, 3])
          real_images[i] = b
        except tf.errors.OutOfRangeError:
          logging.error("Reached the end of dataset. Read: %d samples.", i)
          break

  if real_images.shape[0] != num_examples:
    if failure_on_insufficient_examples:
      raise ValueError("Not enough examples in the dataset %s: %d / %d" %
                       (dataset, real_images.shape[0], num_examples))
    else:
      logging.error("Not enough examples in the dataset %s: %d / %d", dataset,
                    real_images.shape[0], num_examples)

  logging.info("Done loading real data.")
  return real_images


def sample_fake_dataset(sess, generator, num_batches):
  """Returns a generated data set as a NumPy array."""
  logging.info("Generating a fake data set.")
  samples = []
  for _ in range(num_batches):
    x = sess.run(generator)
    # If NaNs were generated, ignore this checkpoint and assign a very high
    # FID score which we handle specially later.
    if np.isnan(x).any():
      logging.error("Detected NaN in fake_images! Returning NaN.")
      raise NanFoundError("Detected NaN in fake images.")
    samples.append(x)
  fake_images = np.concatenate(samples, axis=0)
  fake_images *= 255.0
  # Convert 1-channel datasets (like MNIST) to 3 channels.
  if fake_images.shape[3] == 1:
    fake_images = np.tile(fake_images, [1, 1, 1, 3])
  logging.info("Done sampling a generated data set.")
  return fake_images


def inception_transform(inputs):
  with tf.control_dependencies([
      tf.assert_greater_equal(inputs, 0.0),
      tf.assert_less_equal(inputs, 255.0)]):
    inputs = tf.identity(inputs)
  preprocessed_inputs = tf.map_fn(
      fn=tfgan.eval.preprocess_image, elems=inputs, back_prop=False)
  return tfgan.eval.run_inception(
      preprocessed_inputs,
      graph_def=get_inception_graph_def(),
      output_tensor=["pool_3:0", "logits:0"])


def inception_transform_np(inputs, batch_size):
  """Computes the inception features and logits for a given NumPy array.

  The inputs are first preprocessed to match the input shape required for
  Inception.

  Args:
    inputs: NumPy array of shape [-1, H, W, 3].
    batch_size: Batch size.

  Returns:
    A tuple of NumPy arrays with Inception features and logits for each input.
  """
  with tf.Session(graph=tf.Graph()) as sess:
    inputs_placeholder = tf.placeholder(
        dtype=tf.float32, shape=[None] + list(inputs[0].shape))
    features_and_logits = inception_transform(inputs_placeholder)
    features = []
    logits = []
    num_batches = int(np.ceil(inputs.shape[0] / batch_size))
    for i in range(num_batches):
      input_batch = inputs[i * batch_size:(i + 1) * batch_size]
      x = sess.run(
          features_and_logits, feed_dict={inputs_placeholder: input_batch})
      features.append(x[0])
      logits.append(x[1])
    features = np.vstack(features)
    logits = np.vstack(logits)
    return features, logits
