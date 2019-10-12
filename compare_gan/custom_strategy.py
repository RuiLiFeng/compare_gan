"""Utility to re-use variables created on first device on subsequent devices."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.distribute.shared_variable_creator as sv
from tensorflow.distribute.shared_variable_creator import _canonicalize_variable_name


def make_fn(shared_variable_store, device_id):
  """Construct the variable creator function for device `device_id`.
  Constructs custom variable creator functions for the given device.
  On first device (device_id == 0), it creates the variable using the
  `next_creator`, and stores it in the provided `shared_variable_store`.
  On all other devices (device_id > 0), it tries to re-use the variable
  already created with the same name. If no such variable exists, it throws an
  error.
  Additionally, we de-uniquify variable names before checking for matches. This
  helps re-use variables which are intended to be the same but have different
  names due to variable uniquification happening upstream. Since this might
  mean we may have multiple variables with the same canonical name, we store
  them in a list per canonical name and return them in the same order as well.
  Args:
    shared_variable_store: A dictionary that we will use to store variables
      created on the first device, and re-used by creators for other devices.
    device_id: Integer index of the device whose creator should be
      constructed.
  Returns:
    An appropriate creator function based on device_id.
  """
  variable_scope_access_index = {}
  assert isinstance(device_id, int)

  def create_new_variable(next_creator, *args, **kwargs):
    """Create the variable using `next_creator` and store it."""
    canonical_name = _canonicalize_variable_name(kwargs.get("name"))
    v = next_creator(*args, **kwargs)

    if canonical_name not in shared_variable_store:
      shared_variable_store[canonical_name] = []
    shared_variable_store[canonical_name].append(v)
    return v

  def reuse_variable(next_creator, *args, **kwargs):
    """Re-use existing variable from store with same name (in order)."""
    del next_creator, args
    name = kwargs.get("name")
    canonical_name = _canonicalize_variable_name(name)
    replica_index = canonical_name.find('replica/')
    if replica_index != -1 and "ExponentialMovingAverage" in canonical_name:
        canonical_name = canonical_name[0:replica_index] + canonical_name[replica_index + 8:]

    try:
      variable_index = variable_scope_access_index.get(canonical_name, 0)
      v = shared_variable_store[canonical_name][variable_index]
      # TODO(priyag): Make this variable re-use more robust by adding checks
      # that the requested shape and dtype match the existing variable.
      variable_scope_access_index[canonical_name] = variable_index + 1
      return v
    except (KeyError, IndexError):
      raise RuntimeError(
          "Tried to create variable {} with mismatching name on device {}".
          format(name, device_id))

  if device_id == 0:
    return create_new_variable
  else:
    return reuse_variable


def init_strategy():
    sv.make_fn = make_fn