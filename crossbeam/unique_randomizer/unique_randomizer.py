# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""UniqueRandomizer for CrossBeam.

This differs from standard UniqueRandomizer in two ways:
 * This supports adding new children to trie nodes (new values found during
   search).
 * In order to support that, this implementation keeps track of different info
   per trie node compared to the standard implementation.
"""

import numpy as np

EXPLORATION_FACTOR = 0.5


class _TrieNode(object):
  """A node for the UniqueRandomizer trie.

  Attributes:
    parent: The _TrieNode parent of this node, or None if this node is the root.
    index_in_parent: The index of this node in the parent, or None if this node
      is the root.
    unnorm_probs: the original unnormalized probabilities of its children. In
      the underlying probability distribution, the probability of going to
      child i equals `unnorm_probs[i] / sum(unnorm_probs)`.
    unnorm_unsampled_mass: the unnormalized unsampled probability mass of its
      children. In the altered distribution after some leaves have been sampled,
      the probability of going to child i equals
      `unnorm_unsampled_mass[i] / sum(unnorm_unsampled_mass)`.
    children: A list of _TrieNode children. A child may be None if it is not
      expanded yet. The entire list will be None if this node has never sampled
      a child yet. The list will be empty if this node is a leaf in the trie.
  """

  def __init__(self, parent, index_in_parent):
    self.parent = parent
    self.index_in_parent = index_in_parent
    self.children = None
    self.cache = {}
    self._sum_unnorm_unsampled_mass = None  # Will compute later.

  def sample_child(self, unnorm_probs):
    """Samples a child _TrieNode.

    This will create the child _TrieNode if it does not already exist.

    Args:
      unnorm_probs: A 1-D numpy array containing the initial unnormalized
        probability distribution that this node should use.

    Returns:
      A tuple of the child _TrieNode and the child's index.
    """
    num_elements = len(unnorm_probs)
    if not self.children:
      # This is the first sample. Set up children.
      self.unnorm_probs = np.array(unnorm_probs)
      self._sum_unnorm_probs = np.sum(self.unnorm_probs)
      self.unnorm_unsampled_mass = np.copy(self.unnorm_probs)
      # _sum_unnorm_unsampled_mass is not needed now, compute later in the
      # upward pass upon reaching a leaf.
      self.children = [None] * num_elements
      # Faster to choose from unnorm_probs when it's still accurate (i.e., on
      # the first sample).
      distribution = unnorm_probs / self._sum_unnorm_probs
    elif num_elements > len(self.children):
      # Adding more children.
      old_num_elements = len(self.children)
      new_unnorm_probs = np.array(unnorm_probs)
      assert np.all(self.unnorm_probs == new_unnorm_probs[:old_num_elements])
      # The handwavy part. Children now potentially have more mass to explore
      # (they could have more children as well), so the unsampled mass is
      # reduced. We don't know how much it's reduced until we go to the child
      # and see how much mass it puts toward its new children. For now, we'll
      # assume a constant proportion of new mass to the child, and this will be
      # computed more accurately on the upward pass after visiting the child.
      # Children that are sampled leaves should stay sampled.
      # TODO(kshi): Have a better heuristic here.
      child_leaf_mask = [bool(child) and child.children == []  # pylint: disable=g-explicit-bool-comparison
                         for child in self.children]
      self.unnorm_unsampled_mass = np.where(
          child_leaf_mask,
          self.unnorm_unsampled_mass,
          (self.unnorm_unsampled_mass + self.unnorm_probs * EXPLORATION_FACTOR)
          / (1 + EXPLORATION_FACTOR))
      self.unnorm_unsampled_mass = np.append(
          self.unnorm_unsampled_mass,
          new_unnorm_probs[old_num_elements:])
      self.unnorm_probs = new_unnorm_probs
      self._sum_unnorm_probs = np.sum(self.unnorm_probs)
      self._sum_unnorm_unsampled_mass = np.sum(self.unnorm_unsampled_mass)
      self.children.extend([None] * (num_elements - old_num_elements))
      distribution = (self.unnorm_unsampled_mass /
                      self._sum_unnorm_unsampled_mass)
    else:
      distribution = (self.unnorm_unsampled_mass /
                      self._sum_unnorm_unsampled_mass)

    child_index = int(np.random.choice(np.arange(num_elements), p=distribution))
    child = self.children[child_index]
    if not child:
      child = self.children[child_index] = _TrieNode(
          parent=self, index_in_parent=child_index)
    return child, child_index

  def mark_leaf_sampled(self):
    """Marks this node as a leaf (sampled) and propagates updates upward."""
    self.children = []
    node = self
    parent = node.parent
    while parent is not None:
      parent.unnorm_unsampled_mass[node.index_in_parent] = (
          0 if not node.children else (
              parent.unnorm_probs[node.index_in_parent] *
              node._sum_unnorm_unsampled_mass / node._sum_unnorm_probs))  # pylint: disable=protected-access
      parent._sum_unnorm_unsampled_mass = np.sum(parent.unnorm_unsampled_mass)  # pylint: disable=protected-access
      node = parent
      parent = node.parent

  def compute_sum_cache(self):
    if self.children:
      self._sum_unnorm_unsampled_mass = np.sum(self.unnorm_unsampled_mass)

  def needs_probabilities(self):
    """Returns whether this node needs probabilities."""
    return self.children is None

  def exhausted(self):
    """Returns whether all of the mass at this node has been sampled."""
    # Distinguish [] and None.
    if self.children == []:  # pylint: disable=g-explicit-bool-comparison
      return True
    if self._sum_unnorm_unsampled_mass is None:
      return False  # This node is not a leaf but has never been sampled from.
    return self._sum_unnorm_unsampled_mass == 0


class UniqueRandomizer(object):
  """Samples unique sequences of discrete random choices.

  When using a UniqueRandomizer object to provide randomness, the client
  algorithm must be deterministic and behave identically when given a constant
  sequence of choices.

  When a sequence of choices is complete, the client algorithm must call
  `mark_sequence_complete()`. This will update the internal data so that the
  next sampled choices form a new sequence, which is guaranteed to be different
  from previous complete sequences.

  Choices returned by a UniqueRandomizer object respect the initial probability
  distributions provided by the client algorithm, conditioned on the constraint
  that a complete sequence of choices cannot be sampled more than once.

  The `sample_*` methods all return an int in the range [0, num_choices).

  Attributes:
    current_node: The current node in the trie.
  """

  def __init__(self) -> None:
    """Initializes a UniqueRandomizer object."""
    self._root_node = _TrieNode(None, None)
    self.current_node = self._root_node

  def sample_distribution(self, unnorm_probs):
    """Samples from a given unnormalized probability distribution."""
    self.current_node, choice_index = self.current_node.sample_child(
        unnorm_probs)
    return choice_index

  def mark_sequence_complete(self):
    """Used to mark a complete sequence of choices."""
    self.current_node.mark_leaf_sampled()
    self.current_node = self._root_node

  def clear_sequence(self):
    """Clear the current sequence, as if it were not sampled."""
    node = self.current_node
    while node is not None:
      node.compute_sum_cache()
      node = node.parent
    self.current_node = self._root_node

  def needs_probabilities(self):
    """Returns whether the current node requires probabilities."""
    return self.current_node.needs_probabilities()

  def exhausted(self):
    return self.current_node.exhausted()
