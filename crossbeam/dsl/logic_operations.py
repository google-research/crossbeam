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

"""Operations for the logic programming DSL."""

from crossbeam.dsl import operation_base

class MemorizeDyadicClause(operation_base.OperationBase):
    """
    P(x,y).
    """
    def __init__(self):
        super(MemorizeDyadicClause, self).__init__('MemorizeDyadicClause', 3)

    def apply_single(self, raw_arguments):
        base_relation, l, r = raw_arguments
        new_relation = np.copy(base_relation)
        new_relation[l,r] = True
        return new_relation

    def tokenized_expression(self, arguments):
        assert False

class MemorizeMonadicClause(operation_base.OperationBase):
    """
    P(x).
    """
    def __init__(self):
        super(MemorizeMonadicClause, self).__init__('MemorizeMonadicClause', 2)

    def apply_single(self, raw_arguments):
        base_relation, i = raw_arguments
        new_relation = np.copy(base_relation)
        new_relation[i] = True
        return new_relation

    def tokenized_expression(self, arguments):
        assert False

class RecursiveClause(operation_base.OperationBase):
    """
    P(x,y) <- Q(x,y). % base case
    P(x,y) <- R(x,z), P(z,y). % inductive case
    """
    def __init__(self):
        super(RecursiveClause, self).__init__("RecursiveClause",2)

    def apply_single(self, raw_arguments):
        base_case, inductive_step = raw_arguments

        truth_values = np.copy(base_case)
        while True:
            current_size = np.sum(truth_values)

            truth_values = inductive_step @ truth_values + truth_values
            truth_values = truth_values > 0 # not necessary but I'm paranoid that we won't get bool
            
            new_size = np.sum(truth_values)
            assert new_size >= current_size
            if new_size == current_size: break # reached fix point

        return truth_values

class TransposeClause(operation_base.OperationBase):
    """
    P(x,y) <- Q(y,x).
    """
    def __init__(self):
        super(TransposeClause, self).__init__("TransposeClause",1)
    def apply_single(self, raw_arguments):
        assert len(raw_arguments) == 1
        return raw_arguments[0].T

class DisjunctionClause(operation_base.OperationBase):
    """
    P(x,y) <- Q(x,y).
    P(x,y) <- R(x,y).
    """
    def __init__(self):
        super(DisjunctionClause, self).__init__("DisjunctionClause",2)
    def apply_single(self, raw_arguments):
        assert len(raw_arguments) == 2
        return (raw_arguments[0] + raw_arguments[1]) > 0

class ChainClause(operation_base.OperationBase):
    """
    P(x,y) <- Q(x,y).
    P(x,y) <- R(x,z),T(z,y).
    """
    def __init__(self):
        super(ChainClause, self).__init__("ChainClause",3)
    def apply_single(self, raw_arguments):
        q,r,t = raw_arguments

        return (q + r@t) > 0
    

def get_operations():
  return [
      RecursiveClause(),
      MemorizeMonadicClause(),
      MemorizeDyadicClause(),
      TransposeClause(),
      ChainClause(),
      DisjunctionClause(),
  ]
