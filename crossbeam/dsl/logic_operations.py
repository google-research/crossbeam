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
import numpy as np



class MemorizeDyadicClause(operation_base.OperationBase):
    """
    P(x,y).
    """
    def __init__(self):
        super(MemorizeDyadicClause, self).__init__('MemorizeDyadicClause', 3)

    def apply_single(self, raw_arguments):
        base_relation, l, r = raw_arguments
        new_relation = force_dyadic(np.copy(base_relation))
        new_relation[l,r] = True
        return new_relation

    def tokenized_expression(self, arguments):
        return ['(','assert/2',arguments[0],arguments[1],')']

class MemorizeMonadicClause(operation_base.OperationBase):
    """
    P(x).
    """
    def __init__(self):
        super(MemorizeMonadicClause, self).__init__('MemorizeMonadicClause', 2)

    def apply_single(self, raw_arguments):
        base_relation, i = raw_arguments
        new_relation = force_monadic(np.copy(base_relation))
        new_relation[i] = True
        return new_relation

    def tokenized_expression(self, arguments):
        return ['(','assert/1',arguments[0],')']

class RecursiveClause(operation_base.OperationBase):
    """
    whenever Q is dyadic:
    P(x,y) <- Q(x,y). % base case
    P(x,y) <- R(x,z), P(z,y). % inductive case

    whenever Q is monadic:
    P(x) <- Q(x). % base case
    P(x) <- R(x,y), P(y). % inductive case

    """
    def __init__(self):
        super(RecursiveClause, self).__init__("RecursiveClause",2)

    def apply_single(self, raw_arguments):
        base_case, inductive_step = raw_arguments

        # make it so that we can also work with monadic predicates
        inductive_step = force_dyadic(inductive_step)

        truth_values = base_case
        while True:
            current_size = np.sum(truth_values)

            truth_values = inductive_step @ truth_values + truth_values
            truth_values = truth_values > 0 # not necessary but I'm paranoid that we won't get bool
            
            new_size = np.sum(truth_values)
            assert new_size >= current_size
            if new_size == current_size: break # reached fix point

        return truth_values

    def tokenized_expression(self, arguments):
        return ['(','recursive'] + arguments[0].tokenized_expression() + arguments[1].tokenized_expression() + [')']


class TransposeClause(operation_base.OperationBase):
    """
    P(x,y) <- Q(y,x).
    """
    def __init__(self):
        super(TransposeClause, self).__init__("TransposeClause",1)
    def apply_single(self, raw_arguments):
        assert len(raw_arguments) == 1
        p = force_dyadic(raw_arguments[0])
        return p.T

    def tokenized_expression(self, arguments):
        return ['(','transpose'] + arguments[0].tokenized_expression() + [')']


class DisjunctionClause(operation_base.OperationBase):
    """
    P(vars) <- Q(vars).
    P(vars) <- R(vars).
    """
    def __init__(self):
        super(DisjunctionClause, self).__init__("DisjunctionClause",2)
    def apply_single(self, raw_arguments):
        assert len(raw_arguments) == 2
        p,q = raw_arguments
        if len(p.shape) != len(q.shape):
            p = force_dyadic(p)
            q = force_dyadic(q)
            
        return (p + q) > 0
    def tokenized_expression(self, arguments):
        return ['(','disjunction'] + arguments[0].tokenized_expression() + arguments[1].tokenized_expression() + [')']


class ChainClause(operation_base.OperationBase):
    """
    P(x,y) <- Q(x,y).
    P(x,y) <- R(x,z),T(z,y).
    """
    def __init__(self):
        super(ChainClause, self).__init__("ChainClause",3)
    def apply_single(self, raw_arguments):
        q,r,t = raw_arguments

        q = force_dyadic(q)
        r = force_dyadic(r)
        t = force_dyadic(t)

        return (q + r@t) > 0
    def tokenized_expression(self, arguments):
        return ['(','chain'] + arguments[0].tokenized_expression() + arguments[1].tokenized_expression() + arguments[2].tokenized_expression() + [')']
    
def force_dyadic(maybe_monadic):
    if len(maybe_monadic.shape) == 2: return maybe_monadic
    assert len(maybe_monadic.shape) == 1
    return np.diag(maybe_monadic)
def force_monadic(maybe_dyadic):
    if len(maybe_dyadic.shape) == 1: return maybe_dyadic
    assert len(maybe_dyadic.shape) == 2
    return maybe_dyadic.diagonal()

def get_operations():
  return [
      RecursiveClause(),
      MemorizeMonadicClause(),
      MemorizeDyadicClause(),
      TransposeClause(),
      ChainClause(),
      DisjunctionClause(),
  ]
