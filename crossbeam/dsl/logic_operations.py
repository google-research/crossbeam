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



class RecursiveClause(operation_base.OperationBase):
    """
    whenever B is dyadic:
    P(x,y) <- B(x,y). % base case
    P(x,y) <- Q(x,u), R(y,v), P(u,v). % inductive case

    whenever B is monadic:
    P(x) <- B(x). % base case
    P(x) <- R(x,y), Q(y,z), P(z). % inductive case

    """
    TOKEN = '(recursive '
    def __init__(self):
        super(RecursiveClause, self).__init__("RecursiveClause",3)

    def apply_single(self, raw_arguments):
        base_case, step1, step2 = raw_arguments

        # make it so that we can also work with monadic predicates
        step1 = force_dyadic(step1)
        step2 = force_dyadic(step2)

        truth_values = base_case
        while True:
            current_size = np.sum(truth_values)

            truth_values = step1 @ truth_values @ (step2.T) + truth_values
            truth_values = truth_values > 0 # not necessary but I'm paranoid that we won't get bool
            
            new_size = np.sum(truth_values)
            assert new_size >= current_size
            if new_size == current_size: break # reached fix point

        return truth_values

    def tokenized_expression(self, arguments, arg_variables, free_variables):
        del arg_variables, free_variables
        return [self.__class__.TOKEN] + arguments[0].tokenized_expression() + [' '] + arguments[1].tokenized_expression()  + [' '] + arguments[2].tokenized_expression() + [')']


class TransposeClause(operation_base.OperationBase):
    """
    P(x,y) <- Q(y,x).
    """

    TOKEN = '(transpose '
    def __init__(self):
        super(TransposeClause, self).__init__("TransposeClause",1)
    def apply_single(self, raw_arguments):
        assert len(raw_arguments) == 1
        p = force_dyadic(raw_arguments[0])
        return p.T

    def tokenized_expression(self, arguments, arg_variables, free_variables):
        del arg_variables, free_variables
        return [self.__class__.TOKEN] + arguments[0].tokenized_expression() + [')']


class DisjunctionClause(operation_base.OperationBase):
    """
    P(vars) <- Q(vars).
    P(vars) <- R(vars).
    """

    TOKEN = '(disjunction '
    def __init__(self):
        super(DisjunctionClause, self).__init__("DisjunctionClause",2)
    def apply_single(self, raw_arguments):
        assert len(raw_arguments) == 2
        p,q = raw_arguments
        if len(p.shape) != len(q.shape):
            p = force_dyadic(p)
            q = force_dyadic(q)
            
        return (p + q) > 0
    def tokenized_expression(self, arguments, arg_variables, free_variables):
        del arg_variables, free_variables
        return [self.__class__.TOKEN] + arguments[0].tokenized_expression() + [' '] + arguments[1].tokenized_expression() + [')']


class ChainClause(operation_base.OperationBase):
    """
    P(x,y) <- R(x,z),T(z,y).
    """

    TOKEN = '(chain '
    def __init__(self):
        super(ChainClause, self).__init__("ChainClause",2)
    def apply_single(self, raw_arguments):
        r,t = raw_arguments

        r = force_dyadic(r)
        t = force_dyadic(t)

        return r@t > 0
    def tokenized_expression(self, arguments, arg_variables, free_variables):
        del arg_variables, free_variables
        return [self.__class__.TOKEN] + arguments[0].tokenized_expression() + [' '] + arguments[1].tokenized_expression() + [')']
    
def force_dyadic(maybe_monadic):
    """P'(X, X) <- P(X)."""
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
      # MemorizeMonadicClause(),
      # MemorizeDyadicClause(),
      TransposeClause(),
      ChainClause(),
      DisjunctionClause(),
  ]

def logic_op_names():
    return [ o.__class__.TOKEN for o in get_operations() ] 
