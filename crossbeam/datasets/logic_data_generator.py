from crossbeam.common import config

import random
import numpy as np
from absl import app
from absl import flags
import pickle as cp
from crossbeam.datasets import random_data
from crossbeam.dsl import logic_operations
from crossbeam.dsl.task import Task
from crossbeam.dsl import value as value_module

from crossbeam.datasets.tuple_data_gen import task_gen


FLAGS = flags.FLAGS


def get_consts_and_ops():
  operations = logic_operations.get_operations()
  constants = [np.array([False]*10),
               np.zeros((10,10)) > 1]
  return constants, operations


def trace_gen(value_node):
  if isinstance(value_node, value_module.OperationValue): # non-leaf
    sub_ops = []
    for ch in value_node.arg_values:
      if isinstance(ch, value_module.OperationValue): # non-leaf
        sub_ops.append(ch)
    random.shuffle(sub_ops)
    for ch in sub_ops:
      sub_trace = trace_gen(ch)
      for v in sub_trace:
        yield v
    yield value_node


MAXIMUM_ENTITIES = 10

def make_connected_task(operators,p=0.5):
    global MAXIMUM_ENTITIES
    recursive_clause = operators[0] # bad
    
    A = np.random.random((MAXIMUM_ENTITIES,MAXIMUM_ENTITIES)) < p
    # split it into two islands, can only go from one island to the other in one direction
    A[:MAXIMUM_ENTITIES//2,MAXIMUM_ENTITIES//2:] = False
    C = A
    for _ in range(MAXIMUM_ENTITIES+1):
        C = A + A@A
    A = value_module.InputValue([A],"edge")
    solution = recursive_clause.apply((A,A))
    
    return Task({"edge": A}, solution.values, solution)

def make_divisible_task(k, operators):
    global MAXIMUM_ENTITIES

    transpose = operators[3]
    chain = operators[4]
    recursive = operators[0]

    
    Z = value_module.InputValue([np.array([True] + [False]*(MAXIMUM_ENTITIES-1))],"zero")
    B = value_module.InputValue([np.zeros((MAXIMUM_ENTITIES,)) > 0],"_|_") # bottom
    
    S = np.roll(np.eye(MAXIMUM_ENTITIES),1)
    S[0,0] = 0
    S = S > 0
    S = value_module.InputValue([S],"successor")

    P = transpose.apply([S]) # predecessor
    Pk = P
    for _ in range(k):
        Pk = chain.apply([B,P,Pk]) # k step predecessor
    
    solution = recursive.apply([Z,Pk])   
    
    return Task({"zero": Z,
                 "_|_": B,
                 "successor": S},
                solution.values,
                solution)
    
    

    


def main(argv):
  del argv
  random.seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)

  constants, operations = get_consts_and_ops()
  #eval_tasks = [task_gen(FLAGS, constants, operations) for _ in range(FLAGS.num_eval)]
  eval_tasks = [make_divisible_task(k,operations) for k in [2,3,4] ]
  eval_tasks.extend([make_connected_task(operations,p=p) for p in [0.05,0.4]])

  with open(FLAGS.output_file, 'wb') as f:
    cp.dump(eval_tasks, f, cp.HIGHEST_PROTOCOL)


if __name__ == '__main__':
  app.run(main)
