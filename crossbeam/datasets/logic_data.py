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


FLAGS = flags.FLAGS


def get_consts_and_ops():
  operations = logic_operations.get_operations()
  constants = [np.array([False]*10),
               np.zeros((10,10)) > 1]
  return constants, operations


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
    
    return Task({"edge": A.values}, solution.values, solution)

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
    
    return Task({"zero": Z.values,
                 "_|_": B.values,
                 "successor": S.values},
                solution.values,
                solution)
    
    
def logic_inputs_dict_generator(num_inputs, num_examples):
  assert num_inputs == 4, "number of inputs must be 4"
  assert num_examples == 1, "number of examples must be 1"

  is_zero = np.zeros(MAXIMUM_ENTITIES) > 0
  is_zero[0] = True

  S = np.roll(np.eye(MAXIMUM_ENTITIES),1)
  S[0,0] = 0
  S = S > 0

  bottom1 = np.zeros(MAXIMUM_ENTITIES) > 0
  bottom2 = np.zeros((MAXIMUM_ENTITIES,MAXIMUM_ENTITIES)) > 0
  
  return {"zero": [is_zero],
          "successor": [S],
          "_|_/1": [bottom1],
          "_|_/2": [bottom2]}
  
    


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
