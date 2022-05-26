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
  constants = list(logic_leaves().values())
  return constants, operations


MAXIMUM_ENTITIES = 10
def make_difference_task(k, operators):
    global MAXIMUM_ENTITIES

    recursive, transpose, chain, disjunction = operators
    lv = logic_input_values()
    Z, S, eq, B = lv["zero"], lv["successor"], lv["equal"], lv["bot"]

    Sk = S
    for _ in range(k-1):
        Sk = chain.apply([S,Sk]) # k step predecessor

    Pk = transpose.apply([Sk])
    
    solution = disjunction.apply([Sk,Pk])
    
    return Task({k: v.values for k, v in lv.items()},
                solution.values,
                solution)
  
def make_sub_task(k, operators):
    global MAXIMUM_ENTITIES

    recursive, transpose, chain, disjunction = operators
    lv = logic_input_values()
    Z, S, eq, B = lv["zero"], lv["successor"], lv["equal"], lv["bot"]
    
    Sk = S
    for _ in range(k-1):
        Sk = chain.apply([S,Sk]) # k step predecessor
    
    solution = transpose.apply([Sk])
    
    return Task({k: v.values for k, v in lv.items()},
                solution.values,
                solution)

def make_add_task(k, operators):
    global MAXIMUM_ENTITIES

    recursive, transpose, chain, disjunction = operators
    lv = logic_input_values()
    Z, S, eq, B = lv["zero"], lv["successor"], lv["equal"], lv["bot"]
    
    Sk = S
    for _ in range(k-1):
        Sk = chain.apply([S,Sk]) # k step predecessor
    
    solution = Sk
    
    return Task({k: v.values for k, v in lv.items()},
                solution.values,
                solution)

def make_divisible_task(k, operators):
    global MAXIMUM_ENTITIES

    recursive, transpose, chain, disjunction = operators
    lv = logic_input_values()
    Z, S, eq, B = lv["zero"], lv["successor"], lv["equal"], lv["bot"]
    
    Sk = S
    for _ in range(k-1):
        Sk = chain.apply([S,Sk]) # k step predecessor

    Pk = transpose.apply([Sk]) # predecessor
    
    solution = recursive.apply([Z,Pk,eq])   
    
    return Task({k: v.values for k, v in lv.items()},
                solution.values,
                solution)

def make_multiply_task(k, operators):
    global MAXIMUM_ENTITIES

    recursive, transpose, chain, disjunction = operators
    lv = logic_input_values()
    Z, S, eq, B = lv["zero"], lv["successor"], lv["equal"], lv["bot"]

    P = transpose.apply([S]) # predecessor
    Sk = S
    for _ in range(k-1):
        Sk = chain.apply([S,Sk]) # k step predecessor

    Pk = transpose.apply([Sk]) # predecessor
    
    solution = recursive.apply([transpose.apply([Z]),P,Pk])   
    
    return Task({k: v.values for k, v in lv.items()},
                solution.values,
                solution)

def make_divide_task(k, operators):
    global MAXIMUM_ENTITIES

    recursive, transpose, chain, disjunction = operators
    lv = logic_input_values()
    Z, S, eq, B = lv["zero"], lv["successor"], lv["equal"], lv["bot"]


    P = transpose.apply([S]) # predecessor
    Sk = S
    for _ in range(k-1):
        Sk = chain.apply([S,Sk]) # k step predecessor

    Pk = transpose.apply([Sk]) # predecessor
    
    solution = recursive.apply([transpose.apply([Z]),Pk,P])   
    
    return Task({k: v.values for k, v in lv.items()},
                solution.values,
                solution)
  
def make_greater_than_task(operators):
    global MAXIMUM_ENTITIES

    recursive, transpose, chain, disjunction = operators
    lv = logic_input_values()
    Z, S, eq, B = lv["zero"], lv["successor"], lv["equal"], lv["bot"]

    P = transpose.apply([S]) # predecessor
    
    solution = recursive.apply([P,P,eq])   
    
    return Task({k: v.values for k, v in lv.items()},
                solution.values,
                solution)

def make_less_than_task(operators):
    global MAXIMUM_ENTITIES

    recursive, transpose, chain, disjunction = operators
    lv = logic_input_values()
    Z, S, eq, B = lv["zero"], lv["successor"], lv["equal"], lv["bot"]

    solution = recursive.apply([S,S,eq])   
    
    return Task({k: v.values for k, v in lv.items()},
                solution.values,
                solution)
  
def make_greater_than_or_equal_task(operators):
    global MAXIMUM_ENTITIES

    recursive, transpose, chain, disjunction = operators
    lv = logic_input_values()
    Z, S, eq, B = lv["zero"], lv["successor"], lv["equal"], lv["bot"]

    P = transpose.apply([S]) # predecessor
    
    solution = disjunction.apply([recursive.apply([P,P,eq]), eq])
    
    return Task({k: v.values for k, v in lv.items()},
                solution.values,
                solution)


def make_less_than_or_equal_task(operators):
    global MAXIMUM_ENTITIES

    recursive, transpose, chain, disjunction = operators
    lv = logic_input_values()
    Z, S, eq, B = lv["zero"], lv["successor"], lv["equal"], lv["bot"]

    solution = disjunction.apply([recursive.apply([S,S,eq]),
                                  eq])
    
    return Task({k: v.values for k, v in lv.items()},
                solution.values,
                solution)    

def logic_leaves():
  is_zero = np.zeros(MAXIMUM_ENTITIES) > 0
  is_zero[0] = True

  S = np.roll(np.eye(MAXIMUM_ENTITIES),1)
  S[0,0] = 0
  S = S > 0

  bottom = np.zeros(MAXIMUM_ENTITIES) > 0
  eq = np.eye(MAXIMUM_ENTITIES) > 0

  return {"zero": is_zero, "successor": S,
          "equal": eq, 
          "bot": bottom}

def logic_input_values():
  return {k: value_module.InputValue([v], k) for k, v in logic_leaves().items()}
  
def logic_inputs_dict_generator(num_inputs, num_examples):
  ll = logic_leaves()
  assert num_inputs == len(ll), f"number of inputs must be {len(ll)}"
  assert num_examples == 1, "number of examples must be 1"
  
  return {k: [v] for k, v in ll.items()}
  
    
def all_manual_logic_tasks(operations):
  tasks = [make_difference_task(k,operations) for k in [1,2,3,4,5,6,7] ] +\
    [make_divisible_task(k,operations) for k in [2,3,4] ] +\
    [make_add_task(k,operations) for k in [2,3,4,5,6] ] +\
    [make_sub_task(k,operations) for k in [2,3,4,5,6] ] +\
    [make_divide_task(k,operations) for k in [2,3,4] ] +\
    [make_multiply_task(k,operations) for k in [2,3,4] ] +\
    [make_less_than_task(operations),
     make_greater_than_task(operations),
     make_less_than_or_equal_task(operations),
     make_greater_than_or_equal_task(operations)]
  return tasks
  

def main(argv):
  del argv
  random.seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)

  constants, operations = get_consts_and_ops()
  #eval_tasks = [task_gen(FLAGS, constants, operations) for _ in range(FLAGS.num_eval)]
  eval_tasks = all_manual_logic_tasks(operations)
  #eval_tasks.extend([make_connected_task(operations,p=p) for p in [0.05,0.4]])

  with open(FLAGS.output_file, 'wb') as f:
    cp.dump(eval_tasks, f, cp.HIGHEST_PROTOCOL)


if __name__ == '__main__':
  app.run(main)
