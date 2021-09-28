import pickle5 as pkl
import os
from crossbeam.dsl import domains
from crossbeam.datasets.logic_data import all_manual_logic_tasks

domain = domains.get_domain("logic")
operations = domain.operations
tasks = all_manual_logic_tasks(operations)

import argparse
parser = argparse.ArgumentParser(description = "")
parser.add_argument("-o", "--output", default="~/data/crossbeam/logic_baseline")
parser.add_argument("-i", "--input", default=None)
arguments = parser.parse_args()

output_directory = os.path.expanduser(arguments.output)
os.system(f"mkdir -p {output_directory}")
if arguments.input:
    with open(os.path.expanduser(arguments.input), 'rb') as f:
        tasks = pkl.load(f)

template="""
:- use_module('metagol').
metagol:timeout(30). %%1 default 5 minutes
%% tell Metagol to use the BK
body_pred(successor/2).
body_pred(eq/2).
body_pred(iszero/1).
%% background knowledge
iszero(0).
eq(X,X).
successor(0, 1).
successor(1, 2).
successor(2, 3).
successor(3, 4).
successor(4, 5).
successor(5, 6).
successor(6, 7).
successor(7, 8).
successor(8, 9).

%% metarules
%s

:-
 Pos = [
    %s
  ],
  Neg = [
    %s
  ],
  learn(Pos,Neg).

"""

meta=["""
metarule([P,Q], [P,A,B], [[Q,A,B]]). %% identity
metarule([P,Q], [P,A,B], [[Q,B,A]]). %% inverse
metarule([P,Q,R], [P,A,B], [[Q,A],[R,A,B]]). %% precon
metarule([P,Q,R], [P,A,B], [[Q,A,B],[R,B]]). %% postcon
metarule([P,Q,R], [P,A,B], [[Q,A,C],[R,C,B]]). %% chain
metarule([P,Q], [P,A,B], [[Q,A,C],[P,C,B]]).
metarule([P,Q], [P,A,A], [[Q,A]]). %% force dyadic
metarule([P,Q], [P,A], [[Q,A,A]]). %% force monadic
""",
    """
metarule([P,Q], [P,A,B], [[Q,B,A]]). %% transpose
metarule([P,Q], [P,A,B], [[Q,A,B]]). %% disjunction
metarule([P,Q,R], [P,X,Y], [[Q,X,U],[R,Y,V],[P,U,V]]). %% recursive2
metarule([P,Q,R], [P,X], [[Q,X,U],[R,U,V],[P,V]]). %% recursive1
metarule([P,Q,R], [P,A,B], [[Q,A,C],[R,C,B]]). %% chain
metarule([P,Q], [P,A,A], [[Q,A]]). %% force dyadic
metarule([P,Q], [P,A], [[Q,A,A]]). %% force monadic
"""]


for m in range(2):
    for i in range(len(tasks)):
        specification = tasks[i].outputs[0]
        if len(specification.shape)==2:
            p = ",\n".join(f"target({x}, {y})"
                           for x in range(10)
                           for y in range(10)
                           if specification[x, y])
            n = ",\n".join(f"target({x}, {y})"
                           for x in range(10)
                           for y in range(10)
                           if not specification[x, y])
        else:
            p = ",\n".join(f"target({x})"
                           for x in range(10)
                           if specification[x])
            n = ",\n".join(f"target({x})"
                           for x in range(10)
                           if not specification[x])
        source = template%(meta[m], p, n)
        os.system(f"mkdir -p ~/data/crossbeam/logic_baseline/")
        with open(os.path.expanduser(f"{output_directory}/{i}_{m}.pl"), "w") as handle:
            handle.write(source)
        print(source)
print('Run all of the following')        
for m in range(2):
    for i in range(len(tasks)):
        print(f"timeout 30s swipl -s {output_directory}/{i}_{m}.pl -t halt | tee ~/data/crossbeam/logic_baseline/{i}_{m}.out")

print("Or run just this")
print('for m in `seq 0 1`; do for i in `seq 0 %s`; do timeout 30s swipl -s %s/"$i"_"$m".pl -t halt | tee %s/"$i"_"$m".out; done; done'%(len(tasks)-1, output_directory, output_directory))
