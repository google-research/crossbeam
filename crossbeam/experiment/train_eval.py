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

import numpy as np
import os
import sys
import glob
import pickle5 as cp
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from crossbeam.algorithm import synthesis
from tqdm import tqdm
import math
import functools
from functools import wraps
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
from _thread import start_new_thread
import torch.distributed as dist
import traceback
from crossbeam.dsl import domains
from crossbeam.dsl import value as value_module
from crossbeam.common.config import get_torch_device
from crossbeam.algorithm.variables import MAX_NUM_FREE_VARS
from absl import logging
import timeit
import json
import pprint
import cProfile
from torch.utils import tensorboard


def thread_wrapped_func(func):
    """Wrapped func for torch.multiprocessing.Process.
    With this wrapper we can use OMP threads in subprocesses
    otherwise, OMP_NUM_THREADS=1 is mandatory.
    How to use:
    @thread_wrapped_func
    def func_to_wrap(args ...):
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = Queue()
        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)
    return decorated_function


def task_loss(task, device, training_samples, all_values, all_signatures, model, score_normed=True):
  effective_values = len(all_signatures) + MAX_NUM_FREE_VARS
  all_values = all_values[:effective_values]
  io_embed = model.io([task.inputs_dict], [task.outputs], device=device)
  val_embed = model.val.forward_with_signatures(all_values, device=device, list_normal_signatures=all_signatures)
  loss = 0.0
  for sample in training_samples:
    arg_options, aux_info, true_arg_pos, num_vals, op = sample
    model_arg = model.arg(op)
    arg_options = torch.LongTensor(arg_options).to(device)
    cur_vals = val_embed[:num_vals]
    cur_vals = model.encode_weight(cur_vals, aux_info)
    op_state = model.init(io_embed, cur_vals, op)
    # arg selection
    arg_scores, last_state = model_arg(op_state, cur_vals, arg_options[:, :op.arity].contiguous(), need_last_state=True)
    arg_scores = torch.sum(arg_scores, dim=-1)
    nll = -arg_scores[true_arg_pos]

    # binding options
    argv_options = arg_options[true_arg_pos, op.arity:]
    argv_options = argv_options[argv_options >= 0]
    if argv_options.shape[0]:
      init_argv_state = model_arg.state_select(last_state, [true_arg_pos])
      argv_score = model_arg(init_argv_state, model.special_var_embed, argv_options.unsqueeze(0))
      nll = nll - torch.sum(argv_score)
    # TODO(hadai): if we want to allow contrastive learning then we also need argv_scores for negative samples.
    assert score_normed
    loss = loss + nll
  loss = loss / len(training_samples)
  return loss


def do_eval(eval_tasks, domain, model,
            max_search_weight, beam_size, device, verbose=True,
            timeout=None, max_values_explored=None, is_stochastic=False, use_ur=True,
            use_type_masking=True, static_weight=False):
  if verbose:
    print(f'doing eval on {len(eval_tasks)} tasks...')

  eval_start_time = timeit.default_timer()
  num_tasks_solved = 0
  json_dict = {'results': []}
  for t in eval_tasks:
    start_time = timeit.default_timer()
    if verbose:
      print('\nTask: ', t)
    with torch.no_grad():
      out, (all_values, all_signatures), stats = synthesis.synthesize(
          t, domain, model,
          device=device,
          max_weight=max_search_weight,
          k=beam_size,
          is_training=False,
          timeout=timeout,
          max_values_explored=max_values_explored,
          is_stochastic=is_stochastic,
          random_beam=False,
          use_ur=use_ur,
          masking=use_type_masking,
          static_weight=static_weight)
    elapsed_time = timeit.default_timer() - start_time
    synthesis.update_stats_with_percents(stats)
    if t.solution is None:
      task_solution = task_solution_weight = None
    elif isinstance(t.solution, str):
      task_solution = t.solution
      task_solution_weight = None
    else:
      task_solution = t.solution.expression()
      task_solution_weight = t.solution.get_weight()
    results_dict = {
        'task': str(t),
        'task_solution': task_solution,
        'task_solution_weight': task_solution_weight,
        'success': bool(out),
        'elapsed_time': elapsed_time,
        'num_unique_values': len(all_values),
        'solution': out.expression() if out else None,
        'solution_weight': out.get_weight() if out else None,
        'stats': stats,
    }
    json_dict['results'].append(results_dict)
    if verbose:
      pprint.pprint(results_dict)
      #print('Elapsed time: {:.2f}'.format(elapsed_time))
      #print('Num values explored: {}'.format(stats['num_values_explored']))
      #print('Num unique values: {}'.format(len(all_values)))
      #print('out: {} {}'.format(out, out.expression()) if out else None)
      sys.stdout.flush()
    if out is not None:
      num_tasks_solved += 1
  if verbose:
    print('\nSolved {} of {} tasks'.format(num_tasks_solved, len(eval_tasks)))
  success_rate = num_tasks_solved / len(eval_tasks)
  if verbose:
    print('eval success rate: {:.1f}%'.format(success_rate * 100))

  json_dict['num_tasks'] = len(eval_tasks)
  json_dict['num_tasks_solved'] = num_tasks_solved
  json_dict['success_rate'] = success_rate

  eval_time = timeit.default_timer() - eval_start_time
  print(f'Eval {len(eval_tasks)} tasks took {eval_time:.1f} seconds.')

  return success_rate, json_dict


def _gather_eval_info(rank, device, local_acc, local_num):
  stats = torch.tensor([local_acc * local_num, local_num], dtype=torch.float32).to(device)
  with torch.cuda.device(device):
    dist.reduce(stats, 0, op=dist.ReduceOp.SUM)
  succ = (stats[0] / stats[1]).item()
  if rank == 0:
    print('eval success rate: {:.1f}%'.format(succ * 100))
  return succ


def train_eval_loop(args, device, model, train_files, eval_tasks,
                    task_gen, trace_gen):
  def local_task_gen(domain):
    if len(train_files) or task_gen is None:
      while True:
        for fname in train_files:
          with open(fname, 'rb') as f:
            list_tasks = cp.load(f)
          random.shuffle(list_tasks)
          for i in range(0, len(list_tasks), args.grad_accumulate):
            yield list_tasks[i : i + args.grad_accumulate]
    else:
      while True:
        cur_tasks = [task_gen(domain) for _ in range(args.grad_accumulate)]
        yield cur_tasks
  domain = domains.get_domain(args.domain)
  train_gen = local_task_gen(domain)
  is_distributed = args.num_proc > 1
  if is_distributed:
    rank = dist.get_rank()
  else:
    rank = 0
  if rank == 0:
    log_folder = os.path.join(args.save_dir, 'logs')
    if not os.path.isdir(log_folder):
      os.makedirs(log_folder)
    log_writer = tensorboard.SummaryWriter(log_folder)
  model = model.to(device)
  eval_func = functools.partial(do_eval,
                                max_search_weight=args.max_search_weight,
                                beam_size=args.beam_size,
                                device=device,
                                timeout=args.timeout,
                                max_values_explored=args.max_values_explored,
                                is_stochastic=args.stochastic_beam,
                                use_ur=args.use_ur,
                                use_type_masking=args.type_masking,
                                static_weight=args.static_weight)
  if args.do_test: # test only
    assert args.num_proc == 1
    print('Doing test only!')
    succ, json_dict = eval_func(eval_tasks, domain, model, verbose=not is_distributed)
    if args.json_results_file:
      with open(args.json_results_file, 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)
      print('Wrote JSON results file at {}'.format(args.json_results_file))
    print('Done testing! Exiting.')
    sys.exit()
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  best_succ = -1
  for cur_step in range(0, args.train_steps, args.eval_every):

    # Evaluation
    if cur_step > 0:
      if rank == 0:
        logging.info('eval at step %d' % cur_step)
      succ, json_dict = eval_func(eval_tasks, domain, model, verbose=not is_distributed)
      if rank == 0:
        log_writer.add_scalar('eval/succ', succ, cur_step)
      if args.num_proc > 1:
        succ = _gather_eval_info(rank, device, succ, len(eval_tasks))
      if succ > best_succ and rank == 0 and args.save_dir:
        logging.info('saving best model dump so far with %.2f%% valid succ' % (succ * 100))
        best_succ = succ
        save_file = os.path.join(args.save_dir, 'model-best-valid.ckpt')
        torch.save(model.state_dict(), save_file)
        # Is it too slow to write eval results to a file? It might be a huge file
        # if args.json_results_file:
        #   with open(args.json_results_file, 'w') as f:
        #     json.dump(json_dict, f, indent=4, sort_keys=True)
        #   print('Wrote JSON results file at {}'.format(args.json_results_file))

    # Training
    pbar = range(args.eval_every) if rank else tqdm(range(args.eval_every))
    verbose = False  # TODO(kshi)
    profile = False  # TODO(kshi)
    for inner_step in pbar:
      grad_step_start_time = timeit.default_timer()
      optimizer.zero_grad()
      batch_tasks = next(train_gen)
      batch_traces = [list(trace_gen(t.solution)) for t in batch_tasks]
      loss_acc = []
      total_synthesis_time = 0
      for t, trace in zip(batch_tasks, batch_traces):
        with torch.no_grad():
          synthesis_start_time = timeit.default_timer()
          if profile:
            pr = cProfile.Profile()
            pr.enable()
          training_samples, (all_values, all_signatures), stats = synthesis.synthesize(
              t, domain, model, device=device,
              trace=trace,
              max_weight=args.max_search_weight,
              k=args.beam_size,
              is_training=True,
              random_beam=args.random_beam,
              masking=args.type_masking,
              static_weight=args.static_weight)
          if profile:
            pr.disable()
          synthesis_elapsed_time = timeit.default_timer() - synthesis_start_time
          if profile and synthesis_elapsed_time > 0.5:
            pr.print_stats(sort='cumtime')
            print(f'The above is for a long-running synthesis search of {synthesis_elapsed_time:.2f} seconds.')
            print('Stats:')
            pprint.pprint(stats)
          total_synthesis_time += synthesis_elapsed_time
        synthesis.update_stats_with_percents(stats)
        stats.update({
            'task_num_inputs': len(t.inputs_dict),
            'task_solution_weight': t.solution.get_weight() if t.solution else None,
            'elapsed_time': synthesis_elapsed_time,
            'num_unique_values': len(all_values),
        })
        if verbose:
          pprint.pprint(stats)

        if isinstance(training_samples, list):
          loss = task_loss(t, device, training_samples, all_values, all_signatures, model, score_normed=args.score_normed) / args.num_proc
          loss = loss / args.grad_accumulate
          loss.backward()
          loss_acc.append(loss.item())
      loss = np.sum(loss_acc)
      if is_distributed:
        for param in model.parameters():
          if param.grad is None:
            param.grad = param.data.new(param.data.shape).zero_()
          with torch.cuda.device(device):
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
      if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
      optimizer.step()
      if (inner_step + cur_step) % args.log_every == 0:
        if rank == 0:
          log_writer.add_scalar('train/loss', loss * args.num_proc, inner_step + cur_step)
        logging.info('train/loss: %.4f at step %d', loss * args.num_proc, inner_step + cur_step)
      grad_step_elapsed_time = timeit.default_timer() - grad_step_start_time
      if verbose:
        logging.info(f'Grad step time: {grad_step_elapsed_time:.2f} sec')
        logging.info(f'Synthesis time: {total_synthesis_time:.2f} sec '
              f'({total_synthesis_time * 100 / grad_step_elapsed_time:.1f}% of grad step time)')

  if rank == 0:
    print('Training finished. Performing final evaluation...')
  succ, _ = eval_func(eval_tasks, domain, model, verbose=not is_distributed)
  if args.num_proc > 1:
    _gather_eval_info(rank, device, succ, len(eval_tasks))


@thread_wrapped_func
def train_mp(args, rank, device, model, train_files, eval_tasks, task_gen, trace_gen):
  if args.num_proc > 1:
    torch.set_num_threads(1)
  os.environ['MASTER_ADDR'] = '127.0.0.1'
  os.environ['MASTER_PORT'] = args.port
  if device == 'cpu':
    backend = 'gloo'
  else:
    backend = 'gloo'
  dist.init_process_group(backend, rank=rank, world_size=args.num_proc)
  train_eval_loop(args, device, model, train_files, eval_tasks, task_gen, trace_gen)


def main_train_eval(args, model, eval_tasks, task_gen, trace_gen):
  if args.train_data_glob is not None:
    train_files = sorted(glob.glob(os.path.join(args.data_folder, args.train_data_glob)))
  else:
    train_files = []
  if args.num_proc > 1:
    if args.gpu_list is not None:
      devices = [get_torch_device(int(x.strip())) for x in args.gpu_list.split(',')]
      if len(devices) < args.num_proc:
        assert args.num_proc % len(devices) == 0
        n_proc_per_gpu = args.num_proc // len(devices)
        devices = devices * n_proc_per_gpu
    else:
      devices = ['cpu'] * args.num_proc
    assert len(devices) == args.num_proc
    nq_per_proc = math.ceil(len(eval_tasks) / args.num_proc)
    nf_per_proc = math.ceil(len(train_files) / args.num_proc)
    procs = []
    for rank, device in enumerate(devices):
      local_eval_tasks = eval_tasks[rank * nq_per_proc : (rank + 1) * nq_per_proc]
      if args.num_valid > 0:
        local_eval_tasks = local_eval_tasks[:args.num_valid]
      local_train_files = train_files[rank * nf_per_proc : (rank + 1) * nf_per_proc]
      proc = mp.Process(target=train_mp,
                        args=(args, rank, device, model, local_train_files, local_eval_tasks,
                              task_gen, trace_gen))
      procs.append(proc)
      proc.start()
    for proc in procs:
      proc.join()
  else:
    device = args.gpu
    if args.gpu_list is not None:
      device = int(args.gpu_list.strip())
    if args.num_valid > 0:
      eval_tasks = eval_tasks[:args.num_valid]
    train_eval_loop(args, get_torch_device(device), model, train_files, eval_tasks,
                    task_gen, trace_gen)
  logging.info("Training finished!!")
