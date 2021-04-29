import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from crossbeam.algorithm.synthesis import synthesize
from tqdm import tqdm
import functools
from functools import wraps
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
from _thread import start_new_thread
import torch.distributed as dist
import traceback


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


def task_loss(task, device, training_samples, all_values, model, score_normed=True):
  io_embed = model.io([task.inputs_dict], [task.outputs], device=device)
  val_embed = model.val(all_values, device=device)
  loss = 0.0
  for sample in training_samples:
    arg_options, true_arg_pos, num_vals, op = sample
    arg_options = torch.LongTensor(arg_options).to(device)
    cur_vals = val_embed[:num_vals]
    op_state = model.init(io_embed, cur_vals, op)
    scores = model.arg(op_state, cur_vals, arg_options)
    scores = torch.sum(scores, dim=-1)
    if score_normed:
        nll = -scores[true_arg_pos]
    else:
        nll = -F.log_softmax(scores, dim=0)[true_arg_pos]
    loss = loss + nll
  loss = loss / len(training_samples)
  return loss


def do_eval(eval_tasks, operations, constants, model,
            max_search_weight, beam_size, device, verbose=True):
  if verbose:
    print('doing eval...')
  succ = 0.0
  for t in eval_tasks:
    out, _ = synthesize(t, operations, constants, model,
                        device=device,
                        max_weight=max_search_weight,
                        k=beam_size,
                        is_training=False)
    if out is not None:
      succ += 1.0
  succ /= len(eval_tasks)
  if verbose:
    print('eval success rate: {:.1f}%'.format(succ * 100))
  return succ


def _gather_eval_info(rank, device, local_acc, local_num):
  stats = torch.tensor([local_acc * local_num, local_num], dtype=torch.float32).to(device)
  dist.reduce(stats, 0, op=dist.ReduceOp.SUM)
  succ = (stats[0] / stats[1]).item()
  if rank == 0:
    print('eval success rate: {:.1f}%'.format(succ * 100))


def train_eval_loop(args, device, model, eval_tasks, operations, constants, task_gen, trace_gen):
  is_distributed = args.num_proc > 1
  if is_distributed:
    rank = dist.get_rank()
  else:
    rank = 0
  model = model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  pbar = tqdm(range(args.train_steps)) if rank == 0 else range(args.train_steps)
  eval_func = functools.partial(do_eval, max_search_weight=args.max_search_weight, beam_size=args.beam_size, device=device)
  for i in pbar:
    if i % args.eval_every == 0:
      succ = eval_func(eval_tasks, operations, constants, model, verbose=not is_distributed)
      if args.num_proc > 1:
        _gather_eval_info(rank, device, succ, len(eval_tasks))
    t = task_gen(args, constants, operations)
    trace = list(trace_gen(t.solution))
    with torch.no_grad():
      training_samples, all_values = synthesize(t, operations, constants, model,
                                                device=device,
                                                trace=trace,
                                                max_weight=args.max_search_weight,
                                                k=args.beam_size,
                                                is_training=True)
    optimizer.zero_grad()
    if isinstance(training_samples, list):
      loss = task_loss(t, device, training_samples, all_values, model, score_normed=args.score_normed) / args.num_proc
      loss.backward()
    else:
      loss = 0.0
    if is_distributed:
      for param in model.parameters():
        if param.grad is None:
          param.grad = param.data.new(param.data.shape).zero_()
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
    if args.grad_clip > 0:
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
    optimizer.step()
    if rank == 0:
      pbar.set_description('train loss: %.2f' % (loss * args.num_proc))

  if rank == 0:
    print('Training finished. Performing final evaluation...')
  succ = eval_func(eval_tasks, operations, constants, model, verbose=not is_distributed)
  if args.num_proc > 1:
    _gather_eval_info(rank, device, succ, len(eval_tasks))


@thread_wrapped_func
def train_mp(args, rank, device, model, eval_tasks, operations, constants, task_gen, trace_gen):
  if args.num_proc > 1:
    torch.set_num_threads(1)
  os.environ['MASTER_ADDR'] = '127.0.0.1'
  os.environ['MASTER_PORT'] = args.port
  if device == 'cpu':
    backend = 'gloo'
  else:
    backend = 'nccl'
  dist.init_process_group(backend, rank=rank, world_size=args.num_proc)
  train_eval_loop(args, device, model, eval_tasks, operations, constants, task_gen, trace_gen)
