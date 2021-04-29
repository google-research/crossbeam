import torch
import torch.nn as nn
import torch.nn.functional as F
from crossbeam.algorithm.synthesis import synthesize
from tqdm import tqdm
import functools


def train_step_bsize1(task, device, training_samples, all_values, model, optimizer, 
                      score_normed=True, grad_clip=5.0):
  optimizer.zero_grad()
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
  loss.backward()
  if grad_clip > 0:
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
  optimizer.step()
  return loss


def do_eval(eval_tasks, operations, constants, model,
            max_search_weight, beam_size, device):
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
  print('eval success rate: {:.1f}%'.format(succ * 100))
  return succ


def singleproc_train_eval_loop(args, device, model, optimizer, eval_tasks, operations, constants, task_gen, trace_gen):
  pbar = tqdm(range(args.train_steps))
  eval_func = functools.partial(do_eval, max_search_weight=args.max_search_weight, beam_size=args.beam_size, device=device)
  for i in pbar:
    if i % args.eval_every == 0:
      eval_func(eval_tasks, operations, constants, model)
    t = task_gen(args, constants, operations)
    trace = list(trace_gen(t.solution))
    with torch.no_grad():
      training_samples, all_values = synthesize(t, operations, constants, model,
                                                device=device,
                                                trace=trace,
                                                max_weight=args.max_search_weight,
                                                k=args.beam_size,
                                                is_training=True)
    if isinstance(training_samples, list):
      loss = train_step_bsize1(t, device, training_samples, all_values, model, optimizer,
                               score_normed=args.score_normed, grad_clip=args.grad_clip)
      pbar.set_description('train loss: %.2f' % loss)

  print('Training finished. Performing final evaluation...')
  eval_func(eval_tasks, operations, constants, model)