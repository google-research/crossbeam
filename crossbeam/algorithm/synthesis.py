import jax.numpy as jnp
import numpy as np
from crossbeam.dsl import value as value_module
from crossbeam.algorithm.beam_search import beam_search


def synthesize(task, operations, constants, model, params,
               trace=[], max_weight=10, k=2, is_training=False):
  num_examples = task.num_examples

  all_values = []
  for constant in constants:
    all_values.append(value_module.ConstantValue(constant,
                                                 num_examples=num_examples))
  for input_name, input_value in task.inputs_dict.items():
    all_values.append(value_module.InputValue(input_value, name=input_name))
  output_value = value_module.OutputValue(task.outputs)
  all_value_dict = {v: i for i, v in enumerate(all_values)}

  io_embed = model['io'].encode(params['io'], task.inputs_dict, task.outputs)
  training_samples = []

  while True:
    cur_num_values = len(all_values)
    for operation in operations:
      num_values_before_op = len(all_values)
      val_embed, val_mask = model['val'].padded_encode(params['val'], all_values)
      op_state = model['init'].encode(params['init'], io_embed, val_embed, val_mask, operation)

      args, _ = beam_search(operation.arity, k,
                            val_embed, val_mask, 
                            op_state,
                            params['arg'], model['arg'])
      args = np.array(args, dtype=np.int32)
      if k > (len(all_values) ** operation.arity):
        args = args[:len(all_values) ** operation.arity]

      beam = [[all_values[i] for i in arg_list] for arg_list in args]
      trace_in_beam = -1
      for i, arg_list in enumerate(beam):
        result_value = operation.apply(arg_list)
        if result_value is None or result_value.weight > max_weight:
            continue
        if result_value in all_value_dict:
          # TODO: replace existing one if this way is simpler (less weight)
          continue
        all_value_dict[result_value] = len(all_values)
        all_values.append(result_value)
        if result_value == output_value:
          return result_value, all_values
        if len(trace) and result_value == trace[0] and trace_in_beam < 0:
          trace_in_beam = i
      if is_training and len(trace) and len(trace[0].arg_values) == operation.arity:  # TODO: formal check on compatibility
        if trace_in_beam != 0:  # construct training example
          if trace_in_beam < 0:  # true arg not found
            true_args = []
            true_val = trace[0]
            all_value_dict[true_val] = len(all_values)
            all_values.append(true_val)
            true_arg_vals = true_val.arg_values
            for i in range(operation.arity):
              true_args.append(all_value_dict[true_arg_vals[i]])
            true_args = np.array(true_args, dtype=np.int32)            
            args = np.concatenate((args, np.expand_dims(true_args, 0)), axis=0)
            trace_in_beam = args.shape[0] - 1
          training_samples.append((args, trace_in_beam, num_values_before_op, operation))
        trace.pop(0)
        if len(trace) == 0:
          return training_samples, all_values
    if len(all_values) == cur_num_values:  # no improvement
      break
  return None, None
