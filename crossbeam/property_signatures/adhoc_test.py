from unittest import result
from crossbeam.property_signatures import property_signatures as deepcoder_propsig
from crossbeam.dsl import value as value_module
from crossbeam.dsl import deepcoder_operations

v1 = value_module.get_free_variable(0)
v2 = value_module.get_free_variable(1)
u1 = value_module.get_bound_variable(0)
u2 = value_module.get_bound_variable(1)

x1 = value_module.InputVariable([13, 47, 7], name='x1')
x2 = value_module.InputVariable([[-1, 2, 3, 6, 7, 8, 9], 
                                [-1, 2, 7, 8, 9, 10],
                                [-1, 2, 5, 7]], name='x2')
x3 = value_module.InputVariable([[-29, -4, -3], [-41], [-44, -34, -14]], name='x3')


def case1():
  op = deepcoder_operations.Scanl1()
  result = op.apply([x2, v1], free_variables=[v1])
  output_value = value_module.OutputValue([[-81, -56, -55], [-229], [-72]])
  deepcoder_propsig.property_signature_value(result, output_value, fixed_length=True)


def case2():
  output_value = value_module.OutputValue([55, 50, 39])
  add_op = deepcoder_operations.Add()
  four = value_module.ConstantValue(4)
  a1 = add_op.apply([four, x1])
  a2 = add_op.apply([v1, a1], free_variables=[v1])
  zip_op = deepcoder_operations.ZipWith()
  result = zip_op.apply([a2, x2, x2], [[v1], [], []], free_variables=[v1])
  deepcoder_propsig.property_signature_value(result, output_value, fixed_length=True)


def case2_1():
  output_value = value_module.OutputValue([55, 50, 39])
  sq_op = deepcoder_operations.Square()
  s1 = sq_op.apply([v2], free_variables=[v2])
  map_op = deepcoder_operations.Map()
  result = map_op.apply([s1, x3], [[v1], []], free_variables=[v1])
  deepcoder_propsig.property_signature_value(result, output_value, fixed_length=True)

def case3():
  output_value = value_module.OutputValue([-18, -50])
  even_op = deepcoder_operations.IsEven()
  l1 = even_op.apply([v2], free_variables=[v2])
  zip_op = deepcoder_operations.ZipWith()
  result = zip_op.apply([l1, v1, v1], [[u1], [], []], free_variables=[v1])
  deepcoder_propsig.property_signature_value(result, output_value, fixed_length=True)


if __name__ == '__main__':
  case1()
  case2()
  case2_1()
  case3()