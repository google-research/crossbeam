"""Tests for operation_base."""

from absl.testing import absltest

from crossbeam.dsl import deepcoder_operations
from crossbeam.dsl import value as value_module


class OperationBaseTest(absltest.TestCase):

  def test_apply(self):
    colon = value_module.ConstantValue(':')
    x1 = value_module.InputVariable(['cat', 'dog'], name='x1')
    x2 = value_module.InputVariable(['12', '34'], name='x2')
    x3 = value_module.InputVariable([['abc', 'xyz'], ['Hi', 'Bye']], name='x3')
    v1 = value_module.get_free_variable(0)
    v2 = value_module.get_free_variable(1)
    u1 = value_module.get_bound_variable(0)

    add_op = deepcoder_operations.Add()
    map_op = deepcoder_operations.Map()

    # Apply operation to two concrete values.
    result_1 = add_op.apply([colon, x1])
    self.assertEqual(result_1.expression(), "Add(':', x1)")
    self.assertEqual(result_1.values, [':cat', ':dog'])
    self.assertEqual(result_1.get_weight(), 3)

    # Apply operation to a concrete value and a free variable.
    result_2 = add_op.apply([x1, v1], free_variables=[v1])
    self.assertEqual(result_2.expression(), 'lambda v1: Add(x1, v1)')
    self.assertLen(result_2.values, 2)
    self.assertEqual(result_2.values[0]('?'), 'cat?')
    self.assertEqual(result_2.values[1]('!'), 'dog!')
    self.assertEqual(result_2.get_weight(), 4)

    # Apply operation to two free variables.
    result_3 = add_op.apply([v2, v1], free_variables=[v1, v2])
    self.assertEqual(result_3.expression(), 'lambda v1, v2: Add(v2, v1)')
    self.assertLen(result_3.values, 1)
    self.assertEqual(result_3.values[0]('X', 'Y'), 'YX')
    self.assertEqual(result_3.get_weight(), 5)

    # Apply operation to a lambda, giving it an input variable.
    result_4 = add_op.apply([result_2, colon],
                            arg_variables=[[x2], []],
                            free_variables=[])
    self.assertEqual(result_4.expression(),
                     "Add((lambda v1: Add(x1, v1))(x2), ':')")
    self.assertEqual(result_4.values, ['cat12:', 'dog34:'])
    self.assertEqual(result_4.get_weight(), 6)

    # Apply operation to a lambda, giving it a free variable.
    result_5 = add_op.apply([result_2, colon],
                            arg_variables=[[v1], []],
                            free_variables=[v1])
    self.assertEqual(result_5.expression(),
                     "lambda v1: Add((lambda v1: Add(x1, v1))(v1), ':')")
    self.assertLen(result_5.values, 2)
    self.assertEqual(result_5.values[0]('?'), 'cat?:')
    self.assertEqual(result_5.values[1]('!'), 'dog!:')
    self.assertEqual(result_5.get_weight(), 7)

    # Apply operation with a required lambda, giving it an input variable.
    result_6 = map_op.apply([result_2, x3],
                            arg_variables=[[x2], []],
                            free_variables=[])
    self.assertEqual(result_6.expression(),
                     'Map(lambda u1: (lambda v1: Add(x1, v1))(x2), x3)')
    self.assertEqual(result_6.values, [['cat12', 'cat12'], ['dog34', 'dog34']])
    self.assertEqual(result_6.get_weight(), 6)

    # Apply operation with a required lambda, giving it a bound variable.
    result_7 = map_op.apply([result_2, x3],
                            arg_variables=[[u1], []],
                            free_variables=[])
    self.assertEqual(result_7.expression(),
                     'Map(lambda u1: (lambda v1: Add(x1, v1))(u1), x3)')
    self.assertEqual(result_7.values, [['catabc', 'catxyz'],
                                       ['dogHi', 'dogBye']])
    self.assertEqual(result_7.get_weight(), 6)

    # Apply operation with a required lambda, giving it a free variable.
    result_8 = map_op.apply([result_2, x3],
                            arg_variables=[[v1], []],
                            free_variables=[v1])
    self.assertEqual(
        result_8.expression(),
        'lambda v1: Map(lambda u1: (lambda v1: Add(x1, v1))(v1), x3)')
    self.assertEqual(result_8.values[0]('?'), ['cat?', 'cat?'])
    self.assertEqual(result_8.values[1]('!'), ['dog!', 'dog!'])
    self.assertEqual(result_8.get_weight(), 7)

    # Apply operation with a required lambda, using a concrete value.
    result_9 = map_op.apply([result_1, x3],
                            arg_variables=[[], []],
                            free_variables=[])
    self.assertEqual(result_9.expression(), "Map(lambda u1: Add(':', x1), x3)")
    self.assertEqual(result_9.values, [[':cat', ':cat'], [':dog', ':dog']])
    self.assertEqual(result_9.get_weight(), 5)

    # Apply operation with a required lambda, using a constant.
    result_10 = map_op.apply([colon, x3],
                             arg_variables=[[], []],
                             free_variables=[])
    self.assertEqual(result_10.expression(), "Map(lambda u1: ':', x3)")
    self.assertEqual(result_10.values, [[':', ':'], [':', ':']])
    self.assertEqual(result_10.get_weight(), 3)


if __name__ == '__main__':
  absltest.main()
