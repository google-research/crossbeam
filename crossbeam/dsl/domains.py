"""Supported domains."""
import collections

from crossbeam.datasets import bustle_data
from crossbeam.datasets import logic_data
from crossbeam.datasets import random_data
from crossbeam.dsl import arithmetic_operations
from crossbeam.dsl import checker
from crossbeam.dsl import bustle_operations
from crossbeam.dsl import logic_operations
from crossbeam.dsl import tuple_operations


Domain = collections.namedtuple(
    'Domain',
    ['name', 'operations', 'constants', 'constants_extractor',
     'inputs_dict_generator', 'input_charset', 'input_max_len',
     'output_charset', 'output_max_len', 'value_charset', 'value_max_len',
     'program_tokens', 'output_type', 'small_value_filter', 'checker_function'])


TUPLE_DOMAIN = Domain(
    name='tuple',
    operations=tuple_operations.get_operations(),
    constants=[0],
    constants_extractor=None,
    inputs_dict_generator=random_data.RANDOM_INTEGER_INPUTS_DICT_GENERATOR,
    input_charset='0123456789 ,',
    input_max_len=50,
    output_charset='0123456789() ,',
    output_max_len=50,
    value_charset='0123456789() ,[]intuple:',
    value_max_len=70,
    program_tokens=['(', ')', ', '],
    output_type=None,
    small_value_filter=None,
    checker_function=checker.check_solution)

ARITHMETIC_DOMAIN = Domain(
    name='arithmetic',
    operations=arithmetic_operations.get_operations(),
    constants=[-1, 1, 2, 3],
    constants_extractor=None,
    inputs_dict_generator=random_data.RANDOM_INTEGER_INPUTS_DICT_GENERATOR,
    input_charset='0123456789 ,-',
    input_max_len=50,
    output_charset='0123456789 ,-',
    output_max_len=50,
    value_charset='0123456789 ,-[]int:',
    value_max_len=70,
    program_tokens=['(', ')', ' + ', ' - ', ' * ', ' // '],
    output_type=None,
    small_value_filter=lambda x: abs(x) < 1000,
    checker_function=checker.check_solution)


_BUSTLE_CHARSET = ''.join(bustle_data.CHARSETS) + "'[]:"


def _bustle_small_value_filter(x):
  if isinstance(x, int):
    return abs(x) < 100
  elif isinstance(x, str):
    return len(x) < 100
  elif isinstance(x, bool):
    return True
  else:
    raise TypeError('Intermediate value {} has unknown type {}'.format(
        x, type(x)))

BUSTLE_DOMAIN = Domain(
    name='bustle',
    operations=bustle_operations.get_operations(),
    constants=None,
    constants_extractor=bustle_data.bustle_constants_extractor,
    inputs_dict_generator=bustle_data.bustle_inputs_dict_generator,
    input_charset=_BUSTLE_CHARSET,
    input_max_len=50,
    output_charset=_BUSTLE_CHARSET,
    output_max_len=50,
    value_charset=_BUSTLE_CHARSET,
    value_max_len=70,
    program_tokens=['(', ')', ', '] + bustle_operations.bustle_op_names(),
    output_type=str,
    small_value_filter=_bustle_small_value_filter,
    checker_function=checker.check_bustle_solution)

LOGIC_DOMAIN = Domain(
    name='logic',
    operations=logic_operations.get_operations(),
    constants=[],
    constants_extractor=None,
    inputs_dict_generator=logic_data.logic_inputs_dict_generator,
    input_charset=None,
    input_max_len=50,
    output_charset=None,
    output_max_len=None,
    value_charset=None,
    value_max_len=70,
    program_tokens=['(', ')', ' '] + logic_operations.logic_op_names(),
    output_type=None,
    small_value_filter=None,
    checker_function=None)


def get_domain(domain_str):
  if domain_str == 'tuple':
    return TUPLE_DOMAIN
  elif domain_str == 'arithmetic':
    return ARITHMETIC_DOMAIN
  elif domain_str == 'bustle':
    return BUSTLE_DOMAIN
  elif domain_str == 'logic':
    return LOGIC_DOMAIN
  else:
    raise ValueError('Unknown domain: {}'.format(domain_str))
