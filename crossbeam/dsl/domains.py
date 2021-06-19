"""Supported domains."""
import collections

from crossbeam.datasets import bustle_data
from crossbeam.datasets import random_data
from crossbeam.dsl import arithmetic_operations
from crossbeam.dsl import bustle_operations
from crossbeam.dsl import tuple_operations


Domain = collections.namedtuple(
    'Domain',
    ['operations', 'constants', 'constants_extractor', 'inputs_dict_generator',
     'input_charset', 'input_max_len', 'output_charset', 'output_max_len',
     'value_charset', 'value_max_len', 'program_tokens', 'output_type'])


TUPLE_DOMAIN = Domain(
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
    output_type=None)

ARITHMETIC_DOMAIN = Domain(
    operations=arithmetic_operations.get_operations(),
    constants=[0],
    constants_extractor=None,
    inputs_dict_generator=random_data.RANDOM_INTEGER_INPUTS_DICT_GENERATOR,
    input_charset='0123456789 ,',
    input_max_len=50,
    output_charset='0123456789 ,-',
    output_max_len=50,
    value_charset='0123456789 ,-[]int:',
    value_max_len=70,
    program_tokens=['(', ')', ' + ', ' - ', ' * ', ' // '],
    output_type=None)

_BUSTLE_CHARSET = ''.join(bustle_data.CHARSETS) + "'[]:"
BUSTLE_DOMAIN = Domain(
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
    output_type=str)


def get_domain(domain_str):
  if domain_str == 'tuple':
    return TUPLE_DOMAIN
  elif domain_str == 'arithmetic':
    return ARITHMETIC_DOMAIN
  elif domain_str == 'bustle':
    return BUSTLE_DOMAIN
  else:
    raise ValueError('Unknown domain: {}'.format(domain_str))
