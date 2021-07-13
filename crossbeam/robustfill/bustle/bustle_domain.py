from crossbeam.datasets.bustle_data import CHARSETS, COMMON_CONSTANTS, ALWAYS_CONST_STR
from crossbeam.dsl.domains import BUSTLE_DOMAIN, Domain
from crossbeam.robustfill.bustle.tokenizer import STR_ENDS, STR_PREFIX, STR_STARTS, AC_PREFIX


def get_bustle_char_domain():
    d = BUSTLE_DOMAIN
    toks = d.program_tokens[:]
    chars = ''.join(CHARSETS)
    for c in chars:
        toks.append(c)
    for i in range(len(ALWAYS_CONST_STR)):
        toks.append('%s_%d' % (AC_PREFIX, i))
    for i in range(len(COMMON_CONSTANTS)):
        toks.append('%s_%d' % (STR_PREFIX, i))
    toks.append(STR_STARTS)
    toks.append(STR_ENDS)
    toks = sorted(list(set(toks)))
    new_d = Domain(
        operations=d.operations,
        constants=d.constants,
        constants_extractor=d.constants_extractor,
        inputs_dict_generator=d.inputs_dict_generator,
        input_charset=d.input_charset,
        input_max_len=d.input_max_len,
        output_charset=d.output_charset,
        output_max_len=d.output_max_len,
        value_charset=d.value_charset,
        value_max_len=d.value_max_len,
        program_tokens=toks,
        output_type=d.output_type,
        small_value_filter=d.small_value_filter
    )
    return new_d
