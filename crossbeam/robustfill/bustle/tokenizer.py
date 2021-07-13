from crossbeam.datasets.bustle_data import ALWAYS_USED_CONSTANTS, ALWAYS_CONST_STR, COMMON_CONSTANTS

STR_PREFIX = 'CSTR'
STR_STARTS = 'STR_STARTS'
STR_ENDS = 'STR_ENDS'
AC_PREFIX = 'AC'

def quote_string(s):
  return "'{}'".format(s)


def process_prog(raw_prog):
    prog = []
    for token in raw_prog:
        if token.startswith("'") and token.endswith("'"):  # this is a string literal
            token = token[1:-1]
            if token in COMMON_CONSTANTS:
                idx = COMMON_CONSTANTS.index(token)
                prog.append('%s_%d' % (STR_PREFIX, idx))
            else:  # sequence of characters
                prog.append(STR_STARTS)
                for c in token:
                    prog.append(c)
                prog.append(STR_ENDS)
        elif token in ALWAYS_CONST_STR:
            prog.append('%s_%d' % (AC_PREFIX, ALWAYS_CONST_STR.index(token)))
        else:
            prog.append(token)
    return prog


def unprocess_prog(prog):
    raw_prog = []
    pos = 0
    while pos < len(prog):
        token = prog[pos]
        if token.startswith(AC_PREFIX):
            idx = int(token.split('_')[-1])
            raw_prog.append(ALWAYS_USED_CONSTANTS[idx])
        elif token.startswith(STR_PREFIX):
            idx = int(token.split('_')[-1])
            raw_prog.append(quote_string(COMMON_CONSTANTS[idx]))
        elif token == STR_STARTS:
            pos += 1
            c_str = ''
            while pos < len(prog) and prog[pos] != STR_ENDS:
                c_str += prog[pos]
                pos += 1
            raw_prog.append(quote_string(c_str))
        else:
            raw_prog.append(token)
        pos += 1
    return raw_prog


def seq_equal(pred, gt):
    if len(pred) != len(gt):
        return False
    for i in range(len(pred)):
        if pred[i] != gt[i]:
            if gt[i].startswith('"') and gt[i].endswith('"'):  # a string literal
                if gt[i][1:-1] in COMMON_CONSTANTS:
                    return False
            else:
                return False
    return True
