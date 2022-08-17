"""Operations for the DeepCoder domain."""

from crossbeam.dsl import operation_base


class DeepCoderOperation(operation_base.OperationBase):

  def __init__(self, *args, **kwargs):
    super(DeepCoderOperation, self).__init__(self.__class__.__name__,
                                             *args, **kwargs)

################################################################################
# First-order functions returning int.
################################################################################


class Add(DeepCoderOperation):

  def __init__(self):
    super(Add, self).__init__(2)

  def apply_single(self, raw_args):
    left, right = raw_args
    if type(left) not in (int, str) or type(right) not in (int, str):  # pylint: disable=unidiomatic-typecheck
      return None
    return left + right


class Subtract(DeepCoderOperation):

  def __init__(self):
    super(Subtract, self).__init__(2)

  def apply_single(self, raw_args):
    left, right = raw_args
    if type(left) is not int or type(right) is not int:  # pylint: disable=unidiomatic-typecheck
      return None
    return left - right


class Multiply(DeepCoderOperation):

  def __init__(self):
    super(Multiply, self).__init__(2)

  def apply_single(self, raw_args):
    left, right = raw_args
    if type(left) is not int or type(right) is not int:  # pylint: disable=unidiomatic-typecheck
      return None
    return left * right


class IntDivide(DeepCoderOperation):

  def __init__(self):
    super(IntDivide, self).__init__(2)

  def apply_single(self, raw_args):
    left, right = raw_args
    if type(left) is not int or type(right) is not int:  # pylint: disable=unidiomatic-typecheck
      return None
    return left // right


class Square(DeepCoderOperation):

  def __init__(self):
    super(Square, self).__init__(1)

  def apply_single(self, raw_args):
    x = raw_args[0]
    if type(x) is not int:  # pylint: disable=unidiomatic-typecheck
      return None
    return x ** 2


class Min(DeepCoderOperation):

  def __init__(self):
    super(Min, self).__init__(2)

  def apply_single(self, raw_args):
    left, right = raw_args
    if type(left) is not int or type(right) is not int:  # pylint: disable=unidiomatic-typecheck
      return None
    return min(left, right)


class Max(DeepCoderOperation):

  def __init__(self):
    super(Max, self).__init__(2)

  def apply_single(self, raw_args):
    left, right = raw_args
    if type(left) is not int or type(right) is not int:  # pylint: disable=unidiomatic-typecheck
      return None
    return max(left, right)


################################################################################
# First-order functions returning bool.
################################################################################


class Greater(DeepCoderOperation):

  def __init__(self):
    super(Greater, self).__init__(2)

  def apply_single(self, raw_args):
    left, right = raw_args
    if type(left) is not int or type(right) is not int:  # pylint: disable=unidiomatic-typecheck
      return None
    return left > right


class Less(DeepCoderOperation):

  def __init__(self):
    super(Less, self).__init__(2)

  def apply_single(self, raw_args):
    left, right = raw_args
    if type(left) is not int or type(right) is not int:  # pylint: disable=unidiomatic-typecheck
      return None
    return left < right


class IsEven(DeepCoderOperation):

  def __init__(self):
    super(IsEven, self).__init__(1)

  def apply_single(self, raw_args):
    x = raw_args[0]
    if type(x) is not int:  # pylint: disable=unidiomatic-typecheck
      return None
    return x % 2 == 0


class IsOdd(DeepCoderOperation):

  def __init__(self):
    super(IsOdd, self).__init__(1)

  def apply_single(self, raw_args):
    x = raw_args[0]
    if type(x) is not int:  # pylint: disable=unidiomatic-typecheck
      return None
    return x % 2 == 1


################################################################################
# First-order functions manipulating lists (returning list or an element).
################################################################################


class Head(DeepCoderOperation):

  def __init__(self):
    super(Head, self).__init__(1)

  def apply_single(self, raw_args):
    x = raw_args[0]
    return x[0]


class Last(DeepCoderOperation):

  def __init__(self):
    super(Last, self).__init__(1)

  def apply_single(self, raw_args):
    x = raw_args[0]
    return x[-1]


class Take(DeepCoderOperation):

  def __init__(self):
    super(Take, self).__init__(2)

  def apply_single(self, raw_args):
    xs, n = raw_args
    if type(n) is not int:  # pylint: disable=unidiomatic-typecheck
      return None
    return xs[:n]


class Drop(DeepCoderOperation):

  def __init__(self):
    super(Drop, self).__init__(2)

  def apply_single(self, raw_args):
    xs, n = raw_args
    if type(n) is not int:  # pylint: disable=unidiomatic-typecheck
      return None
    return xs[n:]


class Access(DeepCoderOperation):

  def __init__(self):
    super(Access, self).__init__(2)

  def apply_single(self, raw_args):
    xs, n = raw_args
    # DeepCoder chooses to error if n is negative; we use Python's negative
    # indexing convention (our DSL is a superset of DeepCoder's anyway).
    if type(n) is not int:  # pylint: disable=unidiomatic-typecheck
      return None
    return xs[n]


class Minimum(DeepCoderOperation):

  def __init__(self):
    super(Minimum, self).__init__(1)

  def apply_single(self, raw_args):
    xs = raw_args[0]
    return min(xs)


class Maximum(DeepCoderOperation):

  def __init__(self):
    super(Maximum, self).__init__(1)

  def apply_single(self, raw_args):
    xs = raw_args[0]
    return max(xs)


class Reverse(DeepCoderOperation):

  def __init__(self):
    super(Reverse, self).__init__(1)

  def apply_single(self, raw_args):
    xs = raw_args[0]
    return list(reversed(xs))


class Sort(DeepCoderOperation):

  def __init__(self):
    super(Sort, self).__init__(1)

  def apply_single(self, raw_args):
    xs = raw_args[0]
    return sorted(xs)


class Sum(DeepCoderOperation):

  def __init__(self):
    super(Sum, self).__init__(1)

  def apply_single(self, raw_args):
    xs = raw_args[0]
    return sum(xs)


################################################################################
# Higher-order functions.
################################################################################


def _is_nested_list(x):
  return isinstance(x, list) and len(x) and isinstance(x[0], list)


class Map(DeepCoderOperation):

  def __init__(self):
    super(Map, self).__init__(2, num_bound_variables=[1, 0])

  def apply_single(self, raw_args):
    f, xs = raw_args
    result = list(map(f, xs))
    if _is_nested_list(result):
      return None
    return result


class Filter(DeepCoderOperation):

  def __init__(self):
    super(Filter, self).__init__(2, num_bound_variables=[1, 0])

  def apply_single(self, raw_args):
    f, xs = raw_args
    result = list(filter(f, xs))
    if _is_nested_list(result):
      return None
    return result


class Count(DeepCoderOperation):

  def __init__(self):
    super(Count, self).__init__(2, num_bound_variables=[1, 0])

  def apply_single(self, raw_args):
    f, xs = raw_args
    return len(filter(f, xs))


class ZipWith(DeepCoderOperation):

  def __init__(self):
    super(ZipWith, self).__init__(3, num_bound_variables=[2, 0, 0])

  def apply_single(self, raw_args):
    f, xs, ys = raw_args
    result = [f(x, y) for x, y in zip(xs, ys)]
    if _is_nested_list(result):
      return None
    return result


class Scanl1(DeepCoderOperation):

  def __init__(self):
    super(Scanl1, self).__init__(2, num_bound_variables=[2, 0])

  def apply_single(self, raw_args):
    f, xs = raw_args
    ys = [xs[0]]
    for n in range(1, len(xs)):
      ys.append(f(ys[n-1], xs[n]))
    if _is_nested_list(ys):
      return None
    return ys


def get_operations():
  return [
      Add(),
      Subtract(),
      Multiply(),
      IntDivide(),
      Square(),
      Min(),
      Max(),
      Greater(),
      Less(),
      IsEven(),
      IsOdd(),
      Head(),
      Last(),
      Take(),
      Drop(),
      Access(),
      Minimum(),
      Maximum(),
      Reverse(),
      Sort(),
      Sum(),
      Map(),
      Filter(),
      Count(),
      ZipWith(),
      Scanl1(),
  ]
