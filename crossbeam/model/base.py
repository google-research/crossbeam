import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import functools
import numpy as np
from torch.nn.utils.rnn import pack_sequence, PackedSequence, pad_packed_sequence, pack_padded_sequence


def _glorot_uniform(t):
  if len(t.size()) == 2:
    fan_in, fan_out = t.size()
  elif len(t.size()) == 3:
    fan_in = t.size()[1] * t.size()[2]
    fan_out = t.size()[0] * t.size()[2]
  else:
    fan_in = np.prod(t.size())
    fan_out = np.prod(t.size())

  limit = np.sqrt(6.0 / (fan_in + fan_out))
  t.uniform_(-limit, limit)


def pad_sequence(sequences, max_len=None, batch_first=False, padding_value=0):
  # assuming trailing dimensions and type of all the Tensors
  # in sequences are same and fetching those from sequences[0]
  max_size = sequences[0].size()
  trailing_dims = max_size[1:]
  if max_len is None:
    max_len = max([s.size(0) for s in sequences])
  if batch_first:
    out_dims = (len(sequences), max_len) + trailing_dims
  else:
    out_dims = (max_len, len(sequences)) + trailing_dims

  out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
  for i, tensor in enumerate(sequences):
    length = tensor.size(0)
    # use index notation to prevent duplicate references to the tensor
    if batch_first:
      out_tensor[i, :length, Ellipsis] = tensor
    else:
      out_tensor[:length, i, Ellipsis] = tensor

  return out_tensor


def _param_init(m):
  if isinstance(m, Parameter):
    _glorot_uniform(m.data)
  elif isinstance(m, nn.Linear):
    m.bias.data.zero_()
    _glorot_uniform(m.weight.data)
  elif isinstance(m, nn.Embedding):
    _glorot_uniform(m.weight.data)


def glorot_uniform(m):
  for p in m.modules():
    if isinstance(p, nn.ParameterList) or isinstance(p, nn.ModuleList):
      for pp in p:
        _param_init(pp)
    elif isinstance(p, nn.ParameterDict) or isinstance(p, nn.ModuleDict):
      for key in p:
        _param_init(p[key])
    else:
      _param_init(p)

  for name, p in m.named_parameters():
    if not '.' in name: # top-level parameters
      _param_init(p)

NONLINEARITIES = {
  "tanh": nn.Tanh(),
  "relu": nn.ReLU(),
  "softplus": nn.Softplus(),
  "sigmoid": nn.Sigmoid(),
  "elu": nn.ELU()
}


class MLP(nn.Module):
  def __init__(self, input_dim, hidden_dims, nonlinearity='elu', act_last=None, bn=False, dropout=-1):
    super(MLP, self).__init__()
    self.act_last = act_last
    self.nonlinearity = nonlinearity
    self.input_dim = input_dim
    self.bn = bn

    if isinstance(hidden_dims, str):
      hidden_dims = list(map(int, hidden_dims.split("-")))
    assert len(hidden_dims)
    hidden_dims = [input_dim] + hidden_dims
    self.output_size = hidden_dims[-1]
    
    list_layers = []

    for i in range(1, len(hidden_dims)):
      list_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
      if i + 1 < len(hidden_dims):  # not the last layer
        if self.bn:
          bnorm_layer = nn.BatchNorm1d(hidden_dims[i])
          list_layers.append(bnorm_layer)
        list_layers.append(NONLINEARITIES[self.nonlinearity])
        if dropout > 0:
          list_layers.append(nn.Dropout(dropout))
      else:
        if act_last is not None:
          list_layers.append(NONLINEARITIES[act_last])

    self.main = nn.Sequential(*list_layers)

  def forward(self, z):
    x = self.main(z)
    return x


class CharSeqEncoder(nn.Module):
  def __init__(self, vocab_size, hidden_size):
    super(CharSeqEncoder, self).__init__()
    self.tok_embed = nn.Embedding(vocab_size, hidden_size)
    self.lstm = nn.LSTM(hidden_size, hidden_size, 3, bidirectional=False)

  def forward(self, padded_seq, len_seq):
    packed_seq = pack_padded_sequence(padded_seq, len_seq, enforce_sorted=False)
    tok_embed = self.tok_embed(packed_seq.data)
    packed_input = PackedSequence(data=tok_embed, batch_sizes=packed_seq.batch_sizes,
                    sorted_indices=packed_seq.sorted_indices, unsorted_indices=packed_seq.unsorted_indices)
    with torch.cuda.device(tok_embed.device):
      _, (h, _) = self.lstm(packed_input)
    return h[-1]
