import numpy as np
import torch
import torch.nn as nn

from crossbeam.model.op_arg import LSTMArgSelector
from crossbeam.model.op_init import OpPoolingState
from crossbeam.model.great import Great
from crossbeam.model.encoder import DummyWeightEncoder, ValueWeightEncoder


class LogicModel(nn.Module):
  def __init__(self, args, operations, max_entities=10):
    super(LogicModel, self).__init__()

    self.max_entities = max_entities

    self.op_in_beam = args.op_in_beam
    assert not self.op_in_beam

    self.arg = LSTMArgSelector(hidden_size=args.embed_dim,
                               mlp_sizes=[256, 1],
                               step_score_func=args.step_score_func,
                               step_score_normalize=args.score_normed)
    self.init = OpPoolingState(ops=tuple(operations), state_dim=args.embed_dim, pool_method='mean')
    if args.encode_weight:
      self.encode_weight = ValueWeightEncoder(hidden_size=args.embed_dim)
      print('encode weight')
    else:
      self.encode_weight = DummyWeightEncoder()
    if args.great_transformer:
      print('use great transformer')
      self.entity_project = nn.Embedding(max_entities, args.embed_dim)

      # relations:
      # unary both no
      # unary first yes
      # unary second yes    
      # unary both yes    
      # binary no
      # binary ->
      # binary <-
      # binary <->
      # also applies for spec (2x)
      self.relation_project = nn.Embedding(8+8, args.embed_dim)

      self.great = Great(d_model=args.embed_dim,
                         dim_feedforward=args.embed_dim*4,
                         layers=4,
                         batch_first=True)

      self.final_projection = nn.Linear(max_entities*args.embed_dim,args.embed_dim)
    else:
      print('use mlp')
      self.great = None
      self.embed_spec,self.embed_value,self.embed_input = \
                      [ nn.Sequential(nn.Linear(max_entities*max_entities + 1, args.embed_dim),
                                     nn.ReLU(),
                                     nn.Linear(args.embed_dim, args.embed_dim),
                                     nn.ReLU(),
                                     nn.Linear(args.embed_dim, args.embed_dim),
                                     nn.ReLU(),
                                     nn.Linear(args.embed_dim, args.embed_dim))
                        for _ in range(3) ]

  def batch_init(self, io_embed, io_scatter, val_embed, value_indices, operation, sample_indices=None, io_gather=None):
    return self.init.batch_forward(io_embed, io_scatter, val_embed, value_indices, operation, sample_indices, io_gather)

  @staticmethod
  def serialize_relation(r):
    if len(r.shape) == 1:
      return [0] + list(1*r) + [0]*(r.shape[0]*r.shape[0] - r.shape[0])
    elif len(r.shape) == 2:
      return [1] + list(1*np.reshape(r,-1))
    else:
      assert False, "only handle relations with 1/2 indices"

  def features_of_relation(self, relations, is_specification, device=None):
    if self.great is None:
      x = torch.tensor([LogicModel.serialize_relation(v) for v in relations]).float()
      if device: x = x.to(device)
      if is_specification:
        return self.embed_spec(x)
      else:
        return self.embed_input(x)

    
    x = self.entity_project(torch.arange(self.max_entities,device=device).long())
    x = x.unsqueeze(0).repeat(len(relations),1,1)

    d = {(False,False): 0,
         (False,True): 1,
         (True,False): 2,
         (True,True): 3}

    def o(a):
      nonlocal is_specification
      h = {1:0,2:4}[len(a.shape)]
      if is_specification:
        return h+8
      return h

    def I(matrix,a,b):
      if len(matrix.shape) == 2:
        return matrix[a,b]
      if len(matrix.shape) == 1:
        return matrix[a]
      assert False

    r = [ [ [ o(m) + d[I(m,i,j), I(m,j,i)]
              for j in range(self.max_entities)]
            for i in range(self.max_entities)]
          for m in relations ]
    
    r = self.relation_project(torch.tensor(r).long().to(x.device))

    output = self.great(x,r).view(len(relations),-1)
    output = self.final_projection(output)
    return output

  def io(self, list_input_dictionary, list_outputs, device, needs_scatter_idx=False):
    """input_dictionary/outputs: list of length batch_size, batching over task
    Each element of outputs is the output for a particular I/O example
    Here we only ever have one I/O example
    
    """
    list_feat = []
    for input_dictionary, outputs in zip(list_input_dictionary, list_outputs):
      assert len(outputs) == 1
      outputs = outputs[0]

      specification = self.features_of_relation([outputs], True, device)
    
      values = self.val(list(input_dictionary.values()),
                        device)

      values = values.max(0).values.unsqueeze(0)

      feat = torch.cat((specification,values),-1)
      list_feat.append(feat)
    feats = torch.cat(list_feat, dim=0)
    if needs_scatter_idx:
      idx = torch.arange(len(list_input_dictionary)).to(device)
      return feats, idx
    else:
      return feats

  def val(self, all_values, device, output_values=None):
    """all_values: list of values. each value is represented by its instantiation on each I/O example
    so if you have three I/O examples and ten values then you will have 10x3 matrix as input
    returns as output [number_of_values,embedding_size]"""

    all_values = [v[0] for v in all_values]
    return self.features_of_relation(all_values, False, device)
