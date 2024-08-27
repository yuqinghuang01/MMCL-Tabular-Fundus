from typing import Dict
from collections import OrderedDict

import torch
import torch.nn as nn

from models.GraphEncoder import GraphEncoder

class GraphModel(nn.Module):
  """
  Evaluation model for graph trained with MLP backbone.
  """
  def __init__(self, args):
    super(GraphModel, self).__init__()

    self.encoder = GraphEncoder(args)

    if args.checkpoint:
      # Load weights
      checkpoint = torch.load(args.checkpoint)
      original_args = checkpoint['hyper_parameters']
      state_dict = checkpoint['state_dict']

      self.encoder_name = 'encoder_graph.'
      self.encoder_name2 = 'model.graph_model.encoder.'
      
      # Remove prefix and fc layers
      state_dict_encoder = {}
      for k in list(state_dict.keys()):
        if k.startswith(self.encoder_name) and not 'projection_head' in k and not 'prototypes' in k:
          state_dict_encoder[k[len(self.encoder_name):]] = state_dict[k]
        if k.startswith(self.encoder_name2) and not 'projection_head' in k and not 'prototypes' in k:
          state_dict_encoder[k[len(self.encoder_name2):]] = state_dict[k]
      
      log = self.encoder.load_state_dict(state_dict_encoder, strict=True)
      assert len(log.missing_keys) == 0

      # Freeze if needed
      if args.finetune_strategy == 'frozen':
        for _, param in self.encoder.named_parameters():
          param.requires_grad = False
        parameters = list(filter(lambda p: p.requires_grad, self.encoder.parameters()))
        assert len(parameters)==0
    
    if args.with_lg:
      self.classifier = nn.Linear(2*args.graph_pooled_dim, args.num_classes)
    else:
      self.classifier = nn.Linear(args.graph_pooled_dim, args.num_classes)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.encoder(x)
    x = self.classifier(x)
    return x
    