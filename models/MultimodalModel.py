import torch
import torch.nn as nn
from collections import OrderedDict

from models.TabularModel import TabularModel
from models.ImagingModel import ImagingModel
from models.GraphModel import GraphModel

class MultimodalModel(nn.Module):
  """
  Evaluation model for imaging and tabular data.
  """
  def __init__(self, args) -> None:
    super(MultimodalModel, self).__init__()
    
    self.datatype = args.datatype
    self.with_lg = args.with_lg
    self.embedding_dim = args.embedding_dim
    self.graph_pooled_dim = args.graph_pooled_dim

    if self.datatype == 'imaging_tabular':
      self.imaging_model = ImagingModel(args)
      in_dim = 3072
    if self.datatype == 'graph_tabular':
      self.graph_model = GraphModel(args)
      if self.with_lg:
        in_dim = self.embedding_dim + 2*self.graph_pooled_dim
      else:
        in_dim = self.embedding_dim + self.graph_pooled_dim
    
    self.tabular_model = TabularModel(args)
    self.head = nn.Linear(in_dim, args.num_classes)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.datatype == 'imaging_tabular':
      x_im = self.imaging_model.encoder(x[0])[0]
      x_tab = self.tabular_model.encoder(x[1]).squeeze()
    if self.datatype == 'graph_tabular':
      x_im = self.graph_model.encoder(x).squeeze()
      x_tab = self.tabular_model.encoder(x.tabular).squeeze()
    x = torch.cat([x_im, x_tab], dim=1)
    x = self.head(x)
    return x