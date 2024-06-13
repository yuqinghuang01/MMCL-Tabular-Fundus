from typing import Dict
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, global_mean_pool


class GraphEncoder(nn.Module):
  """
  Main contrastive model used in SCARF. Consists of an encoder that takes the input and 
  creates an embedding of size {args.embedding_dim}.
  Also supports providing a checkpoint with trained weights to be loaded.
  """
  def __init__(self, args) -> None:
    super(GraphEncoder, self).__init__()
    self.args = args
    
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.hidden_sizes = self.args.hidden_sizes
    self.heads = self.args.heads
    self.num_node_features = len(self.args.node_features_name)
    self.num_edge_features = len(self.args.edge_features_name)
    # 3 GAT layers
    self.conv1 = GATv2Conv(self.num_node_features, self.hidden_sizes[0], heads=self.heads[0], edge_dim=self.num_edge_features).to(self.device)
    self.conv2 = GATv2Conv(self.hidden_sizes[0]*self.heads[0], self.hidden_sizes[1], heads=self.heads[1], edge_dim=self.num_edge_features).to(self.device)
    self.conv3 = GATv2Conv(self.hidden_sizes[1]*self.heads[1], self.hidden_sizes[2], heads=self.heads[2], edge_dim=self.num_edge_features).to(self.device)

  def forward(self, data) -> torch.Tensor:
    """
    Passes input through encoder and projector. 
    Output is ready for loss calculation.
    """
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    
    x = self.conv1(x, edge_index, edge_attr=edge_attr) # adding edge features here!
    x = F.relu(x)
    x = F.dropout(x, training=self.training)
    
    x = self.conv2(x, edge_index, edge_attr=edge_attr) # adding edge features here!
    x = F.relu(x)
    x = F.dropout(x, training=self.training)
    
    x = self.conv3(x, edge_index, edge_attr=edge_attr)
    
    x = global_mean_pool(x, data.batch)
    return x