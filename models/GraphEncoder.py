from typing import Dict
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, GINConv, SAGEConv, EdgeConv, global_mean_pool
from torch_geometric.transforms import LineGraph


class GraphEncoder(nn.Module):
  """
  Main contrastive model used in SCARF. Consists of an encoder that takes the input and 
  creates an embedding of size {args.embedding_dim}.
  Also supports providing a checkpoint with trained weights to be loaded.
  """
  def __init__(self, args, conv1d_channels=[16, 32], conv1d_kws=[512, 5]) -> None:
    super(GraphEncoder, self).__init__()
    self.args = args
    
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.hidden_sizes = self.args.hidden_sizes
    self.heads = self.args.heads
    self.num_node_features = len(self.args.node_features_name)
    self.num_edge_features = len(self.args.edge_features_name)
    # 3 GAT layers
    if self.args.gnn_layer_type == 'GATConv':
      self.conv1 = GATv2Conv(self.num_node_features, self.hidden_sizes[0], heads=self.heads[0], edge_dim=self.num_edge_features).to(self.device)
      self.conv2 = GATv2Conv(self.hidden_sizes[0]*self.heads[0], self.hidden_sizes[1], heads=self.heads[1], edge_dim=self.num_edge_features).to(self.device)
      self.conv3 = GATv2Conv(self.hidden_sizes[1]*self.heads[1], self.hidden_sizes[2], heads=self.heads[2], edge_dim=self.num_edge_features).to(self.device)
    elif self.args.gnn_layer_type == 'GINConv':
      self.conv1 = GINConv(nn.Sequential(nn.Linear(self.num_node_features, self.hidden_sizes[0]*self.heads[0]), nn.ReLU())).to(self.device)
      self.conv2 = GINConv(nn.Sequential(nn.Linear(self.hidden_sizes[0]*self.heads[0], self.hidden_sizes[1]*self.heads[1]), nn.ReLU())).to(self.device)
      self.conv3 = GINConv(nn.Sequential(nn.Linear(self.hidden_sizes[1]*self.heads[1], self.args.graph_pooled_dim), nn.ReLU())).to(self.device)
    elif self.args.gnn_layer_type == 'GCNConv':
      self.conv1 = GCNConv(self.num_node_features, self.hidden_sizes[0]*self.heads[0]).to(self.device)
      self.conv2 = GCNConv(self.hidden_sizes[0]*self.heads[0], self.hidden_sizes[1]*self.heads[1]).to(self.device)
      self.conv3 = GCNConv(self.hidden_sizes[1]*self.heads[1], self.args.graph_pooled_dim).to(self.device)
    elif self.args.gnn_layer_type == 'SAGEConv':
      self.conv1 = SAGEConv(self.num_node_features, self.hidden_sizes[0]*self.heads[0]).to(self.device)
      self.conv2 = SAGEConv(self.hidden_sizes[0]*self.heads[0], self.hidden_sizes[1]*self.heads[1]).to(self.device)
      self.conv3 = SAGEConv(self.hidden_sizes[1]*self.heads[1], self.args.graph_pooled_dim).to(self.device)
    elif self.args.gnn_layer_type == 'EdgeConv':
      self.conv1 = EdgeConv(nn.Sequential(nn.Linear(2*self.num_node_features, self.hidden_sizes[0]*self.heads[0]), nn.ReLU())).to(self.device)
      self.conv2 = EdgeConv(nn.Sequential(nn.Linear(2*self.hidden_sizes[0]*self.heads[0], self.hidden_sizes[1]*self.heads[1]), nn.ReLU())).to(self.device)
      self.conv3 = EdgeConv(nn.Sequential(nn.Linear(2*self.hidden_sizes[1]*self.heads[1], self.args.graph_pooled_dim), nn.ReLU())).to(self.device)
    else:
      raise Exception('graph layer type must be one of [GATConv, GINConv, GCNConv, SAGEConv, EdgeConv]')
    
    if self.args.with_lg:
      # Line graph
      self.line_graph = LineGraph()
      if self.args.gnn_layer_type == 'GATConv':
        self.l_conv1 = GATv2Conv(self.num_edge_features, self.hidden_sizes[0], heads=self.heads[0]).to(self.device)
        self.l_conv2 = GATv2Conv(self.hidden_sizes[0]*self.heads[0], self.hidden_sizes[1], heads=self.heads[1]).to(self.device)
        self.l_conv3 = GATv2Conv(self.hidden_sizes[1]*self.heads[1], self.hidden_sizes[2], heads=self.heads[2]).to(self.device)
      elif self.args.gnn_layer_type == 'GINConv':
        self.l_conv1 = GINConv(nn.Sequential(nn.Linear(self.num_edge_features, self.hidden_sizes[0]*self.heads[0]), nn.ReLU())).to(self.device)
        self.l_conv2 = GINConv(nn.Sequential(nn.Linear(self.hidden_sizes[0]*self.heads[0], self.hidden_sizes[1]*self.heads[1]), nn.ReLU())).to(self.device)
        self.l_conv3 = GINConv(nn.Sequential(nn.Linear(self.hidden_sizes[1]*self.heads[1], self.args.graph_pooled_dim), nn.ReLU())).to(self.device)
      elif self.args.gnn_layer_type == 'GCNConv':
        self.l_conv1 = GCNConv(self.num_edge_features, self.hidden_sizes[0]*self.heads[0]).to(self.device)
        self.l_conv2 = GCNConv(self.hidden_sizes[0]*self.heads[0], self.hidden_sizes[1]*self.heads[1]).to(self.device)
        self.l_conv3 = GCNConv(self.hidden_sizes[1]*self.heads[1], self.args.graph_pooled_dim).to(self.device)
      elif self.args.gnn_layer_type == 'SAGEConv':
        self.l_conv1 = SAGEConv(self.num_edge_features, self.hidden_sizes[0]*self.heads[0]).to(self.device)
        self.l_conv2 = SAGEConv(self.hidden_sizes[0]*self.heads[0], self.hidden_sizes[1]*self.heads[1]).to(self.device)
        self.l_conv3 = SAGEConv(self.hidden_sizes[1]*self.heads[1], self.args.graph_pooled_dim).to(self.device)
      elif self.args.gnn_layer_type == 'EdgeConv':
        self.l_conv1 = EdgeConv(nn.Sequential(nn.Linear(2*self.num_edge_features, self.hidden_sizes[0]*self.heads[0]), nn.ReLU())).to(self.device)
        self.l_conv2 = EdgeConv(nn.Sequential(nn.Linear(2*self.hidden_sizes[0]*self.heads[0], self.hidden_sizes[1]*self.heads[1]), nn.ReLU())).to(self.device)
        self.l_conv3 = EdgeConv(nn.Sequential(nn.Linear(2*self.hidden_sizes[1]*self.heads[1], self.args.graph_pooled_dim), nn.ReLU())).to(self.device)
      else:
        raise Exception('graph layer type must be one of [GATConv, GINConv, GCNConv, SAGEConv, EdgeConv]')

    if self.args.with_sp:
      # For sortpooling of the original graph
      self.k = self.args.k
      self.conv1d_params1 = nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0]).to(self.device)
      self.maxpool1d = nn.MaxPool1d(2, 2).to(self.device)
      self.conv1d_params2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1).to(self.device)
      self.conv1d_activation = eval('nn.{}()'.format('ReLU')).to(self.device)
      dense_dim = int((self.k - 2) / 2 + 1)
      self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
      self.out_params = nn.Linear(self.dense_dim, self.args.graph_pooled_dim).to(self.device)
      if self.args.with_lg:
        # For sortpooling of the line graph
        self.l_conv1d_params1 = nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0]).to(self.device)
        self.l_conv1d_params2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1).to(self.device)
        self.l_conv1d_activation = eval('nn.{}()'.format('ReLU')).to(self.device)
        self.l_out_params = nn.Linear(self.dense_dim, self.args.graph_pooled_dim).to(self.device)
  
  def forward_graph(self, data, batch_size, line_graph=False) -> torch.Tensor:
    """
    Passes a graph through encoder. 
    Output is ready for loss calculation.
    """
    if (not self.args.with_lg) or (not line_graph):
      x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

      if self.args.gnn_layer_type == 'GATConv':
        x = self.conv1(x, edge_index, edge_attr=edge_attr) # adding edge features here!
      elif self.args.gnn_layer_type in ['GCNConv', 'GINConv', 'SAGEConv', 'EdgeConv']:
        x = self.conv1(x, edge_index)
      x = F.relu(x)
      x = F.dropout(x, training=self.training)

      if self.args.gnn_layer_type == 'GATConv':
        x = self.conv2(x, edge_index, edge_attr=edge_attr) # adding edge features here!
      elif self.args.gnn_layer_type in ['GCNConv', 'GINConv', 'SAGEConv', 'EdgeConv']:
        x = self.conv2(x, edge_index)
      x = F.relu(x)
      x = F.dropout(x, training=self.training)

      if self.args.gnn_layer_type == 'GATConv':
        x = self.conv3(x, edge_index, edge_attr=edge_attr) # adding edge features here!
      elif self.args.gnn_layer_type in ['GCNConv', 'GINConv', 'SAGEConv', 'EdgeConv']:
        x = self.conv3(x, edge_index)
    else:
      x, edge_index = data.x, data.edge_index
      x = self.l_conv1(x, edge_index)
      x = F.relu(x)
      x = F.dropout(x, training=self.training)

      x = self.l_conv2(x, edge_index)
      x = F.relu(x)
      x = F.dropout(x, training=self.training)

      x = self.l_conv3(x, edge_index)
    
    '''
    First pooling method
    '''
    if not self.args.with_sp:
      x = global_mean_pool(x, data.batch) # Size: torch.Size([64, 512])
      return x
    # print(data.batch.size()) # Size: torch.Size([8753])
    # print(data.batch) # tensor([ 0,  0,  0,  ..., 63, 63, 63], device='cuda:0')

    '''
    Second pooling method
    '''
    if line_graph:
      x = self.sortpooling_embedding(data, x, batch_size)
      x = x.view((-1, 1, self.k * x.size()[-1]))
      x = self.l_conv1d_params1(x)
      x = self.l_conv1d_activation(x)
      x = self.maxpool1d(x)
      x = self.l_conv1d_params2(x)
      x = self.l_conv1d_activation(x)
      x = x.view(batch_size, -1)
      # print(x.size()) #torch.Size([64, 192])
      x = self.l_out_params(x)
      x = self.l_conv1d_activation(x)
      # print(x.size()) #torch.Size([64, 512])
    else:
      x = self.sortpooling_embedding(data, x, batch_size)
      x = x.view((-1, 1, self.k * x.size()[-1]))
      x = self.conv1d_params1(x)
      x = self.conv1d_activation(x)
      x = self.maxpool1d(x)
      x = self.conv1d_params2(x)
      x = self.conv1d_activation(x)
      x = x.view(batch_size, -1)
      x = self.out_params(x)
      x = self.conv1d_activation(x)
    return x
  
  def forward(self, data) -> torch.Tensor:
    """
    Passes input through encoder and projector. 
    Output is ready for loss calculation.
    """
    # print(data.is_directed()) # True
    embed = self.forward_graph(data, data.batch_size)
    
    if not self.args.with_lg:
      return embed
    
    # Line graph
    #print(data.num_nodes)
    l_batch = torch.zeros(data.num_edges, dtype=torch.int64).to(self.device)
    idx = 0
    for i in range(data.batch_size):
      l_batch[i:i+data.get_example(i).num_edges] = i
      i += data.get_example(0).num_edges
    l_data = Data(x=data.l_x, edge_index=data.l_edge_index)
    l_data.batch = l_batch
    
    l_embed = self.forward_graph(l_data, data.batch_size, True)
    ret = torch.cat((embed, l_embed), 1)
    #print(ret.size()) #torch.Size([64, 1024])

    return ret
  
  def sortpooling_embedding(self, data, feat, batch_size) -> torch.Tensor:
    """
    Compute sort pooling.
    
    Sort Pooling from `An End-to-End Deep Learning Architecture for Graph Classification
    <https://www.cse.wustl.edu/~ychen/public/DGCNN.pdf>`__

    It first sorts the node features in ascending order along the feature dimension,
    and selects the sorted features of top-k nodes (ranked by the largest value of each node).
    
    Parameters
    ----------
    feat : torch.Tensor
        The input node feature with shape :math:`(N, D)`, where :math:`N` is the
        number of nodes in the graph, and :math:`D` means the size of features.

    Returns
    -------
    torch.Tensor
        The output feature with shape :math:`(B, k, D)`, where :math:`B` refers
        to the batch size of input graphs.
    """
    feat, _ = feat.sort(dim=-1)
    # Sort nodes according to their last features (note the batch size)
    ret = torch.zeros(batch_size, self.k, feat.size()[-1]).to(self.device)
    for i in range(batch_size):
      cur_feat = feat[data.batch == i]
      cur_feat_flip = torch.flip(cur_feat, dims=[1])
      cur_feat_flip, indices = torch.sort(cur_feat_flip, dim=0)
      feat_sorted = torch.index_select(cur_feat, 0, indices[:,0])
      if cur_feat.size()[0] >= self.k:
        ret[i] = feat_sorted[0:self.k,]
      else:
        to_pad = torch.zeros(self.k-cur_feat.size()[0], feat.size()[-1]).to(self.device)
        feat_sorted = torch.cat((feat_sorted, to_pad), 0)
        ret[i] = feat_sorted
    return ret