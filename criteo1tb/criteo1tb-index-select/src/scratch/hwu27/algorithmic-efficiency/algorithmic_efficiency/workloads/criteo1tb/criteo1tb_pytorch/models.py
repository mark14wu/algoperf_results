"""Pytorch implementation of DLRM-Small."""

import math

import torch
from torch import nn


class DenseBlock(nn.Module):
  """Dense block with optional residual connection.""" ""

  def __init__(self, module, resnet=False):
    super().__init__()
    self.module = module
    self.resnet = resnet

  def forward(self, x):
    if self.resnet:
      return self.module(x) + x
    else:
      return self.module(x)


class DotInteract(nn.Module):
  """Performs feature interaction operation between dense or sparse features."""

  def __init__(self, num_sparse_features):
    super().__init__()
    self.triu_indices = torch.triu_indices(num_sparse_features + 1,
                                           num_sparse_features + 1)

  def forward(self, dense_features, sparse_features):
    combined_values = torch.cat((dense_features.unsqueeze(1), sparse_features),
                                dim=1)
    interactions = torch.bmm(combined_values,
                             torch.transpose(combined_values, 1, 2))
    interactions_flat = interactions[:,
                                     self.triu_indices[0],
                                     self.triu_indices[1]]
    return torch.cat((dense_features, interactions_flat), dim=1)


class DLRMResNet(nn.Module):
  """Define a DLRM-Small model.

  Parameters:
    vocab_size: vocab size of embedding table.
    num_dense_features: number of dense features as the bottom mlp input.
    mlp_bottom_dims: dimensions of dense layers of the bottom mlp.
    mlp_top_dims: dimensions of dense layers of the top mlp.
    embed_dim: embedding dimension.
  """

  def __init__(self,
               vocab_size,
               num_dense_features=13,
               num_sparse_features=26,
               mlp_bottom_dims=(256, 256, 256),
               mlp_top_dims=(256, 256, 256, 256, 1),
               embed_dim=128,
               dropout_rate=0.0,
               use_layer_norm=False,
               embedding_init_multiplier=None):
    super().__init__()
    self.vocab_size = torch.tensor(vocab_size, dtype=torch.int32)
    self.num_dense_features = num_dense_features
    self.num_sparse_features = num_sparse_features
    self.mlp_bottom_dims = mlp_bottom_dims
    self.mlp_top_dims = mlp_top_dims
    self.embed_dim = embed_dim

    # Ideally, we should use the pooled embedding implementation from
    # `TorchRec`. However, in order to have identical implementation
    # with that of Jax, we define a single embedding matrix.
    num_chunks = 4
    assert vocab_size % num_chunks == 0
    self.embedding_table_chucks = []
    scale = 1.0 / torch.sqrt(self.vocab_size)
    for i in range(num_chunks):
      chunk = nn.Parameter(
          torch.Tensor(self.vocab_size // num_chunks, self.embed_dim))
      chunk.data.uniform_(0, 1)
      chunk.data = scale * chunk.data
      self.register_parameter(f'embedding_chunk_{i}', chunk)
      self.embedding_table_chucks.append(chunk)

    input_dim = self.num_dense_features
    bot_mlp_blocks = []
    for layer_idx, dense_dim in enumerate(self.mlp_bottom_dims):
      block = []
      block.append(nn.Linear(input_dim, dense_dim))
      block.append(nn.ReLU(inplace=True))
      block = nn.Sequential(*block)
      if layer_idx > 0:
        block = DenseBlock(block, resnet=True)
      else:
        block = DenseBlock(block)
      bot_mlp_blocks.append(block)
      input_dim = dense_dim
    self.bot_mlp = nn.Sequential(*bot_mlp_blocks)

    for module in self.bot_mlp.modules():
      if isinstance(module, nn.Linear):
        limit = math.sqrt(6. / (module.in_features + module.out_features))
        nn.init.uniform_(module.weight.data, -limit, limit)
        nn.init.normal_(module.bias.data,
                        0.,
                        math.sqrt(1. / module.out_features))

    # Number of sparse features = 26
    fan_in = (26 * self.embed_dim) + self.mlp_bottom_dims[-1]
    num_layers_top = len(self.mlp_top_dims)
    mlp_top_blocks = []
    for layer_idx, fan_out in enumerate(self.mlp_top_dims):
      block = []
      block.append(nn.Linear(fan_in, fan_out))
      if layer_idx < (num_layers_top - 1):
        block.append(nn.ReLU(inplace=True))
      if (dropout_rate is not None and dropout_rate > 0.0 and
          layer_idx == num_layers_top - 2):
        block.append(nn.Dropout(p=dropout_rate))
      block = nn.Sequential(*block)
      if (layer_idx != 0) and (layer_idx != num_layers_top - 1):
        block = DenseBlock(block, resnet=True)
      else:
        block = DenseBlock(block)
      mlp_top_blocks.append(block)
      fan_in = fan_out
    self.top_mlp = nn.Sequential(*mlp_top_blocks)

    for module in self.top_mlp.modules():
      if isinstance(module, nn.Linear):
        nn.init.normal_(
            module.weight.data,
            0.,
            math.sqrt(2. / (module.in_features + module.out_features)))
        nn.init.normal_(module.bias.data,
                        0.,
                        math.sqrt(1. / module.out_features))

  def forward(self, x):
    batch_size = x.shape[0]

    dense_features, sparse_features = torch.split(
      x, [self.num_dense_features, self.num_sparse_features], 1)

    # Bottom MLP.
    embedded_dense = self.bot_mlp(dense_features)

    # Sparse feature processing.
    sparse_features = sparse_features.to(dtype=torch.int32)
    idx_lookup = torch.reshape(sparse_features, [-1]) % self.vocab_size
    embedding_table = torch.cat(self.embedding_table_chucks, dim=0)
    embedded_sparse = embedding_table[idx_lookup]
    print(idx_lookup)
    exit(-1)
    x = torch.randint(0, 20, [3, 40000], dtype=torch.float32, device='cuda', requires_grad=True)
    bad_index = torch.randint(19999, 20000, (1, 224, 512), dtype=torch.long, device='cuda')
    y = x.index_select(1, bad_index.reshape(-1)).reshape(x.shape[:1] + bad_index.shape)
    # embedded_sparse = embedding_table.
    embedded_sparse = torch.reshape(embedded_sparse,
                                    [batch_size, 26 * self.embed_dim])
    top_mlp_input = torch.cat([embedded_dense, embedded_sparse], axis=1)

    # Final MLP.
    logits = self.top_mlp(top_mlp_input)
    return logits


class DlrmSmall(nn.Module):
  """Define a DLRM-Small model.

  Parameters:
    vocab_size: vocab size of embedding table.
    num_dense_features: number of dense features as the bottom mlp input.
    mlp_bottom_dims: dimensions of dense layers of the bottom mlp.
    mlp_top_dims: dimensions of dense layers of the top mlp.
    embed_dim: embedding dimension.
  """

  def __init__(self,
               vocab_size,
               num_dense_features=13,
               num_sparse_features=26,
               mlp_bottom_dims=(512, 256, 128),
               mlp_top_dims=(1024, 1024, 512, 256, 1),
               embed_dim=128,
               dropout_rate=0.0,
               use_layer_norm=False,
               embedding_init_multiplier=None):
    super().__init__()
    self.vocab_size = torch.tensor(vocab_size, dtype=torch.int32)
    self.num_dense_features = num_dense_features
    self.num_sparse_features = num_sparse_features
    self.mlp_bottom_dims = mlp_bottom_dims
    self.mlp_top_dims = mlp_top_dims
    self.embed_dim = embed_dim
    self.embedding_init_multiplier = embedding_init_multiplier

    # Ideally, we should use the pooled embedding implementation from
    # `TorchRec`. However, in order to have identical implementation
    # with that of Jax, we define a single embedding matrix.
    num_chunks = 4
    assert vocab_size % num_chunks == 0
    self.embedding_table_chucks = []

    if self.embedding_init_multiplier is None:
      scale = 1.0 / torch.sqrt(self.vocab_size)
    else:
      scale = self.embedding_init_multiplier

    for i in range(num_chunks):
      chunk = nn.Parameter(
          torch.Tensor(self.vocab_size // num_chunks, self.embed_dim))
      chunk.data.uniform_(0, 1)
      chunk.data = scale * chunk.data
      self.register_parameter(f'embedding_chunk_{i}', chunk)
      self.embedding_table_chucks.append(chunk)

    input_dim = self.num_dense_features
    bottom_mlp_layers = []
    for dense_dim in self.mlp_bottom_dims:
      bottom_mlp_layers.append(nn.Linear(input_dim, dense_dim))
      bottom_mlp_layers.append(nn.ReLU(inplace=True))
      if use_layer_norm:
        bottom_mlp_layers.append(nn.LayerNorm(dense_dim, eps=1e-6))
      input_dim = dense_dim
    self.bot_mlp = nn.Sequential(*bottom_mlp_layers)
    for module in self.bot_mlp.modules():
      if isinstance(module, nn.Linear):
        limit = math.sqrt(6. / (module.in_features + module.out_features))
        nn.init.uniform_(module.weight.data, -limit, limit)
        nn.init.normal_(module.bias.data,
                        0.,
                        math.sqrt(1. / module.out_features))

    self.dot_interact = DotInteract(num_sparse_features=num_sparse_features,)

    # TODO: Write down the formula here instead of the constant.
    input_dims = 506
    num_layers_top = len(self.mlp_top_dims)
    top_mlp_layers = []
    for layer_idx, fan_out in enumerate(self.mlp_top_dims):
      fan_in = input_dims if layer_idx == 0 \
          else self.mlp_top_dims[layer_idx - 1]
      top_mlp_layers.append(nn.Linear(fan_in, fan_out))
      if layer_idx < (num_layers_top - 1):
        top_mlp_layers.append(nn.ReLU(inplace=True))
        if use_layer_norm:
          top_mlp_layers.append(nn.LayerNorm(fan_out, eps=1e-6))
      if (dropout_rate is not None and dropout_rate > 0.0 and
          layer_idx == num_layers_top - 2):
        top_mlp_layers.append(nn.Dropout(p=dropout_rate))
    self.top_mlp = nn.Sequential(*top_mlp_layers)
    if use_layer_norm:
      self.embed_ln = nn.LayerNorm(self.embed_dim, eps=1e-6)
    else:
      self.embed_ln = None
    for module in self.top_mlp.modules():
      if isinstance(module, nn.Linear):
        nn.init.normal_(
            module.weight.data,
            0.,
            math.sqrt(2. / (module.in_features + module.out_features)))
        nn.init.normal_(module.bias.data,
                        0.,
                        math.sqrt(1. / module.out_features))

  def forward(self, x):
    batch_size = x.shape[0]

    dense_features, sparse_features = torch.split(
      x, [self.num_dense_features, self.num_sparse_features], 1)

    # Bottom MLP.
    embedded_dense = self.bot_mlp(dense_features)

    # Sparse feature processing.
    sparse_features = sparse_features.to(dtype=torch.int32)
    idx_lookup = torch.reshape(sparse_features, [-1]) % self.vocab_size
    embedding_table = torch.cat(self.embedding_table_chucks, dim=0)
    # embedded_sparse = embedding_table[idx_lookup]
    # print('dlrm_small:', idx_lookup)
    embedded_sparse = embedding_table.index_select(0, idx_lookup)
    embedded_sparse = torch.reshape(embedded_sparse,
                                    [batch_size, -1, self.embed_dim])
    if self.embed_ln:
      embedded_sparse = self.embed_ln(embedded_sparse)
    # Dot product interactions.
    concatenated_dense = self.dot_interact(
        dense_features=embedded_dense, sparse_features=embedded_sparse)

    # Final MLP.
    logits = self.top_mlp(concatenated_dense)
    return logits
