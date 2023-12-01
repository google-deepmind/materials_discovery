# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Any, Callable, Iterable, Optional
from typing import Dict, Tuple, Union

import flax.linen as nn
from jax import ops
from jax import vmap
from jax.nn import initializers
import jax.numpy as jnp
from jax_md import space
import jraph
from ml_collections import ConfigDict
import numpy as onp

from . import gnn


Array = jnp.ndarray
PRNGKey = Array
Shape = Iterable[int]
Dtype = Any
InitializerFn = Callable[[PRNGKey, Shape, Dtype], Array]
UnaryFn = Callable[[Array], Array]
GraphsTuple = jraph.GraphsTuple


def segment_normalized(normalization):
  def normalized_sum(*args, **kwargs):
    return ops.segment_sum(*args, **kwargs) / normalization

  return normalized_sum


# Average coordination number computed from the materials project data.
AVERAGE_NODE_COORDINATION = 9
AVERAGE_EDGE_COORDINATION = 17
AGGREGATION = {
    'coordination': segment_normalized(AVERAGE_EDGE_COORDINATION),
    'mean': jraph.segment_mean,
    'sum': ops.segment_sum,
}


class BetaSwish(nn.Module):

  @nn.compact
  def __call__(self, x):
    features = x.shape[-1]
    beta = self.param('Beta', nn.initializers.ones, (features,))
    return x * nn.sigmoid(beta * x)


NONLINEARITY = {
    'none': lambda x: x,
    'relu': nn.relu,
    'swish': lambda x: BetaSwish()(x),
    'raw_swish': nn.swish,
    'tanh': nn.tanh,
    'sigmoid': nn.sigmoid,
    'silu': nn.silu,
}


normal = lambda var: initializers.variance_scaling(var, 'fan_in', 'normal')


INITIALIZER = {
    'none': initializers.orthogonal(onp.sqrt(1.0)),
    'relu': normal(2.0),
    'swish': initializers.orthogonal(onp.sqrt(2.75)),
    'raw_swish': initializers.orthogonal(onp.sqrt(2.75)),
    'tanh': normal(1.0),
}


def get_nonlinearity_by_name(name: str) -> Tuple[UnaryFn, InitializerFn]:
  if name in NONLINEARITY:
    return NONLINEARITY[name], INITIALIZER[name]
  raise ValueError(f'Nonlinearity "{name}" not found.')


class MLP(nn.Module):
  features: Tuple[int, ...]
  nonlinearity: str

  use_bias: bool = True
  scalar_mlp_std: Optional[float] = None

  @nn.compact
  def __call__(self, x):
    features = self.features

    dense = partial(nn.Dense, use_bias=self.use_bias)
    phi, init_fn = get_nonlinearity_by_name(self.nonlinearity)

    if self.scalar_mlp_std is not None:
      kernel_init = normal(self.scalar_mlp_std)
    else:
      kernel_init = init_fn

    for h in features[:-1]:
      x = phi(dense(h, kernel_init=kernel_init)(x))
    return dense(features[-1], kernel_init=normal(1.0))(x)


def mlp(
    hidden_features: Union[int, Tuple[int, ...]], nonlinearity: str, **kwargs
) -> Callable[..., Array]:

  if isinstance(hidden_features, int):
    hidden_features = (hidden_features,)

  def mlp_fn(*args):
    fn = MLP(hidden_features, nonlinearity, **kwargs)
    return jraph.concatenated_args(fn)(*args)

  return mlp_fn


FeaturizerFn = Callable[
    [GraphsTuple, Array, Array, Optional[Array]], GraphsTuple
]


def graph_featurizer(
    atom_feature_fn: Callable[[GraphsTuple, Array], Array],
    edge_feature_fn: Callable[[Array], Array],
    global_feature_fn: Callable[[Array], Array],
) -> FeaturizerFn:
  def featurize(
      graph: GraphsTuple,
      position: jnp.ndarray,
      box: jnp.ndarray,
      box_perturbation: jnp.ndarray,
  ) -> GraphsTuple:
    edge_features = None
    dR = None
    if position is not None:
      Rb = position[graph.senders]
      Ra = position[graph.receivers]
      if box.ndim == 4:
        box = box[:, 0, :, :]
      node_box = jnp.repeat(
          box, graph.n_node, axis=0, total_repeat_length=len(position)
      )
      edge_box = node_box[graph.senders]
      _, Ts = graph.edges
      dR = vmap(space.transform)(edge_box, Ra - Rb - Ts)
      if box_perturbation is not None:
        box_perturbation = jnp.repeat(
            box_perturbation,
            graph.n_node,
            axis=0,
            total_repeat_length=len(position),
        )
        box_perturbation = box_perturbation[graph.senders]
        dR = jnp.einsum('nij,nj->ni', box_perturbation, dR)
      edge_features = edge_feature_fn(dR)
    return GraphsTuple(
        n_node=graph.n_node,
        n_edge=graph.n_edge,
        nodes=atom_feature_fn(graph, position),
        edges=edge_features,
        senders=graph.senders,
        receivers=graph.receivers,
        globals=global_feature_fn(jnp.zeros([graph.n_node.shape[0], 1])),
    )

  return featurize


@partial(jnp.vectorize, signature='(f),(d)->(f)')
def gaussian_edge_features(r_0: Array, dR: Array) -> Array:
  dr = space.distance(dR)
  return jnp.exp(-((dr - r_0) ** 2) / 0.5**2)


def gaussian_features(r_0: Array) -> FeaturizerFn:
  return graph_featurizer(
      lambda graph, dR: graph.nodes,
      partial(gaussian_edge_features, r_0),
      lambda g: g,
  )


def standard_gaussian_features() -> FeaturizerFn:
  return gaussian_features(jnp.linspace(0.05, 4, 30))


class CrystalEnergyModel(nn.Module):
  graph_net_steps: int
  mlp_width: Tuple[int]
  mlp_nonlinearity: Union[str, Dict[str, str]]
  embedding_dim: int

  featurizer: str

  shift: float
  scale: float

  feature_band_limit: Optional[int] = 0
  conditioning_band_limit: Optional[int] = 0
  extra_scalars_for_gating: Optional[bool] = False

  residual: Optional[str] = None
  aggregate_edges_for_nodes_fn: Callable = ops.segment_sum
  aggregate_edges_for_globals_fn: Callable = jraph.segment_mean
  readout_aggregate_edges_for_globals_fn: Optional[Callable] = None
  normalize_edges_for_globals_by_nodes: bool = False

  @nn.compact
  def __call__(self, graph, positions, box, box_perturbation=None):
    is_compositional = positions is None
    nonlinearity = self.mlp_nonlinearity

    featurizer_fn = standard_gaussian_features()
    graph = featurizer_fn(graph, positions, box, box_perturbation)

    mlp_fn = partial(mlp, self.mlp_width, nonlinearity)
    edge_update_fn = node_update_fn = global_update_fn = mlp_fn
    node_embed_fn = nn.Dense(self.embedding_dim, name='Node Embedding')
    edge_embed_fn = nn.Dense(self.embedding_dim, name='Edge Embedding')
    global_embed_fn = nn.Dense(self.embedding_dim, name='Global Embedding')

    readout_width = self.mlp_width[:-1] + (1,)
    node_readout_fn = node_update_fn
    edge_readout_fn = edge_update_fn
    global_readout_fn = mlp(readout_width, nonlinearity, name='Readout')

    if is_compositional:
      edge_update_fn = lambda name: None
      edge_embed_fn = None
      edge_readout_fn = lambda name: None

    embed = jraph.GraphMapFeatures(
        embed_node_fn=node_embed_fn,
        embed_edge_fn=edge_embed_fn,
        embed_global_fn=global_embed_fn,
    )

    def step(i, graph):
      return gnn.GraphNetwork(
          update_node_fn=node_update_fn(name=f'Node Update {i}'),
          update_edge_fn=edge_update_fn(name=f'Edge Update {i}'),
          update_global_fn=global_update_fn(name=f'Global Update {i}'),
          aggregate_edges_for_nodes_fn=self.aggregate_edges_for_nodes_fn,
          aggregate_nodes_for_globals_fn=jraph.segment_mean,
          aggregate_edges_for_globals_fn=self.aggregate_edges_for_globals_fn,
          normalize_edges_for_globals_by_nodes=self.normalize_edges_for_globals_by_nodes,
      )(graph)

    readout = gnn.GraphNetwork(
        update_node_fn=node_readout_fn(name='Node Readout'),
        update_edge_fn=edge_readout_fn(name='Edge Readout'),
        update_global_fn=global_readout_fn,
        aggregate_edges_for_nodes_fn=self.aggregate_edges_for_nodes_fn,
        aggregate_nodes_for_globals_fn=jraph.segment_mean,
        aggregate_edges_for_globals_fn=jraph.segment_mean,
        normalize_edges_for_globals_by_nodes=self.normalize_edges_for_globals_by_nodes,
    )

    embedding = embed(graph)
    output = embedding

    for i in range(self.graph_net_steps - 1):
      res = embedding
      output = step(i, output)
      if self.residual:
        output = add_residual(res, output, self.residual)

    n_node = 1.0 if is_compositional else graph.n_node[:, None]
    output = readout(output).globals

    if self.feature_band_limit > 0:
      output = [o.array for o in output if o.p == 1]
      assert len(output) == 1
      output = output[0][:, :, 0]

    return n_node * (self.scale * output + self.shift)


def add_residual(res: GraphsTuple, x: GraphsTuple, style: str) -> GraphsTuple:
  if style == 'none':
    return x
  raise ValueError(f'Unkown residual connection of type {style}.')


def crystal_energy_model(cfg: ConfigDict) -> CrystalEnergyModel:
  """Define crystal energy model used in GNoME experiments."""
  node_aggregation_fn = AGGREGATION[cfg.node_aggregation]
  edges_for_globals_aggregation_fn = AGGREGATION[
      cfg.edges_for_globals_aggregation
  ]
  normalize_edges_for_globals_by_nodes = (
      cfg.edges_for_globals_aggregation == 'coordination'
  )
  readout_edges_for_globals_aggregation_fn = AGGREGATION[
      cfg.readout_edges_for_globals_aggregation
  ]

  shift, scale = -1.6526496, 1.0

  return CrystalEnergyModel(
      cfg.graph_net_steps,
      cfg.mlp_width,
      cfg.mlp_nonlinearity,
      cfg.embedding_dim,
      cfg.featurizer,
      shift,
      scale,
      cfg.feature_band_limit,
      cfg.conditioning_band_limit,
      cfg.extra_scalars_for_gating,
      cfg.residual,
      aggregate_edges_for_nodes_fn=node_aggregation_fn,
      aggregate_edges_for_globals_fn=edges_for_globals_aggregation_fn,
      readout_aggregate_edges_for_globals_fn=readout_edges_for_globals_aggregation_fn,
      normalize_edges_for_globals_by_nodes=normalize_edges_for_globals_by_nodes,
  )
