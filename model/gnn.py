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
"""Graph network definitions."""

import functools
from typing import Any, Callable, Iterable, Optional
import jax
import jax.numpy as jnp
import jraph

Array = jnp.ndarray
PRNGKey = Array
Shape = Iterable[int]
Dtype = Any
InitializerFn = Callable[[PRNGKey, Shape, Dtype], Array]

partial = functools.partial

EPSILON = 1e-7

tree = jax.tree_util
segment_sum = jraph.segment_sum
segment_mean = jraph.segment_mean

GNUpdateNodeFn = jraph.GNUpdateNodeFn
GNUpdateEdgeFn = jraph.GNUpdateEdgeFn
GNUpdateGlobalFn = jraph.GNUpdateGlobalFn
AggregateEdgesToNodesFn = jraph.AggregateEdgesToNodesFn
AggregateEdgesToNodesFn = jraph.AggregateEdgesToNodesFn
AggregateEdgesToGlobalsFn = jraph.AggregateEdgesToGlobalsFn
AggregateNodesToGlobalsFn = jraph.AggregateNodesToGlobalsFn


def GraphNetwork(
    update_edge_fn: Optional[GNUpdateEdgeFn],
    update_node_fn: Optional[GNUpdateNodeFn],
    update_global_fn: Optional[GNUpdateGlobalFn] = None,
    aggregate_edges_for_nodes_fn: AggregateEdgesToNodesFn = segment_sum,
    aggregate_nodes_for_globals_fn: AggregateNodesToGlobalsFn = segment_sum,
    aggregate_edges_for_globals_fn: AggregateEdgesToGlobalsFn = segment_sum,
    attention_logit_fn: Optional[jraph.AttentionLogitFn] = None,
    attention_normalize_fn: Optional[Any] = jraph.segment_softmax,
    attention_reduce_fn: Optional[jraph.AttentionReduceFn] = None,
    normalize_edges_for_globals_by_nodes: bool = False,
):
  """Define the graph neural network."""

  not_both_supplied = lambda x, y: (x != y) and ((x is None) or (y is None))
  if not_both_supplied(attention_reduce_fn, attention_logit_fn):
    raise ValueError(
        'attention_logit_fn and attention_reduce_fn must both be supplied.'
    )

  def _ApplyGraphNet(graph):
    """Applies a configured GraphNetwork to a graph.

    This implementation follows Algorithm 1 in https://arxiv.org/abs/1806.01261

    There is one difference. For the nodes update the class aggregates over the
    sender edges and receiver edges separately. This is a bit more general
    the algorithm described in the paper. The original behaviour can be
    recovered by using only the receiver edge aggregations for the update.

    In addition this implementation supports softmax attention over incoming
    edge features.

    Many popular Graph Neural Networks can be implemented as special cases of
    GraphNets, for more information please see the paper.

    Args:
      graph: a `GraphsTuple` containing the graph.

    Returns:
      Updated `GraphsTuple`.
    """
    # pylint: disable=g-long-lambda
    nodes, edges, receivers, senders, globals_, n_node, n_edge = graph
    # Equivalent to jnp.sum(n_node), but jittable
    sum_n_node = tree.tree_leaves(nodes)[0].shape[0]
    sum_n_edge = senders.shape[0]
    if not tree.tree_all(
        tree.tree_map(lambda n: n.shape[0] == sum_n_node, nodes)
    ):
      raise ValueError(
          'All node arrays in nest must contain the same number of nodes.'
      )

    sent_attributes = tree.tree_map(lambda n: n[senders], nodes)
    received_attributes = tree.tree_map(lambda n: n[receivers], nodes)
    # Here we scatter the global features to the corresponding edges,
    # giving us tensors of shape [num_edges, global_feat].
    global_edge_attributes = tree.tree_map(
        lambda g: jnp.repeat(g, n_edge, axis=0, total_repeat_length=sum_n_edge),
        globals_,
    )

    if update_edge_fn:
      edges = update_edge_fn(
          edges, sent_attributes, received_attributes, global_edge_attributes
      )

    if attention_logit_fn:
      logits = attention_logit_fn(
          edges, sent_attributes, received_attributes, global_edge_attributes
      )
      tree_calculate_weights = functools.partial(
          attention_normalize_fn, segment_ids=receivers, num_segments=sum_n_node
      )
      weights = tree.tree_map(tree_calculate_weights, logits)
      edges = attention_reduce_fn(edges, weights)

    if update_node_fn:
      sent_attributes = tree.tree_map(
          lambda e: aggregate_edges_for_nodes_fn(e, senders, sum_n_node), edges
      )
      received_attributes = tree.tree_map(
          lambda e: aggregate_edges_for_nodes_fn(e, receivers, sum_n_node),
          edges,
      )
      # Here we scatter the global features to the corresponding nodes,
      # giving us tensors of shape [num_nodes, global_feat].
      global_attributes = tree.tree_map(
          lambda g: jnp.repeat(
              g, n_node, axis=0, total_repeat_length=sum_n_node
          ),
          globals_,
      )
      nodes = update_node_fn(
          nodes, sent_attributes, received_attributes, global_attributes
      )

    if update_global_fn:
      n_graph = n_node.shape[0]
      graph_idx = jnp.arange(n_graph)
      # To aggregate nodes and edges from each graph to global features,
      # we first construct tensors that map the node to the corresponding graph.
      # For example, if you have `n_node=[1,2]`, we construct the tensor
      # [0, 1, 1]. We then do the same for edges.
      node_gr_idx = jnp.repeat(
          graph_idx, n_node, axis=0, total_repeat_length=sum_n_node
      )
      edge_gr_idx = jnp.repeat(
          graph_idx, n_edge, axis=0, total_repeat_length=sum_n_edge
      )
      # We use the aggregation function to pool the nodes/edges per graph.
      node_attributes = tree.tree_map(
          lambda n: aggregate_nodes_for_globals_fn(n, node_gr_idx, n_graph),
          nodes,
      )
      norm = jnp.ones_like(n_node)
      if normalize_edges_for_globals_by_nodes:
        norm = n_node

      def edge_agg_fn(e):
        Z = norm.reshape(norm.shape[:1] + (1,) * (len(e.shape) - 1))
        return aggregate_edges_for_globals_fn(e, edge_gr_idx, n_graph) / Z

      edge_attributes = tree.tree_map(edge_agg_fn, edges)
      # These pooled nodes are the inputs to the global update fn.
      globals_ = update_global_fn(node_attributes, edge_attributes, globals_)
    # pylint: enable=g-long-lambda
    return jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        receivers=receivers,
        senders=senders,
        globals=globals_,
        n_node=n_node,
        n_edge=n_edge,
    )

  return _ApplyGraphNet
