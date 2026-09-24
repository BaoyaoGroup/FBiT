from __future__ import annotations

import inspect

import numpy as np
import pytest

from fbit.envs import make_stationary_env
from fbit.graph_adapter import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    PIPELINE_ARRIVAL_HORIZON,
    CoreEnvGraphAdapter,
)


@pytest.mark.parametrize(
    ("topology", "nodes", "actions", "observation_width", "pipeline_width"),
    (
        ("base", 9, 11, 71, 61),
        ("serial", 5, 3, 15, 8),
    ),
)
def test_topology_shapes(
    topology, nodes, actions, observation_width, pipeline_width
):
    env = make_stationary_env(topology)
    observation, _ = env.reset(seed=42)
    adapter = CoreEnvGraphAdapter(env)
    graph = adapter.graph(observation, [])
    assert observation.shape == (observation_width,)
    assert adapter.num_nodes == nodes
    assert adapter.num_actions == actions
    assert adapter.pipeline_length == pipeline_width
    assert graph.x.shape == (nodes, len(NODE_FEATURE_NAMES))
    assert graph.edge_attr.shape[1] == len(EDGE_FEATURE_NAMES)
    assert graph.action_edge_index.shape == (2, actions)
    assert graph.pipeline.shape == (actions, 1)
    env.close()


@pytest.mark.parametrize("topology", ("base", "serial"))
def test_pipeline_buckets_are_action_aligned(topology):
    env = make_stationary_env(topology)
    observation, _ = env.reset(seed=1)
    adapter = CoreEnvGraphAdapter(env)
    parts = adapter.split_observation(observation)
    buckets = np.arange(1, adapter.pipeline_length + 1, dtype=float)
    arrivals = adapter.action_aligned_arrivals(buckets)
    actual = adapter.action_aligned_pipeline(buckets)
    expected = []
    cursor = 0
    for index, edge in enumerate(adapter.reorder_links):
        lead_time = int(adapter.network.lead_times[edge])
        expected.append(buckets[cursor : cursor + lead_time].sum())
        np.testing.assert_allclose(
            arrivals[index, :lead_time],
            buckets[cursor : cursor + lead_time],
        )
        assert not arrivals[index, lead_time:].any()
        cursor += lead_time
    np.testing.assert_allclose(actual, expected)
    assert arrivals.shape == (adapter.num_actions, PIPELINE_ARRIVAL_HORIZON)
    for index, edge in enumerate(adapter.reorder_links):
        if int(adapter.network.lead_times[edge]) == 0:
            assert actual[index] == 0.0
    env.close()


@pytest.mark.parametrize("topology", ("base", "serial"))
def test_node_incoming_features_preserve_arrival_times(topology):
    env = make_stationary_env(topology)
    observation, _ = env.reset(seed=2)
    adapter = CoreEnvGraphAdapter(env)
    buckets = np.arange(1, adapter.pipeline_length + 1, dtype=float)
    pipeline_start = 2 * adapter.num_retail + len(adapter.network.main_nodes)
    observation[pipeline_start : pipeline_start + adapter.pipeline_length] = buckets
    graph = adapter.graph(observation, [])

    expected = np.zeros((adapter.num_nodes, PIPELINE_ARRIVAL_HORIZON))
    arrivals = adapter.action_aligned_arrivals(buckets)
    for (_supplier, purchaser), schedule in zip(
        adapter.reorder_links, arrivals
    ):
        expected[adapter.node_to_local[purchaser]] += schedule
    pipeline_scale = max(1.0, float(np.max(env.action_space.high)))
    start = NODE_FEATURE_NAMES.index("incoming_t0")
    actual = graph.x[:, start : start + PIPELINE_ARRIVAL_HORIZON].numpy()
    np.testing.assert_allclose(actual, expected / pipeline_scale)
    env.close()


def test_history_contains_only_observed_lagged_demand():
    env = make_stationary_env("serial")
    observation, _ = env.reset(seed=7)
    adapter = CoreEnvGraphAdapter(env)
    history = []
    adapter.append_observation(history, observation, current_period=0)
    assert history == []
    observation, *_ = env.step(np.zeros(adapter.num_actions))
    adapter.append_observation(history, observation, current_period=1)
    assert len(history) == 1
    np.testing.assert_allclose(history[0], adapter.observed_demand(observation))
    graph = adapter.graph(observation, history)
    assert graph.history_length.item() == 1
    np.testing.assert_allclose(
        graph.demand_history[0, 0].numpy(), history[0]
    )
    env.close()


def test_adapter_does_not_read_private_future_matrices():
    source = inspect.getsource(CoreEnvGraphAdapter)
    for forbidden in (
        "self.env.D",
        "self.env.R",
        "self.env.demand_engine",
        "get_current_mu",
        "external_series",
    ):
        assert forbidden not in source
