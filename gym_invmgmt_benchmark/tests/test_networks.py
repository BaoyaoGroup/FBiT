from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from fbit.checkpoint import (
    build_system_from_payload,
    load_payload,
    model_metadata,
    save_checkpoint,
)
from fbit.config import FBiTConfig
from fbit.envs import make_stationary_env
from fbit.graph_adapter import CoreEnvGraphAdapter
from fbit.networks import DemandPredictor, FBiTSystem


def make_system(topology="serial"):
    env = make_stationary_env(topology)
    observation, _ = env.reset(seed=11)
    adapter = CoreEnvGraphAdapter(env)
    config = FBiTConfig(
        topology=topology,
        hidden_dim=8,
        graph_dim=16,
        graph_heads=1,
        predictor_dim=8,
        predictor_heads=1,
        predictor_layers=1,
        actor_hidden=32,
        critic_hidden=32,
    )
    metadata = model_metadata(adapter)
    system = FBiTSystem(
        config,
        adapter.num_retail,
        metadata["retailer_node_indices"],
        adapter.num_nodes,
        adapter.num_actions,
    )
    graph = adapter.graph(observation, [])
    return env, adapter, config, metadata, system, graph


def test_predictor_is_causal_for_prefix():
    config = FBiTConfig(
        predictor_dim=8,
        predictor_heads=1,
        predictor_layers=1,
    )
    predictor = DemandPredictor(config, num_retail=1).eval()
    first = torch.zeros(1, 30, 1)
    second = first.clone()
    first[:, :4, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    second[:, :4, 0] = first[:, :4, 0]
    second[:, 4:, 0] = 999.0
    length = torch.tensor([4])
    with torch.no_grad():
        left = predictor(first, length)
        right = predictor(second, length)
    torch.testing.assert_close(left, right)


@pytest.mark.parametrize("topology", ("base", "serial"))
def test_actor_critic_shapes_and_finite_values(topology):
    env, adapter, config, _metadata, system, graph = make_system(topology)
    assert system.q1.fc_s.in_features == (
        adapter.num_nodes * config.graph_dim * 2
    )
    assert system.q1.fc_s.out_features == config.critic_hidden
    assert system.q1.fc_a.in_features == adapter.num_actions
    assert system.q1.fc_a.out_features == config.critic_hidden
    assert system.q1.fc_cat.in_features == config.critic_hidden * 2
    batch = Batch.from_data_list([graph, graph])
    retail = system.predict_retail_demand(batch)
    node_demand = system.demand_to_nodes(batch, retail)
    action, log_prob = system.actor(batch, node_demand)
    q_value = system.q1(batch, action, node_demand)
    assert action.shape == (2 * adapter.num_actions,)
    assert log_prob.shape == (2, 1)
    assert q_value.shape == (2, 1)
    assert torch.isfinite(action).all()
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(q_value).all()
    assert (action.abs() <= 1.0).all()
    env.close()


def test_critic_action_branch_receives_complete_raw_action_vector():
    env, adapter, _config, _metadata, system, graph = make_system("base")
    batch = Batch.from_data_list([graph, graph])
    action = torch.linspace(
        -1.0, 1.0, steps=2 * adapter.num_actions, dtype=torch.float32
    )
    captured = []
    hook = system.q1.fc_a.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    with torch.no_grad():
        node_demand = torch.zeros((batch.num_nodes, 1), dtype=torch.float32)
        system.q1(batch, action, node_demand)
    hook.remove()
    assert len(captured) == 1
    torch.testing.assert_close(
        captured[0], action.reshape(2, adapter.num_actions)
    )
    env.close()


def test_checkpoint_round_trip(tmp_path):
    env, adapter, config, metadata, system, graph = make_system("serial")
    batch = Batch.from_data_list([graph])
    system.eval()
    with torch.no_grad():
        retail = system.predict_retail_demand(batch)
        demand = system.demand_to_nodes(batch, retail)
        expected, _ = system.actor(batch, demand, deterministic=True)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, system, config, metadata)
    payload = load_payload(path, torch.device("cpu"))
    restored, restored_config, restored_metadata = build_system_from_payload(
        payload, torch.device("cpu")
    )
    restored.eval()
    with torch.no_grad():
        retail = restored.predict_retail_demand(batch)
        demand = restored.demand_to_nodes(batch, retail)
        actual, _ = restored.actor(batch, demand, deterministic=True)
    torch.testing.assert_close(actual, expected)
    assert restored_config.topology == "serial"
    assert restored_metadata["reorder_links"] == metadata["reorder_links"]
    env.close()


def test_old_checkpoint_version_is_rejected(tmp_path):
    env, _adapter, config, metadata, system, _graph = make_system("serial")
    path = tmp_path / "old_checkpoint.pt"
    save_checkpoint(path, system, config, metadata)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["checkpoint_version"] = 4
    with pytest.raises(ValueError, match="Unsupported FBiT checkpoint version: 4"):
        build_system_from_payload(payload, torch.device("cpu"))
    env.close()
