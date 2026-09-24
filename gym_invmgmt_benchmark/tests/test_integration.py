from __future__ import annotations

import numpy as np
import pytest
import torch

from fbit.agent import FBiTAgent
from fbit.checkpoint import model_metadata, save_checkpoint
from fbit.config import (
    ACTION_SCALING_EDGE_CAPS,
    ACTION_SCALING_ENVIRONMENT_HIGH,
    FBiTConfig,
)
from fbit.envs import make_stationary_env
from fbit.edge_cap_wrapper import EdgeCapRescaleAction
from fbit.graph_adapter import CoreEnvGraphAdapter
from fbit.networks import FBiTSystem
from fbit.trainer import FBiTTrainer
from fbit.utils import fbit_action_bounds, scaled_to_raw_action
from scripts.evaluate_fbit import run_episode


def small_config(topology: str) -> FBiTConfig:
    return FBiTConfig(
        topology=topology,
        hidden_dim=8,
        graph_dim=16,
        graph_heads=1,
        predictor_dim=8,
        predictor_heads=1,
        predictor_layers=1,
        actor_hidden=32,
        critic_hidden=32,
        warmup_episodes=1,
        updates_per_episode=1,
        episode_batch_size=1,
        seed=123,
    )


@pytest.mark.parametrize(
    ("topology", "expected_environment_high", "expected_fbit_high"),
    (
        (
            "base",
            3100.0,
            [3100, 3100, 90, 90, 90, 80, 80, 100, 100, 100, 100],
        ),
        ("serial", 3200.0, [3200, 100, 100]),
    ),
)
def test_per_edge_action_scaling_does_not_change_environment(
    topology, expected_environment_high, expected_fbit_high
):
    env = make_stationary_env(topology)
    original_high = env.action_space.high.copy()
    low, high = fbit_action_bounds(
        env, ACTION_SCALING_EDGE_CAPS, raw_material_cap=100.0
    )
    np.testing.assert_allclose(low, 0.0)
    np.testing.assert_allclose(high, expected_fbit_high)
    np.testing.assert_allclose(original_high, expected_environment_high)
    np.testing.assert_array_equal(env.action_space.high, original_high)
    env.close()


def test_per_edge_action_scaling_endpoints():
    high = np.asarray(
        [3100, 3100, 90, 90, 90, 80, 80, 100, 100, 100, 100],
        dtype=np.float64,
    )
    low = np.zeros_like(high)
    np.testing.assert_allclose(scaled_to_raw_action(-np.ones(11), low, high), 0)
    np.testing.assert_allclose(scaled_to_raw_action(np.zeros(11), low, high), high / 2)
    np.testing.assert_allclose(scaled_to_raw_action(np.ones(11), low, high), high)


@pytest.mark.parametrize(
    ("topology", "expected_high"),
    (
        ("base", [3100, 3100, 90, 90, 90, 80, 80, 100, 100, 100, 100]),
        ("serial", [3200, 100, 100]),
    ),
)
def test_edge_cap_training_wrapper(topology, expected_high):
    env = make_stationary_env(topology)
    original_high = env.action_space.high.copy()
    wrapped = EdgeCapRescaleAction(env, raw_material_cap=100.0)
    np.testing.assert_allclose(wrapped.action(-np.ones_like(original_high)), 0.0)
    np.testing.assert_allclose(wrapped.action(np.zeros_like(original_high)), np.asarray(expected_high) / 2.0)
    np.testing.assert_allclose(wrapped.action(np.ones_like(original_high)), expected_high)
    np.testing.assert_allclose(wrapped.reverse_action(expected_high), 1.0)
    np.testing.assert_array_equal(env.action_space.high, original_high)
    wrapped.close()


def test_config_without_explicit_scaling_uses_current_edge_caps():
    values = FBiTConfig().to_dict()
    values.pop("action_scaling_mode")
    values.pop("raw_material_action_cap")
    restored = FBiTConfig.from_dict(values)
    assert restored.action_scaling_mode == ACTION_SCALING_EDGE_CAPS


@pytest.mark.parametrize("topology", ("base", "serial"))
def test_agent_runs_complete_episode_with_bounded_actions(tmp_path, topology):
    env = make_stationary_env(topology)
    adapter = CoreEnvGraphAdapter(env)
    config = small_config(topology)
    metadata = model_metadata(adapter)
    system = FBiTSystem(
        config,
        adapter.num_retail,
        metadata["retailer_node_indices"],
        adapter.num_nodes,
        adapter.num_actions,
    )
    checkpoint = tmp_path / f"{topology}.pt"
    save_checkpoint(checkpoint, system, config, metadata)
    agent = FBiTAgent(env, checkpoint)
    expected_low, expected_high = fbit_action_bounds(
        env, ACTION_SCALING_EDGE_CAPS, raw_material_cap=100.0
    )
    np.testing.assert_array_equal(agent.action_low, expected_low)
    np.testing.assert_array_equal(agent.action_high, expected_high)
    observation, _ = env.reset(seed=42)
    for period in range(30):
        action = agent.get_action(observation, period)
        assert action.shape == env.action_space.shape
        assert np.isfinite(action).all()
        assert (action >= env.action_space.low).all()
        assert (action <= env.action_space.high).all()
        observation, _, terminated, truncated, _ = env.step(action)
    assert terminated or truncated
    env.close()


def test_resume_rejects_legacy_action_scaling(tmp_path):
    topology = "base"
    env = make_stationary_env(topology)
    adapter = CoreEnvGraphAdapter(env)
    legacy_config = small_config(topology)
    legacy_config.action_scaling_mode = ACTION_SCALING_ENVIRONMENT_HIGH
    metadata = model_metadata(adapter)
    system = FBiTSystem(
        legacy_config,
        adapter.num_retail,
        metadata["retailer_node_indices"],
        adapter.num_nodes,
        adapter.num_actions,
    )
    checkpoint = tmp_path / "legacy_resume.pt"
    save_checkpoint(checkpoint, system, legacy_config, metadata)
    env.close()

    trainer = FBiTTrainer(
        small_config(topology),
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path / "checkpoints",
        log_dir=tmp_path / "logs",
    )
    with pytest.raises(ValueError, match="action scaling mode"):
        trainer.load_resume(checkpoint)
    trainer.env.close()


@pytest.mark.slow
@pytest.mark.parametrize("topology", ("base", "serial"))
def test_one_episode_training_smoke(tmp_path, topology):
    config = small_config(topology)
    trainer = FBiTTrainer(
        config,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path / topology / "checkpoints",
        log_dir=tmp_path / topology / "logs",
    )
    final = trainer.train(total_steps=30, save_every=0)
    assert final.exists()
    payload = torch.load(final, map_location="cpu", weights_only=False)
    assert payload["trainer_state"]["global_step"] == 30
    assert payload["trainer_state"]["episode"] == 1

    resumed = FBiTTrainer(
        config,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path / topology / "checkpoints",
        log_dir=tmp_path / topology / "logs",
    )
    resumed_final = resumed.train(
        total_steps=60, resume=resumed.resume_path, save_every=0
    )
    resumed_payload = torch.load(
        resumed_final, map_location="cpu", weights_only=False
    )
    assert resumed_payload["trainer_state"]["global_step"] == 60
    assert resumed_payload["trainer_state"]["episode"] == 2


@pytest.mark.slow
def test_single_scenario_evaluation_smoke(tmp_path):
    env = make_stationary_env("serial")
    adapter = CoreEnvGraphAdapter(env)
    config = small_config("serial")
    metadata = model_metadata(adapter)
    system = FBiTSystem(
        config,
        adapter.num_retail,
        metadata["retailer_node_indices"],
        adapter.num_nodes,
        adapter.num_actions,
    )
    checkpoint = tmp_path / "serial.pt"
    save_checkpoint(checkpoint, system, config, metadata)
    agent = FBiTAgent(env, checkpoint)
    kpis = run_episode(env, agent, seed=42)
    for key in ("Profit", "FillRate", "AvgInv", "Unfulfilled", "Latency"):
        assert np.isfinite(kpis[key])
    env.close()
