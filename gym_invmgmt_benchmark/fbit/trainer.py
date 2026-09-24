from __future__ import annotations

import csv
import random
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch.utils.tensorboard import SummaryWriter

from .checkpoint import (
    build_system_from_payload,
    load_payload,
    model_metadata,
    save_checkpoint,
)
from .config import FBiTConfig
from .envs import make_training_env
from .graph_adapter import CoreEnvGraphAdapter
from .networks import FBiTSystem
from .replay import EpisodeReplayBuffer, Transition
from .utils import fbit_action_bounds, scaled_to_raw_action, set_global_seed


class FBiTTrainer:
    def __init__(
        self,
        config: FBiTConfig,
        device: torch.device,
        checkpoint_dir: Path,
        log_dir: Path,
    ):
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.env = make_training_env(config.topology, config.num_periods)
        self.adapter = CoreEnvGraphAdapter(self.env, device="cpu")
        self.action_low, self.action_high = fbit_action_bounds(
            self.env,
            config.action_scaling_mode,
            config.raw_material_action_cap,
        )
        if self.adapter.num_retail <= 0:
            raise ValueError("FBiT requires at least one retail demand link")
        metadata = model_metadata(self.adapter)
        metadata["action_scaling_high"] = self.action_high.tolist()
        self.metadata = metadata
        self.system = FBiTSystem(
            config,
            num_retail=self.adapter.num_retail,
            retailer_node_indices=metadata["retailer_node_indices"],
            num_nodes=self.adapter.num_nodes,
            num_actions=self.adapter.num_actions,
        ).to(device)
        self.optimizers = {
            "actor": torch.optim.Adam(
                self.system.actor.parameters(), lr=config.actor_lr
            ),
            "q1": torch.optim.Adam(
                self.system.q1.parameters(), lr=config.critic_lr
            ),
            "q2": torch.optim.Adam(
                self.system.q2.parameters(), lr=config.critic_lr
            ),
            "predictor": torch.optim.Adam(
                self.system.predictor.parameters(), lr=config.predictor_lr
            ),
            "alpha": torch.optim.Adam(
                [self.system.log_alpha], lr=config.alpha_lr
            ),
        }
        self.replay = EpisodeReplayBuffer(config.replay_capacity_steps)
        self.global_step = 0
        self.episode = 0
        self.update_step = 0
        self.best_rolling_return = -float("inf")
        self.recent_returns: deque[float] = deque(maxlen=10)

    @property
    def topology_label(self) -> str:
        return "base" if self.config.topology == "base" else "serial"

    @property
    def resume_path(self) -> Path:
        return self.checkpoint_dir / f"fbit_{self.topology_label}_blind_resume.pt"

    @property
    def final_path(self) -> Path:
        return self.checkpoint_dir / f"fbit_{self.topology_label}_blind_final.pt"

    @property
    def best_path(self) -> Path:
        return self.checkpoint_dir / f"fbit_{self.topology_label}_blind_best.pt"

    def _trainer_state(self) -> Dict[str, Any]:
        return {
            "global_step": self.global_step,
            "episode": self.episode,
            "update_step": self.update_step,
            "best_rolling_return": self.best_rolling_return,
            "recent_returns": list(self.recent_returns),
        }

    def load_resume(self, path: Path) -> None:
        # Replay graphs must remain on CPU. Loading the whole payload directly
        # onto CUDA would mix historical CUDA graphs with newly collected CPU
        # graphs after resume.
        payload = load_payload(path, torch.device("cpu"))
        loaded_system, loaded_config, metadata = build_system_from_payload(
            payload, self.device
        )
        if loaded_config.topology != self.config.topology:
            raise ValueError(
                f"Resume topology {loaded_config.topology} does not match "
                f"{self.config.topology}"
            )
        if loaded_config.action_scaling_mode != self.config.action_scaling_mode:
            raise ValueError(
                "Resume action scaling mode "
                f"{loaded_config.action_scaling_mode!r} does not match "
                f"{self.config.action_scaling_mode!r}"
            )
        if (
            self.config.action_scaling_mode == "edge_caps"
            and loaded_config.raw_material_action_cap
            != self.config.raw_material_action_cap
        ):
            raise ValueError(
                "Resume raw-material action cap "
                f"{loaded_config.raw_material_action_cap} does not match "
                f"{self.config.raw_material_action_cap}"
            )
        stored_high = metadata.get("action_scaling_high")
        if stored_high is not None and not np.array_equal(
            np.asarray(stored_high, dtype=np.float64), self.action_high
        ):
            raise ValueError(
                "Resume per-edge action bounds do not match the training environment"
            )
        self.system.load_state_dict(loaded_system.state_dict())
        for name, state in payload.get("optimizers", {}).items():
            if name in self.optimizers:
                self.optimizers[name].load_state_dict(state)
        if "replay" in payload:
            self.replay.load_state_dict(payload["replay"])
        state = payload.get("trainer_state", {})
        self.global_step = int(state.get("global_step", 0))
        self.episode = int(state.get("episode", 0))
        self.update_step = int(state.get("update_step", 0))
        self.best_rolling_return = float(
            state.get("best_rolling_return", -float("inf"))
        )
        self.recent_returns.extend(state.get("recent_returns", []))
        self.metadata = metadata
        rng = payload.get("rng", {})
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch" in rng:
            torch.set_rng_state(rng["torch"].cpu())
        if (
            torch.cuda.is_available()
            and rng.get("cuda") is not None
        ):
            torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])

    def _save_resume(self) -> None:
        save_checkpoint(
            self.resume_path,
            self.system,
            self.config,
            self.metadata,
            trainer_state=self._trainer_state(),
            optimizers=self.optimizers,
            replay_state=self.replay.state_dict(),
        )

    def _save_deployment(self, path: Path) -> None:
        save_checkpoint(
            path,
            self.system,
            self.config,
            self.metadata,
            trainer_state=self._trainer_state(),
        )

    def _predicted_nodes(self, graph_batch) -> torch.Tensor:
        retail = self.system.predict_retail_demand(graph_batch)
        return self.system.demand_to_nodes(graph_batch, retail)

    def update(self) -> Dict[str, float]:
        batch = self.replay.sample(
            self.config.episode_batch_size, self.device
        )
        graphs = batch["graphs"]
        next_graphs = batch["next_graphs"]

        predicted_retail = self.system.predict_retail_demand(graphs)
        predictor_loss = F.smooth_l1_loss(
            predicted_retail, batch["next_demands"]
        )
        self.optimizers["predictor"].zero_grad()
        predictor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.system.predictor.parameters(), 5.0)
        self.optimizers["predictor"].step()

        with torch.no_grad():
            node_demand = self._predicted_nodes(graphs).detach()
            next_node_demand = self._predicted_nodes(next_graphs).detach()
            next_actions, next_log_prob = self.system.actor(
                next_graphs, next_node_demand
            )
            target_q = torch.minimum(
                self.system.q1_target(
                    next_graphs, next_actions, next_node_demand
                ),
                self.system.q2_target(
                    next_graphs, next_actions, next_node_demand
                ),
            )
            target = batch["rewards"] + self.config.gamma * (
                1.0 - batch["dones"]
            ) * (target_q - self.system.alpha.detach() * next_log_prob)

        q1_value = self.system.q1(graphs, batch["actions"], node_demand)
        q2_value = self.system.q2(graphs, batch["actions"], node_demand)
        q1_loss = F.smooth_l1_loss(q1_value, target)
        q2_loss = F.smooth_l1_loss(q2_value, target)
        for name, loss, network in (
            ("q1", q1_loss, self.system.q1),
            ("q2", q2_loss, self.system.q2),
        ):
            self.optimizers[name].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 10.0)
            self.optimizers[name].step()

        sampled_actions, log_prob = self.system.actor(graphs, node_demand)
        actor_q = torch.minimum(
            self.system.q1(graphs, sampled_actions, node_demand),
            self.system.q2(graphs, sampled_actions, node_demand),
        )
        actor_loss = (
            self.system.alpha.detach() * log_prob - actor_q
        ).mean()
        self.optimizers["actor"].zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.system.actor.parameters(), 10.0)
        self.optimizers["actor"].step()

        target_entropy = -float(self.adapter.num_actions)
        alpha_loss = -(
            self.system.log_alpha
            * (log_prob.detach() + target_entropy)
        ).mean()
        named_losses = {
            "predictor_loss": predictor_loss,
            "q1_loss": q1_loss,
            "q2_loss": q2_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
        }
        non_finite = [
            name
            for name, value in named_losses.items()
            if not bool(torch.isfinite(value).all())
        ]
        if non_finite:
            raise FloatingPointError(
                "Non-finite FBiT training loss(es): " + ", ".join(non_finite)
            )
        self.optimizers["alpha"].zero_grad()
        alpha_loss.backward()
        self.optimizers["alpha"].step()
        self.system.soft_update_targets()
        self.update_step += 1
        return {
            "predictor_loss": float(predictor_loss.detach()),
            "q1_loss": float(q1_loss.detach()),
            "q2_loss": float(q2_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha": float(self.system.alpha.detach()),
        }

    def train(
        self,
        total_steps: int,
        resume: Optional[Path] = None,
        save_every: int = 10_000,
    ) -> Path:
        set_global_seed(self.config.seed)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if resume is not None:
            self.load_resume(resume)
        writer = SummaryWriter(str(self.log_dir))
        csv_path = self.log_dir / "training_log.csv"
        csv_exists = csv_path.exists()
        csv_handle = csv_path.open("a", encoding="utf-8", newline="")
        csv_writer = csv.DictWriter(
            csv_handle,
            fieldnames=[
                "episode",
                "global_step",
                "return",
                "rolling_return",
                "alpha",
                "predictor_loss",
                "q1_loss",
                "q2_loss",
                "actor_loss",
            ],
        )
        if not csv_exists:
            csv_writer.writeheader()

        next_save = (
            ((self.global_step // save_every) + 1) * save_every
            if save_every > 0
            else total_steps + 1
        )
        try:
            while self.global_step < total_steps:
                observation, _ = self.env.reset(
                    seed=self.config.seed + self.episode
                )
                history: list[np.ndarray] = []
                graph = self.adapter.graph(observation, history)
                episode_return = 0.0
                done = False
                while not done:
                    graph_batch = Batch.from_data_list([graph]).to(self.device)
                    with torch.no_grad():
                        predicted_nodes = self._predicted_nodes(graph_batch)
                        scaled_action, _ = self.system.actor(
                            graph_batch, predicted_nodes, deterministic=False
                        )
                    scaled_np = scaled_action.cpu().numpy()
                    if not np.isfinite(scaled_np).all():
                        raise FloatingPointError(
                            "FBiT actor produced a non-finite action"
                        )
                    raw_action = scaled_to_raw_action(
                        scaled_np,
                        self.action_low,
                        self.action_high,
                    )
                    next_obs, reward, terminated, truncated, _ = self.env.step(
                        raw_action
                    )
                    done = bool(terminated or truncated)
                    next_history = list(history)
                    self.adapter.append_observation(
                        next_history,
                        next_obs,
                        current_period=len(history) + 1,
                    )
                    next_graph = self.adapter.graph(next_obs, next_history)
                    self.replay.append(
                        Transition(
                            graph=graph.cpu(),
                            action=scaled_np.astype(np.float32, copy=True),
                            reward=float(reward) * self.config.reward_scale,
                            next_graph=next_graph.cpu(),
                            done=done,
                            next_observed_demand=self.adapter.observed_demand(
                                next_obs
                            ),
                        )
                    )
                    graph = next_graph
                    history = next_history
                    episode_return += float(reward)
                    self.global_step += 1

                self.replay.finish_episode()
                self.episode += 1
                self.recent_returns.append(episode_return)
                rolling = float(np.mean(self.recent_returns))
                losses = {
                    "predictor_loss": float("nan"),
                    "q1_loss": float("nan"),
                    "q2_loss": float("nan"),
                    "actor_loss": float("nan"),
                    "alpha": float(self.system.alpha.detach()),
                }
                if len(self.replay) >= max(1, self.config.warmup_episodes):
                    for _ in range(self.config.updates_per_episode):
                        losses = self.update()

                writer.add_scalar("rollout/episode_return", episode_return, self.global_step)
                writer.add_scalar("rollout/rolling_return_10", rolling, self.global_step)
                for key, value in losses.items():
                    if np.isfinite(value):
                        writer.add_scalar(f"train/{key}", value, self.global_step)
                csv_writer.writerow(
                    {
                        "episode": self.episode,
                        "global_step": self.global_step,
                        "return": episode_return,
                        "rolling_return": rolling,
                        **losses,
                    }
                )
                csv_handle.flush()

                if len(self.recent_returns) == self.recent_returns.maxlen and rolling > self.best_rolling_return:
                    self.best_rolling_return = rolling
                    self._save_deployment(self.best_path)
                if self.global_step >= next_save:
                    self._save_resume()
                    next_save += save_every
                if self.episode % 10 == 0:
                    print(
                        f"[{self.topology_label}] episode={self.episode} "
                        f"steps={self.global_step}/{total_steps} "
                        f"return={episode_return:.1f} rolling10={rolling:.1f} "
                        f"alpha={losses['alpha']:.4f}",
                        flush=True,
                    )
        finally:
            csv_handle.close()
            writer.close()
            self.env.close()

        self._save_deployment(self.final_path)
        self._save_resume()
        return self.final_path
