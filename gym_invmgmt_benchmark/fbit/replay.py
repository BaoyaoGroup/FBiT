from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Sequence

import numpy as np
import torch
from torch_geometric.data import Batch, Data


@dataclass
class Transition:
    graph: Data
    action: np.ndarray
    reward: float
    next_graph: Data
    done: bool
    next_observed_demand: np.ndarray


class EpisodeReplayBuffer:
    """Episode-aware replay with a step-based capacity."""

    def __init__(self, capacity_steps: int):
        self.capacity_steps = int(capacity_steps)
        self.episodes: Deque[List[Transition]] = deque()
        self.current_episode: List[Transition] = []
        self.num_steps = 0

    def append(self, transition: Transition) -> None:
        self.current_episode.append(transition)

    def finish_episode(self) -> None:
        if not self.current_episode:
            return
        episode = self.current_episode
        self.current_episode = []
        self.episodes.append(episode)
        self.num_steps += len(episode)
        while self.num_steps > self.capacity_steps and self.episodes:
            self.num_steps -= len(self.episodes.popleft())

    def __len__(self) -> int:
        return len(self.episodes)

    def sample(self, episode_count: int, device: torch.device) -> Dict[str, object]:
        if not self.episodes:
            raise RuntimeError("Cannot sample from an empty replay buffer")
        count = min(int(episode_count), len(self.episodes))
        selected = random.sample(list(self.episodes), count)
        transitions = [item for episode in selected for item in episode]
        graphs = Batch.from_data_list([item.graph for item in transitions]).to(device)
        next_graphs = Batch.from_data_list(
            [item.next_graph for item in transitions]
        ).to(device)
        return {
            "graphs": graphs,
            "actions": torch.as_tensor(
                np.concatenate([item.action for item in transitions]),
                dtype=torch.float32,
                device=device,
            ),
            "rewards": torch.tensor(
                [item.reward for item in transitions],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(1),
            "next_graphs": next_graphs,
            "dones": torch.tensor(
                [item.done for item in transitions],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(1),
            "next_demands": torch.as_tensor(
                np.stack([item.next_observed_demand for item in transitions]),
                dtype=torch.float32,
                device=device,
            ),
        }

    def state_dict(self) -> Dict[str, object]:
        return {
            "capacity_steps": self.capacity_steps,
            "episodes": list(self.episodes),
            "current_episode": self.current_episode,
            "num_steps": self.num_steps,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.capacity_steps = int(state["capacity_steps"])
        self.episodes = deque(state["episodes"])
        self.current_episode = list(state.get("current_episode", []))
        self.num_steps = int(state["num_steps"])
