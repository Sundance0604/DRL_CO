"""Frozen inference owns a private policy random stream; never an optimizer."""
import torch
from ...strategic.contracts import OBSERVATION_VERSION
from .checkpoints import load_actor
from .networks import sample_action


class FrozenPolicy:
    def __init__(self, checkpoint, policy_seed=0, deterministic=True):
        self.network, self.environment, self.baseline, self.manifest = load_actor(checkpoint)
        self.generator = torch.Generator().manual_seed(policy_seed)
        self.deterministic = deterministic

    def act(self, observation):
        if (observation.version != OBSERVATION_VERSION or
                observation.hubs != tuple(self.environment["action"]["hubs"])):
            raise ValueError("policy observation schema/hub order mismatch")
        action, _, _, _ = sample_action(self.network, observation, self.generator, self.deterministic)
        return tuple(action.tolist())
