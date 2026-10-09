"""Frozen CPU/serial prototype configuration, independent of physical parameters."""
from pydantic import BaseModel, ConfigDict, Field


class PPOConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    updates: int = Field(default=10, ge=1)
    episodes_per_update: int = Field(default=4, ge=1)
    epochs: int = Field(default=4, ge=1)
    minibatch_size: int = Field(default=64, ge=1)
    hidden_size: int = Field(default=64, ge=8, le=512)
    learning_rate: float = Field(default=3e-4, gt=0)
    clip_ratio: float = Field(default=.2, gt=0, lt=1)
    gae_lambda: float = Field(default=.95, ge=0, le=1)
    gamma: float = Field(default=1., ge=1, le=1)
    value_coefficient: float = Field(default=.5, ge=0)
    entropy_coefficient: float = Field(default=0., ge=0)
    reward_scale: float = Field(default=1000., gt=0)
    max_grad_norm: float = Field(default=.5, gt=0)
    target_kl: float = Field(default=.03, gt=0)
    training_seed: int = Field(default=0, ge=0, le=2**31-1)
    episode_timeout_seconds: float = Field(default=600., gt=0)
    validation_every: int = Field(default=1, ge=1)

    def resume_identity(self):
        # Only extending the number of full updates is permitted on resume.
        return self.model_dump(exclude={"updates"})
