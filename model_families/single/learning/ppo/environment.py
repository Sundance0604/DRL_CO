"""Learning-side reset/step adapter. Failures raise; they are never terminals."""


class PPOEnvironment:
    def __init__(self, episode, reward_scale):
        self.episode, self.reward_scale = episode, reward_scale

    def reset(self):
        return self.episode.reset(), {}

    def step(self, action):
        transition = self.episode.step(tuple(float(x) for x in action))
        return (transition.observation, transition.reward/self.reward_scale,
                transition.terminated, False,
                dict(reward_USD=transition.reward, transition_id=transition.transition_id))
