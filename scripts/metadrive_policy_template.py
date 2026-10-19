"""Copy and edit this policy to test your modified model.

Usage: --policy scripts.metadrive_policy_template:load_policy --checkpoint PATH
The default implementation loads the current repository's Dreamer model.
"""
from scripts.test_metadrive_render import DreamerPolicy


class MyPolicy(DreamerPolicy):
    def load_model(self, config, obs_space, action_space, checkpoint):
        # MODEL INTERFACE: replace this with your model constructor and weight loader.
        # Return the loaded model in eval mode on self.device.
        return super().load_model(config, obs_space, action_space, checkpoint)

    def reset(self, seed):
        # EPISODE INTERFACE: clear any recurrent state/history here.
        super().reset(seed)

    def act(self, obs):
        # INFERENCE INTERFACE: obs['state'] is the training LiDAR state (259 floats).
        # Return np.ndarray shape (2,): [steering, throttle/brake], in [-1, 1].
        # Keep the model's recurrent state between steps; never reset it here.
        return super().act(obs)


def load_policy(config, obs_space, action_space, checkpoint, device):
    """Factory required by --policy module:factory. No optimizer updates are run."""
    return MyPolicy(config, obs_space, action_space, checkpoint, device)
