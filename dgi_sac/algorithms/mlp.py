"""SAC with an MLP actor approximately matched to DGI-SAC in parameter count."""

from ..networks import GaussianMLPActor
from ..utils import isolated_rng
from .dgi_sac import make_graph_actor
from .sac import SACAgent


def matched_width(num_inputs, num_actions, target_parameters):
    """Width of a two-layer Gaussian MLP actor whose parameter count is closest to the target."""

    def count(w):  # two hidden layers, then the mean and log-std heads
        return num_inputs * w + w + w * w + w + 2 * (w * num_actions + num_actions)

    return min(range(256, 2049), key=lambda w: abs(count(w) - target_parameters))


class CapacityMatchedSACAgent(SACAgent):
    def build_actor(self, num_inputs, num_actions, action_space):
        # A standard-width actor is created first, as in the paper runs, so that the random-number
        # sequence (and hence the initial weights of the wider actor) matches the published models.
        super().build_actor(num_inputs, num_actions, action_space)
        with isolated_rng():
            target = sum(p.numel() for p in make_graph_actor(self.config, num_inputs, num_actions,
                                                             action_space).parameters())
        width = matched_width(num_inputs, num_actions, target)
        actor = GaussianMLPActor(num_inputs, num_actions, width, action_space)
        actual = sum(p.numel() for p in actor.parameters())
        assert abs(actual - target) / target < 0.01
        self.capacity_match = {"width": width, "target": target, "actual": actual}
        return actor
