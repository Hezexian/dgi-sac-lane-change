"""The five methods compared in the paper; configs/<name>.yaml selects one with `algorithm: <name>`.

    dgi_sac   DGI-SAC (proposed)       dgi_sac.py
    sac       SAC                      sac.py
    mlp       capacity-matched MLP     mlp.py
    td3       TD3                      td3.py
    ddpg      DDPG                     ddpg.py
"""

from .ddpg import DDPGAgent
from .dgi_sac import DGISACAgent
from .mlp import CapacityMatchedSACAgent
from .sac import SACAgent
from .td3 import TD3Agent

AGENTS = {"dgi_sac": DGISACAgent, "sac": SACAgent, "mlp": CapacityMatchedSACAgent, "td3": TD3Agent,
          "ddpg": DDPGAgent}


def make_agent(config, seed, action_space, device="cpu", threads=1):
    return AGENTS[config["algorithm"]](config, seed, action_space, device=device, threads=threads)


__all__ = ["make_agent", "AGENTS", "DGISACAgent", "SACAgent", "CapacityMatchedSACAgent", "TD3Agent", "DDPGAgent"]
