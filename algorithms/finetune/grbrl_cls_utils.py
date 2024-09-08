import torch
import copy
import torch.nn.functional as F
from typing import Any, Dict
import numpy as np
from iql import (
    ImplicitQLearning,
    MLP,
    EXP_ADV_MAX,
    TensorBatch,
    ReplayBuffer
)
import torch.nn as nn


class AlphaNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [input_dim, *([hidden_dim] * n_hidden), 1]
        self.a = MLP(dims, output_activation_fn=nn.Sigmoid)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        alpha = self.a(input)
        alpha = alpha-0.5
        
        alpha = torch.clamp(alpha, 0, 1)
        return alpha

def get_back(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                get_back(n[0])

class GRBRL(ImplicitQLearning):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        self.alpha_network = kwargs["alpha_network"]
        self.alpha_optimizer = kwargs["alpha_optimizer"]
        del kwargs["alpha_network"]
        del kwargs["alpha_optimizer"]
        super().__init__(*args, **kwargs)
        self.num_steps = 0

    def _update_policy(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        #back_agent_types = self.alpha_network(observations)
        #alpha_loss = (1-back_agent_types.mean())-policy_loss
        #alpha_loss = policy_loss + 0.01*(policy_loss*(1-back_agent_types.mean()))*(1-(1000000-self.num_steps)/1000000)
        #extra = 0.01*(policy_loss*(1-back_agent_types.mean()))*(1-(1000000-self.num_steps)/1000000)
        
        #alpha_loss = policy_loss
        log_dict["actor_loss"] = policy_loss.item()
        #log_dict["alpha_loss"] = alpha_loss.item()
        
        #print(policy_loss)
        #print("before: ", self.alpha_network.state_dict()['a.net.4.bias'])
        #print("before: ", self.actor.state_dict()['net.net.4.bias'])
        if self.alpha_network is not None:
            self.actor_optimizer.zero_grad()
            self.alpha_optimizer.zero_grad()
            #alpha_loss.backward()
            self.actor_optimizer.step()
            self.alpha_optimizer.step()
        else:
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()
        #print("after: ", self.alpha_network.state_dict()['a.net.4.bias'])
        #print("after: ", self.actor.state_dict()['net.net.4.bias'])
        #import pdb;pdb.set_trace()
        #get_back(alpha_loss.grad_fn)

        if self.actor_lr_schedule is not None:
            self.actor_lr_schedule.step()
            
        self.num_steps += 1

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones
        ) = batch
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        # Update value function
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        # Update actor
        self._update_policy(adv, observations, actions, log_dict)
        
        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        if self.alpha_network is not None:
            state_dict["alpha"] = self.alpha_network.state_dict()
            state_dict["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        super().load_state_dict(state_dict)
        if "alpha" in state_dict:
            self.alpha_network.load_state_dict(state_dict["alpha"])
            self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])

    def partial_load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict, but don't load optimisers."""
        self.qf.load_state_dict(state_dict["qf"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])

        self.actor.load_state_dict(state_dict["actor"])
        if self.actor_lr_schedule is not None:
            self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"]
        
        if "alpha" in state_dict:
            self.alpha_network.load_state_dict(state_dict["alpha"])
            
class ExtendedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        super().__init__(state_dim, action_dim, buffer_size, device)
        self._agent_types = torch.ones((buffer_size, 1),
            dtype=torch.float32, device=device
        )

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._size, size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        agent_types = self._agent_types[indices]
        return [states, actions, rewards, next_states, dones, agent_types]

    def add_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        agent_type: float = 1.0
    ):
        # Use this method to add new data into the replay buffer during fine-tuning.
        self._states[self._pointer] = self._to_tensor(state)
        self._actions[self._pointer] = self._to_tensor(action)
        self._rewards[self._pointer] = self._to_tensor(reward)
        self._next_states[self._pointer] = self._to_tensor(next_state)
        self._dones[self._pointer] = self._to_tensor(done)
        self._agent_types[self._pointer] = agent_type

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)
        