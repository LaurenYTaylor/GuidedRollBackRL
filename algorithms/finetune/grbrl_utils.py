# source: https://github.com/gwthomas/IQL-PyTorch
# and (CORL): https://github.com/tinkoff-ai/CORL/blob/main/algorithms/finetune/iql.py
# IQL: https://arxiv.org/pdf/2110.06169.pdf
# GRBRL: (Under Review)
####################################################################################################
# Docstrings were generated using ChatGPT (GPT3.5).                                                #
# OpenAI. (2024). Docstrings for GuidedRollBackRL algorithm. Retrieved from ChatGPT on 22 May 2024.#
####################################################################################################
import torch
import numpy as np
from collections import deque
from pathlib import PosixPath, Path
from goal_horizon_fns import goal_dist_calc
from torch import nn
import guide_heuristics
from variance_learner import StateDepFunction, VarianceLearner
from iql import (
    DeterministicPolicy,
    GaussianPolicy,
    ImplicitQLearning,
    TwinQ,
    ValueFunction
)

horizon_str = ""  # this is set in grbrl_w_iql.py


def add_grbrl_metrics(eval_log, config):
    """
    Add GRBRL metrics to the evaluation log.

    Parameters
    ----------
    eval_log : dict
        The evaluation log to which GRBRL metrics will be added.
    config : GrbrlTrainConfig
        The configuration parameters containing GRBRL metrics.

    Returns
    -------
    dict
        The evaluation log with GRBRL metrics added.
    """
    eval_log["eval/grbrl/curriculum_stage_idx"] = config.curriculum_stage_idx
    eval_log["eval/grbrl/curriculum_stage"] = config.curriculum_stage
    if isinstance(config.best_eval_score, dict):
        eval_score = max(list(config.best_eval_score.values()))
    else:
        eval_score = config.best_eval_score
    eval_log["eval/grbrl/best_eval_score"] = eval_score
    eval_log["eval/grbrl/mean_horizon_reached"] = -config.mean_horizon_reached
    eval_log["eval/grbrl/mean_agent_type"] = config.eval_mean_agent_type
    return eval_log


def horizon_update_callback(config, eval_reward, N):
    """
    Update the curriculum stage and agent type threshold based on the evaluation reward.

    This function updates the curriculum stage and agent type threshold based on the evaluation score achieved
    during training. It calculates the rolling mean of evaluation scores and checks if it surpasses
    the previous best evaluation score by a certain tolerance margin.

    Parameters
    ----------
    config : GrbrlTrainConfig
        The configuration parameters for the GRBRL training process.
    eval_reward : float
        The evaluation reward obtained during training.

    Returns
    -------
    GrbrlTrainConfig
        The updated configuration parameters after horizon update.
    """
    prev_best = -np.inf
    eval_reward = -config.mean_horizon_reached
    
    if config.agent_type_stage in config.best_eval_score:
        prev_best = config.best_eval_score[config.agent_type_stage]
    
    if config.best_eval_score[0] < 0:
        score_with_tolerance = -N+config.tolerance*(config.best_eval_score[0]+N)
    else:
        score_with_tolerance = config.tolerance*config.best_eval_score[0]
    
    if (
        eval_reward >= config.best_eval_score[0]
    ):
        config.best_eval_score[config.agent_type_stage] = eval_reward
        if config.rolled_back:
            config.agent_type_stage = max(list(config.best_eval_score.keys()))
            config.rolled_back = False
        else:
            config.agent_type_stage = min(1.0, config.agent_type_stage+config.learner_frac)
    elif config.enable_rollback and (eval_reward < score_with_tolerance):
        config.best_eval_score[config.agent_type_stage] = eval_reward
        if config.agent_type_stage != min(list(config.best_eval_score.keys())):
            best_prevs = sorted(config.best_eval_score.items(), key=lambda x: x[1])[-1]
            best_prev = best_prevs[0]
            if best_prev == config.agent_type_stage:
                best_prev = best_prevs[1]
            config.agent_type_stage = best_prev
            config.rolled_back = True
    print(f"{config.seed}: {score_with_tolerance}/{eval_reward}: curr best: {prev_best}, eval rew: {eval_reward}, new agent type: {config.agent_type_stage}")
    return config


def load_guide(trainer, pretrained):
    """
    Load a pretrained guide model for GRBRL.

    This function loads a pretrained guide model from a file path, if one was provided.
    The guide model is then set to evaluation mode.

    Parameters
    ----------
    trainer : nn.Module
        The trainer model containing the guide model.
    pretrained : Union[PosixPath, None]
        The pretrained guide model or the file path to load the pretrained model from.

    Returns
    -------
    nn.Module
        The pretrained guide model loaded and set to evaluation mode.
    """
    if not isinstance(pretrained, PosixPath):
        return pretrained
    try:
        trainer.load_state_dict(torch.load(pretrained))
    except RuntimeError:
        trainer.load_state_dict(
            torch.load(pretrained, map_location=torch.device("cpu"))
        )

    guide = trainer.actor
    guide.eval()
    return guide


def prepare_finetuning(init_horizon, mean_return, config):
    """
    Prepare the configuration for online fine-tuning of a learner using a guide policy.

    This function initializes the configuration parameters required for online fine-tuning of a learner
    using a guide policy. It generates curriculum stages based on the initial horizon and the
    number of curriculum stages specified in the configuration. It sets up agent types if not disabled
    and initializes other relevant configuration parameters such as the current curriculum stage, agent
    type stage, best evaluation score (initialised to -np.inf), and rolling mean rewards.

    Parameters
    ----------
    init_horizon : int
        The initial horizon for the online fine-tuning process.
    config : GrbrlTrainConfig
        The configuration parameters for the GRBRL training process.

    Returns
    -------
    GrbrlTrainConfig
        The updated configuration parameters prepared for online fine-tuning.
    """
    config.curriculum_stage_idx = 0
    if config.n_curriculum_stages == 1:
        config.agent_type_stage = 1
    if config.learner_frac < 0:
        H = int(init_horizon) 
        guide_sample = config.sample_rate
        learner_sample = (1-config.correct_learner_action)
        config.learner_frac = 1-(((config.tolerance)**(1/H)*guide_sample-(1-learner_sample))/(guide_sample-(1-learner_sample)))
    config.agent_type_stage = config.learner_frac
    config.best_eval_score = {}
    config.best_eval_score[0] = -init_horizon
    config.rolled_back = False
    return config

def get_var_predictor(env, config, max_steps, guide, n_updates = 10000):
    """
    If the horizon function specified in the configuration is "variance",
    this trains or loads a state variance predictor for use in online fine-tuning,
    in determining if a guide or learner agent should be used to take an action.

    Parameters
    ----------
    env : gym.Env
        The Gym environment in which the predictor will be trained or used.
    config : GrbrlTrainConfig
        The configuration parameters for the GRBRL training process.
    max_steps : int
        The maximum number of steps to run the environment for during training.
    guide : nn.Module
        The guide function or model used for guiding the learner.
    n_updates : int
        The number of updates to the variance predictor model parameters.

    Returns
    -------
    GrbrlTrainConfig
        The updated configuration parameters containing the trained or loaded variance predictor.
    """
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    if config.horizon_fn == "variance":
        var_actor = guide
        try:
            vf = StateDepFunction(state_dim)
            mf = StateDepFunction(state_dim)
            fn = f"grbrl-CORL/algorithms/finetune/var_functions/{env.unwrapped.spec.name}_guide_{n_updates}_{str(config.variance_learn_frac).replace('.','-')}"
            vf.load_state_dict(torch.load(fn+"_vf.pt"))
            mf.load_state_dict(torch.load(fn+"_mf.pt"))
            v_learner = VarianceLearner(state_dim, action_dim, config, var_actor)
            v_learner.vf = vf
            v_learner.mf = mf
            #v_learner.test_model(env, max_steps, guide)
        except FileNotFoundError:
            vf = VarianceLearner(state_dim, action_dim, config, var_actor).run_training(env, max_steps, guide, n_updates=n_updates, evaluate=True)
        config.vf = vf.eval()
    return config


def make_actor(config, state_dim, action_dim, max_action, device=None, max_steps=None):
    """
    Create an actor (guide or learner) for Implicit Q-Learning (IQL) based on the provided configuration.

    This function constructs an actor (guide or learner) for Implicit Q-Learning (IQL) based on the specified configuration
    parameters. It initializes the neural networks for the Q-function, value function, and actor, as well
    as the corresponding optimizers. The actor can be either deterministic or Gaussian based on the
    configuration.

    Parameters
    ----------
    config : GrbrlTrainConfig
        The configuration parameters for the GRBRL training process.
    state_dim : int
        The dimensionality of the state space.
    action_dim : int
        The dimensionality of the action space.
    max_action : float
        The maximum value of the action space.
    device : torch.device, optional
        The device on which to construct the networks and perform computations (default is None, which
        implies the device specified in the configuration).
    max_steps : int, optional
        The maximum number of steps in the environment (default is None).

    Returns
    -------
    ImplicitQLearning
        An instance of the ImplicitQLearning class configured based on the provided parameters.
    """
    if device is None:
        device = config.device
    q_network = TwinQ(state_dim, action_dim).to(device)
    v_network = ValueFunction(state_dim).to(device)
    actor = (
        DeterministicPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
        if config.iql_deterministic
        else GaussianPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
    ).to(device)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": max_steps,
    }
    return ImplicitQLearning(**kwargs)

def get_guide_agent(config, trainer, state_dim, action_dim, max_action):
    """
    Determine and create the guide agent based on the provided configuration.

    This function determines what type of guide agent to use based on the configuration parameters. If a
    guide heuristic function is specified in the configuration, it retrieves the corresponding guide
    heuristic function. If no trainer is provided, it creates a new actor as the guide agent and loads
    a pretrained policy into the actor if specified in the configuration.
    Otherwise, it uses the actor from the trainer pre-trained offline as the guide agent.

    Parameters
    ----------
    config : GrbrlTrainConfig
        The configuration parameters for the GRBRL training process.
    trainer : nn.Module or None
        The trainer model containing the guide agent, or None if no trainer is provided.
    state_dim : int
        The dimensionality of the state space.
    action_dim : int
        The dimensionality of the action space.
    max_action : float
        The maximum value of the action space.

    Returns
    -------
    Tuple[nn.Module, Optional[nn.Module]]
        A tuple containing the guide agent and the trainer for the guide agent, if applicable.

    """
    if config.guide_heuristic_fn is not None:
        guide = getattr(guide_heuristics, config.guide_heuristic_fn)
        guide_trainer = None
    elif trainer is None:
        guide_trainer = make_actor(config, state_dim, action_dim, max_action)
        guide = load_guide(guide_trainer, Path(config.pretrained_policy_path))
        guide.eval()
    else:
        guide = trainer.actor
        guide_trainer = trainer
        guide.eval()
    return guide, guide_trainer

def get_learning_agent(config, guide_trainer, init_horizon, mean_return, state_dim, action_dim, max_action):
    """
    Create a learning agent for online fine-tuning based on the provided configuration.

    Parameters
    ----------
    config : GrbrlTrainConfig
        The configuration parameters for the GRBRL training process.
    guide_trainer : nn.Module or None
        The trainer model containing the guide agent, or None if no guide trainer is provided.
    init_horizon : int
        The initial horizon for curriculum stage 1.
    state_dim : int
        The dimensionality of the state space.
    action_dim : int
        The dimensionality of the action space.
    max_action : float
        The maximum value of the action space.

    Returns
    -------
    Tuple[nn.Module, GrbrlTrainConfig]
        A tuple containing the learning agent trainer and the updated configuration parameters for
        fine-tuning.
    """
    trainer = make_actor(config, state_dim, action_dim, max_action)
    if config.n_curriculum_stages == 1 and config.guide_heuristic_fn is None:
        state_dict = guide_trainer.state_dict()
        trainer.partial_load_state_dict(state_dict)
    trainer.total_it = config.offline_iterations # iterations done so far
    config = prepare_finetuning(init_horizon, mean_return, config)
    return trainer, config

def variance_horizon(_, s, _e, config):
    """
    Determine whether to use the learner or guide based on the state variance.
    Unused parameter placeholders ensure the horizon functions have the same signature.

    Parameters
    ----------
    _ : Any
        Time step, not used.
    s : numpy.ndarray
        The current state of the environment.
    _e : Any
        Environment, not used.
    config : GrbrlTrainConfig
        The configuration parameters for the GRBRL training process.

    Returns
    -------
    Tuple[bool, float]
        A tuple containing a boolean indicating whether to use the learner and the calculated state
        variance.

    """
    use_learner = False
    var = config.vf(torch.Tensor(s))

    if np.isnan(config.curriculum_stage):
        return True, var
    if (
        (var <= config.curriculum_stage
         or config.curriculum_stage_idx == (config.n_curriculum_stages-1))
        and config.ep_agent_type <= config.agent_type_stage
    ):
        use_learner = True
    return use_learner, var

def timestep_horizon(step, _s, _e, config):
    """
    Determine whether to use the learner or guide based on the timestep.
    Unused parameter placeholders ensure the horizon functions have the same signature.

    Parameters
    ----------
    step : int
        The current timestep.
    _s : Any
        State, not used.
    _e : Any
        Env, not used.
    config : GrbrlTrainConfig
        The configuration parameters for the GRBRL training process.

    Returns
    -------
    Tuple[bool, int]
        A tuple containing a boolean indicating whether to use the learner and the current timestep.

    """
    use_learner = False
    if np.isnan(config.curriculum_stage):
        return True, step
    if (
        (step >= config.curriculum_stage
         or config.curriculum_stage_idx == (config.n_curriculum_stages-1))
        and config.ep_agent_type <= config.agent_type_stage
    ):
        use_learner = True
    return use_learner, step


def goal_distance_horizon(_t, s, env, config):
    """
    Determine whether to use the learner or guide based on the distance from the goal.
    Unused parameter placeholders ensure the horizon functions have the same signature.

    Parameters
    ----------
    _t : Any
        Time step, not used.
    s : numpy.ndarray
        The current state of the environment.
    env : gym.Env
        The environment.
    config : GrbrlTrainConfig
        The configuration parameters for the GRBRL training process.

    Returns
    -------
    Tuple[bool, float]
        A tuple containing a boolean indicating whether to use the learner and the calculated goal
        distance.

    """
    use_learner = False
    goal_dist = goal_dist_calc(s, env)
    if np.isnan(config.curriculum_stage):
        return True, goal_dist
    if (
        (goal_dist <= config.curriculum_stage
        or config.curriculum_stage_idx == (config.n_curriculum_stages-1))
        and config.ep_agent_type <= config.agent_type_stage
    ) or (
        goal_dist > config.all_curriculum_stages[-1]
        and config.ep_agent_type <= config.agent_type_stage
    ):
        use_learner = True
    return use_learner, goal_dist


def max_accumulator(v):
    return np.max(v)


def mean_accumulator(v):
    return np.mean(v)


def max_to_min_curriculum(init_horizon, n_curriculum_stages):
    """
    Generates a max to min curriculum (for time step).
    """
    init_horizon = 0
    return np.linspace(init_horizon, 0, n_curriculum_stages)


def min_to_max_curriculum(init_horizon, n_curriculum_stages):
    """
    Generates a min to max curriculum (for goal distance and variance).
    """
    return np.linspace(0, init_horizon, n_curriculum_stages)


HORIZON_FNS = {
    "time_step": {
        "horizon_fn": timestep_horizon,
        "accumulator_fn": mean_accumulator,
        "generate_curriculum_fn": max_to_min_curriculum,
    },
    "goal_dist": {
        "horizon_fn": goal_distance_horizon,
        "accumulator_fn": mean_accumulator,
        "generate_curriculum_fn": min_to_max_curriculum,
    },
    "variance": {
        "horizon_fn": variance_horizon,
        "accumulator_fn": mean_accumulator,
        "generate_curriculum_fn": min_to_max_curriculum,
    }
}


def accumulate(vals):
    return HORIZON_FNS[horizon_str]["accumulator_fn"](vals)


def learner_or_guide_action(state, step, env, learner, guide, config, device, eval=False):
    """
    Determine whether to use the learner or guide policy to choose an action.
    It calculates the horizon and decides whether to use the learner based on the
    specified curriculum stage. Then, it selects an action either from the learner
    or guide policy accordingly.

    Parameters
    ----------
    state : numpy.ndarray
        The current state of the environment.
    step : int
        The current env timestep.
    env : gym.Env
        The environment.
    learner : nn.Module
        The learner policy model.
    guide : nn.Module
        The guide policy.
    config : GrbrlTrainConfig
        The configuration parameters for the GRBRL training process.
    device : torch.device
        The device on which the model computations are performed.
    eval : bool, optional
        Flag indicating whether evaluation mode is active (default is False).
        If True, the learner or guide's evaluation mode will be enabled.

    Returns
    -------
    Tuple[Any, bool, float]
        A tuple containing the chosen action, a boolean indicating whether the learner is used,
        and the calculated horizon.
    """
    if guide is None:
        horizon = step
        use_learner = True
    else:
        if ((np.random.random() <= config.agent_type_stage) and 
            (config.ep_agent_type <= config.agent_type_stage)):
            use_learner = True
        else:
            use_learner = False
        horizon = step

    if use_learner:
        # other than the actual learner, this may also be the training guide policy,
        # or the guide being evaluated before online training starts
        if not (isinstance(learner, GaussianPolicy) or isinstance(learner, DeterministicPolicy)):
            action = learner(env, state, config.sample_rate)
        else:
            if eval:
                action = learner.act(state, device)
            else:
                action = learner(
                    torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
                )
    else:
        if not isinstance(guide, GaussianPolicy):
            action = guide(env, state, config.sample_rate)
        else:
            action = guide.act(state, device)
        if not eval:
             action = torch.tensor(action)
    return action, use_learner, horizon
