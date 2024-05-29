import numpy as np


def combination_lock(env, _, sample_rate):
    """
    Generates an action for solving a combination lock environment.

    This function interacts with a combination lock environment, where the goal is to
    correctly input a sequence of numbers. The function determines the next number
    to input based on the environment's current step in the combination sequence.
    It uses a sampling mechanism to occasionally introduce random actions.

    Args:
        env (gym.Env): The environment representing the combination lock.
        _ (any): This parameter is ignored and included for compatibility with broader 
                 interfaces that might pass additional arguments.
        sample_rate (float): The probability of selecting a random number instead of the
                correct next number in the sequence.

    Returns:
        np.ndarray: A one-hot encoded action array indicating the selected number to input
                    for the current step. The length of the array matches the length of the
                    combination sequence, with the selected number's index set to 1 and all
                    others set to 0.
    """
    next_number = env.unwrapped.combination[env.unwrapped.combo_step]
    next_num = int(next_number)
    action = np.zeros(len(env.unwrapped.combination))
    if np.random.random() < 1-sample_rate:
        random_int = np.random.randint(len(env.unwrapped.combination))
        while random_int==next_num:
            random_int = np.random.randint(len(env.unwrapped.combination))
        next_num = random_int
    action[next_num] = 1
    return action

def lunar_lander(env, state):
    """
    This modified heuristic originates from Farama Foundation:
    https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/box2d/lunar_lander.py
    https://zenodo.org/records/11232524
    
    It is modified to be imperfect, so there is still a learning opportunity for an RL agent.
    
    The heuristic for
    1. Testing
    2. Demonstration rollout.

    Args:
        env: The environment
        s (list): The state. Attributes:
            state[0] is the horizontal coordinate
            state[1] is the vertical coordinate
            state[2] is the horizontal speed
            state[3] is the vertical speed
            state[4] is the angle
            state[5] is the angular speed
            state[6] 1 if first leg has contact, else 0
            state[7] 1 if second leg has contact, else 0

    Returns:
         a: The heuristic to be fed into the step function defined above to determine the next step and reward.
    """

    angle_targ = state[0] * 0.5 + state[2] * 1.0  # angle should point towards center
    if angle_targ > 0.8:
        angle_targ = 0.8  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.8:
        angle_targ = -0.8
    hover_targ = 0.55 * np.abs(
        state[0]
    )  # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - state[4]) * 0.5 - (state[5]) * 1.0
    hover_todo = (hover_targ - state[1]) * 0.25 - (state[3]) * 0.25

    '''
    if state[6] or state[7]:  # legs have contact
        angle_todo = 0
        hover_todo = (
            -(state[3]) * 0.5
        )  # override to reduce fall speed, that's all we need after contact
    '''

    if env.continuous:
        a = np.array([hover_todo * 15 - 1, -angle_todo * 15])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1
    return a

def lunar_lander_perfect(env, state):
    """
    This heuristic comes from Farama Foundation:
    https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/box2d/lunar_lander.py
    https://zenodo.org/records/11232524
    
    The heuristic for
    1. Testing
    2. Demonstration rollout.

    Args:
        env: The environment
        s (list): The state. Attributes:
            state[0] is the horizontal coordinate
            state[1] is the vertical coordinate
            state[2] is the horizontal speed
            state[3] is the vertical speed
            state[4] is the angle
            state[5] is the angular speed
            state[6] 1 if first leg has contact, else 0
            state[7] 1 if second leg has contact, else 0

    Returns:
         a: The heuristic to be fed into the step function defined above to determine the next step and reward.
    """

    angle_targ = state[0] * 0.5 + state[2] * 1.0  # angle should point towards center
    if angle_targ > 0.4:
        angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4:
        angle_targ = -0.4
    hover_targ = 0.55 * np.abs(
        state[0]
    )  # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - state[4]) * 0.5 - (state[5]) * 1.0
    hover_todo = (hover_targ - state[1]) * 0.5 - (state[3]) * 0.5

    if state[6] or state[7]:  # legs have contact
        angle_todo = 0
        hover_todo = (
            -(state[3]) * 0.5
        )  # override to reduce fall speed, that's all we need after contact

    if env.continuous:
        a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1
    return a