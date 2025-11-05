import gymnasium as gym
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

"""Fuzzy logic controller for car environment in gymnasium.
    The car must build up momentum to get out of the valley to reach the goal on the top of the hill.
    The car has 3 inputs: position, velocity and previous action.
    The fuzzy controller controls car's throttle, allowing it to either accelerate to the left or to the right.
    A set of fuzzy rules determine how the throttle is applied.
"""


# INPUTS
# (range ≈ -1.2 … 0.6) - observation Space in gymnasium
pos = ctrl.Antecedent(np.arange(-1.2, 0.61, 0.01), "pos")
# (range ≈ -0.07 … 0.07)
vel = ctrl.Antecedent(np.arange(-0.07, 0.071, 0.001), "vel")
# previous action of throttle
prev_act = ctrl.Antecedent(np.arange(-1.0, 1.01, 0.01), "prev_act")

# membership functions (poor/average/good‑like)
# trapmf trapez, trimf trójkąt

pos["far_left"] = fuzz.trapmf(pos.universe, [-1.2, -1.2, -1.0, -0.8])
pos["mid_left"] = fuzz.trimf(pos.universe, [-1.0, -0.8, -0.5])
pos["center"] = fuzz.trimf(pos.universe, [-0.6, -0.2, 0.2])
pos["mid_right"] = fuzz.trimf(pos.universe, [0.0, 0.3, 0.5])
pos["goal"] = fuzz.trapmf(pos.universe, [0.4, 0.5, 0.6, 0.6])

vel["neg_fast"] = fuzz.trapmf(vel.universe, [-0.07, -0.07, -0.05, -0.03])
vel["neg_slow"] = fuzz.trimf(vel.universe, [-0.04, -0.02, 0.0])
vel["zero"] = fuzz.trimf(vel.universe, [-0.005, 0.0, 0.005])
vel["pos_slow"] = fuzz.trimf(vel.universe, [0.0, 0.02, 0.04])
vel["pos_fast"] = fuzz.trapmf(vel.universe, [0.03, 0.05, 0.07, 0.07])

prev_act["very_left"] = fuzz.trapmf(prev_act.universe, [-1.0, -1.0, -0.8, -0.6])
prev_act["little_left"] = fuzz.trimf(prev_act.universe, [-0.7, -0.4, -0.1])
prev_act["neutral"] = fuzz.trimf(prev_act.universe, [-0.15, 0.0, 0.15])
prev_act["little_right"] = fuzz.trimf(prev_act.universe, [0.1, 0.4, 0.7])
prev_act["very_right"] = fuzz.trapmf(prev_act.universe, [0.6, 0.8, 1.0, 1.0])

# OUTPUT
# (continuous action in [-1, 1])
throttle = ctrl.Consequent(np.arange(-1.0, 1.01, 0.01), "throttle")

throttle["very_left"] = fuzz.trapmf(throttle.universe, [-1.0, -1.0, -0.8, -0.6])
throttle["little_left"] = fuzz.trimf(throttle.universe, [-0.7, -0.4, -0.1])
throttle["neutral"] = fuzz.trimf(throttle.universe, [-0.15, 0.0, 0.15])
throttle["little_right"] = fuzz.trimf(throttle.universe, [0.1, 0.4, 0.7])
throttle["very_right"] = fuzz.trapmf(throttle.universe, [0.6, 0.8, 1.0, 1.0])

# rules

rules = [
    ctrl.Rule(
        pos["far_left"] & (vel["neg_fast"] | vel["neg_slow"]) & prev_act["very_left"],
        throttle["very_left"],
    ),
    ctrl.Rule(
        pos["mid_left"] & (vel["neg_slow"] | vel["zero"]) & prev_act["little_left"],
        throttle["little_left"],
    ),
    ctrl.Rule(pos["center"] & vel["zero"] & prev_act["neutral"], throttle["very_left"]),
    ctrl.Rule(
        (pos["mid_right"] | pos["goal"])
        & (vel["pos_slow"] | vel["zero"])
        & prev_act["little_right"],
        throttle["little_right"],
    ),
    ctrl.Rule(
        pos["goal"] & (vel["pos_fast"] | vel["pos_slow"]) & prev_act["very_right"],
        throttle["very_right"],
    ),
    ctrl.Rule(
        pos["far_left"] & vel["pos_slow"] & prev_act["little_right"],
        throttle["little_left"],
    ),
    ctrl.Rule(
        pos["goal"] & vel["neg_slow"] & prev_act["little_left"],
        throttle["little_right"],
    ),
    ctrl.Rule(
        pos["center"] & vel["neg_fast"] & prev_act["very_left"], throttle["little_left"]
    ),
    ctrl.Rule(
        pos["center"] & vel["pos_fast"] & prev_act["very_right"],
        throttle["little_right"],
    ),
    ctrl.Rule(
        pos["center"] & vel["neg_slow"] & prev_act["neutral"], throttle["very_left"]
    ),
    ctrl.Rule(
        pos["center"] & vel["neg_slow"] & prev_act["very_right"], throttle["very_left"]
    ),
    ctrl.Rule(
        pos["mid_right"] & vel["pos_slow"] & prev_act["very_right"],
        throttle["little_right"],
    ),
    ctrl.Rule(
        pos["goal"] & vel["pos_slow"] & prev_act["little_right"],
        throttle["little_right"],
    ),
    ctrl.Rule(pos["center"] & vel["pos_slow"], throttle["very_right"]),
    ctrl.Rule(pos["center"] & vel["neg_slow"], throttle["very_left"]),
]


fuzzy_ctrl = ctrl.ControlSystem(rules)
fuzzy_sim = ctrl.ControlSystemSimulation(fuzzy_ctrl)


def fuzzy_action(position, velocity, prev_action):

    fuzzy_sim.reset()
    fuzzy_sim.input["pos"] = float(position)
    fuzzy_sim.input["vel"] = float(velocity)
    fuzzy_sim.input["prev_act"] = float(prev_action)

    fuzzy_sim.compute()

    throttle_val = fuzzy_sim.output.get("throttle", 0.0)
    return float(throttle_val)


env = gym.make(
    "MountainCarContinuous-v0", render_mode="human"
)  # change to None for headless

obs, _ = env.reset(
    seed=1
)  # obs = [position, velocity], python "_" bo dostaje tuple i nie uzywa sie
position, velocity = obs
prev_action = 0.0
total_reward = 0.0
steps = 0

done = False
while not done:

    throttle = fuzzy_action(position, velocity, prev_action)
    action = np.array([throttle])

    print(
        f"Step {steps:3d} | "
        f"pos={position: .4f} | vel={velocity: .4f} | "
        f"prev_throttle={prev_action: .4f} → throttle={throttle: .4f}"
    )

    obs, reward, terminated, truncated, _ = env.step(action)
    position, velocity = obs
    total_reward += reward
    steps += 1

    prev_action = action.item()

    done = (
        terminated or truncated
    )  # Truncation: The length of the episode is 999 from gymnasium

env.close()
print(f"\ntaken {steps} steps – total reward: {total_reward:.2f}")
