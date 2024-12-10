# Reward Aggregator

## Introduction

This project provides an offline reward update method, which is suitable for reward annotation tasks in reinforcement learning, especially for reward updates in methods like Iterative DPO.

The core component of this project is `RewardAggregater`, which supports flexible configuration of the following parameters:

- `phi` function (for reward aggregation)
- `alpha` function (time decay factor)
- `gamma` (discount factor)

This tool can compute **intermediate rewards** and **outcome rewards** based on different model outputs.

## Usage

The `rewards_test.py` file provides a complete usage example. It demonstrates how to initialize the `RewardAggregater`, load test examples, and compute rewards.

Simply run the `rewards_test.py` script:

```bash
python rewards_test.py
```

This will compute and print the rewards for the predefined test cases provided in `examples.py`.



## Features

- **Reward Aggregation**: The `phi` function aggregates the final and intermediate rewards, computing a smoother cumulative reward.
- **Time Decay**: Supports adjustable linear decay factor `alpha(t)`, allowing the reward weight to be adjusted according to the time step.
- **Offline Update**: Supports offline reward calculation and updates, enabling batch processing of reward signals without requiring online training.
- **Flexible Configuration**: The `RewardAggregater` class allows users to customize reward computation methods, reward decay functions, discount factors, and other parameters.

## Planned Updates

- **Reinforcement Learning Training Code**
- **Online Reward Update**: Plan to add support for online reward updates in future versions.
