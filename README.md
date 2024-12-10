# O1-CODER
[O1-CODER: An O1 Replication for Coding (Paper)](https://arxiv.org/abs/2412.00154)

## Overview

**O1-CODER** is an attempt to replicate OpenAI's **O1 model**, focused on coding tasks. The approach combines **Reinforcement Learning (RL)** and **Monte Carlo Tree Search (MCTS)** to enhance the model’s **System-2** thinking capabilities, aiming to generate more efficient and logical code.

### Method

The core components of **O1-CODER** are:

1. **Test Case Generator (TCG)**: Automatically generates standardized test cases to evaluate the correctness of the generated code.
2. **Self-Play and Reinforcement Learning**: The model generates reasoning data through self-play, and uses RL and MCTS to iteratively optimize the policy model.
These methods work in an iterative cycle, continuously refining the model to improve systematic reasoning and optimization in coding tasks.

<div align="center">
  <img src="assets/algo.jpeg" width="600" />
</div>

## News

### Latest Updates
#### - 2024-12-10
- Updated the Reward Aggregator

#### - 2024-12-07
- Updated the training code for the process reward model and Test Case Generator.
- Updated the MCTS-based data synthesis code for O1-CODER.

#### - 2024-12-01
- Updated the technical report for O1-CODER.

---

### Planned Updates

TODO: Reinforcement Learning code

TODO: Curated datasets and derived models

---

## License

This work is released under the MIT License. See the [LICENSE](./LICENSE) file for more details. By using this code or associated materials, you agree to comply with the terms outlined in the license.


## Citation

If you use **O1-CODER** or parts of this work in your research or applications, please cite the following paper:
```
@misc{zhang2024o1codero1replicationcoding,
      title={O1-Coder: An O1 Replication for Coding}, 
      author={Yuxiang Zhang and Shangxi Wu and Yuqi Yang and Jiangming Shu and Jinlin Xiao and Chao Kong and Jitao Sang},
      year={2024},
      eprint={2412.00154},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2412.00154}, 
}
```
