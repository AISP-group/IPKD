# IPKD: Interaction-aware Policy Knowledge Distillation for Cooperative Multi-Agent Reinforcement Learning

> Official PyTorch implementation.

IPKD distills the knowledge of a large Transformer-based teacher policy into a lightweight student policy for cooperative MARL, while explicitly modeling agent-to-agent interactions via a learnable correlation matrix.

---

## Overview

Large Transformer-based policies have shown strong performance in cooperative MARL, but their heavy inference cost limits real-world deployment. **IPKD** (Interaction-aware Policy Knowledge Distillation) addresses this by:

1. Training a **Teacher** — an autoregressive Transformer encoder-decoder that models the multi-agent sequential decision paradigm.
2. Distilling its policy into a light **Student** — a compact actor that uses:
   - **SAM** 
   - **IAF** 
---

## Installation

### Dependences
``` Bash
pip install -r requirements.txt
```

### Multi-agent MuJoCo
Following the instructios in https://github.com/openai/mujoco-py and https://github.com/schroederdewitt/multiagent_mujoco to setup a mujoco environment. In the end, remember to set the following environment variables:
``` Bash
LD_LIBRARY_PATH=${HOME}/.mujoco/mujoco200/bin;
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```

### StarCraft II & SMAC
Run the script
``` Bash
bash install_sc2.sh
```
Or you could install them manually to other path you like, just follow here: https://github.com/oxwhirl/smac.

### Google Research Football
Please following the instructios in https://github.com/google-research/football. 

### HORM 
Please following the instructios in - [Highway-env](https://github.com/eleurent/highway-env) 

## How to run
When your environment is ready, you could run shells in the "scripts" folder with algo="ipkd". For example:
``` Bash
./train_football.sh  # run IPKD on Football
```
If you would like to change the configs of experiments, you could modify sh files or look for config.py for more details.


## Performance Comparisons on Cooperative MARL Benchmarks

IPKD consistently outperforms its rivals, indicating its modeling capability for homogeneous-agent tasks (agents are interchangeable).


## Citation
Please cite IPKD: Interaction-aware Policy Knowledge Distillation for Cooperative Multi-Agent Reinforcement Learning if you think this work is helpful for you

