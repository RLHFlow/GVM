<div align="center">

# DIBS - Dynamic Inference Budget Scheduling
[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2505.02391) [![Github](https://img.shields.io/badge/GVM-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/RLHFlow/GVM)
</div>

## Table of Contents
- [DIBS - Dynamic Inference Budget Scheduling](#dibs---dynamic-inference-budget-scheduling)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Environment Setup](#environment-setup)
  - [Experiments Running](#experiments-running)
  - [Acknowledgement](#acknowledgement)

## Introduction
Chain-of-thought (CoT) reasoning in large language models (LLMs) can be formalized as a latent variable problem, where the model needs to generate intermediate reasoning steps. While prior approaches such as iterative reward-ranked fine-tuning (RAFT) have relied on such formulations, they typically apply uniform inference budgets across prompts, which fails to account for variability in difficulty and convergence behavior. This work identifies the main bottleneck in CoT training as inefficient stochastic gradient estimation due to static sampling strategies. We propose GVM-RAFT, a prompt-specific Dynamic Sample Allocation Strategy designed to minimize stochastic gradient variance under a computational budget constraint. The method dynamically allocates computational resources by monitoring prompt acceptance rates and stochastic gradient norms, ensuring that the resulting gradient variance is minimized. Our theoretical analysis shows that the proposed dynamic sampling strategy leads to accelerated convergence guarantees under suitable conditions. Experiments on mathematical reasoning show that GVM-RAFT achieves a 2-4 $\times$ speedup and considerable accuracy improvements over vanilla RAFT. The proposed dynamic sampling strategy is general and can be incorporated into other reinforcement learning algorithms, such as GRPO, leading to similar improvements in convergence and test accuracy.

<p align="center">
  <img src="figures/main_fig.png" width="85%" />
  <img src="figures/alg.png" width="85%">
</p>

**Main Takeaways**
1. We revisit the EM framework and RAFT in the context of CoT reasoning, and identify that a major limitation of current approaches lies in inefficient stochastic gradient estimation caused by uniform and static sampling strategies (i.e., best-of-n sampling), which fail to account for prompt-specific difficulty and convergence behavior.
2. Motivated by the goal of minimizing the variance of stochastic gradient, we propose a dynamic sampling strategy that adaptively allocates computational resources based on prompt hardness and gradient norms. Our approach provides both intuitive theoretical insight and rigorous convergence guarantees, establishing a principled framework for efficient on-policy sampling under computational budget constraints.
3. We apply our method to both RAFT++ and GRPO algorithms with real-world experiments on mathematical reasoning tasks. Our results demonstrate that the proposed approach achieves 2-4 $\times$ speedup in convergence rate and also considerably improve the final test accuracy. 


<p align="center">
  <img src="figures/res.png" width="75%" />
</p>


<p align="center">
  <img src="figures/res_fig.png" width="75%" />
</p>

## Environment Setup
1. Create a new environment.
   ```bash
   python -m venv ~/.python/gvm
   source ~/.python/gvm/bin/activate
   # You can also use conda 
   #conda create -n gvm python==3.10
   #conda activate gvm
   ```
2. Install dependencies
   ```bash
   pip install pip --upgrade
   pip install uv
   git clone https://github.com/RLHFlow/GVM.git
   cd GVM/
   python -m uv pip install -r requirements.txt
   ```

## Experiments Running
1. Prepare the training and test datasets.
    ```bash
    python runs/data_preprocess/math_dataset.py
    python runs/data_preprocess/numina_process.py
    ```
2. Start the training loop.
   ```bash
   bash runs/scripts/run_em.sh
   bash runs/scripts/run_raft.sh
   bash runs/scripts/run_grpo.sh
   ```

## Acknowledgement
We greatly thanks [verl](https://github.com/volcengine/verl) for providing the awesome codebase!