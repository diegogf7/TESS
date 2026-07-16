# All this code is from Claude
"""Single source of truth for the instrument ablation task grid.

Used by the trainers, the SLURM array scripts (via the CLI below), and the
tests that prove every (arm, target, seed, weight, lr) combination is covered
exactly once.

CLI:  python -m src.instrument_v2.ablation_config pretrain <task_id>
      python -m src.instrument_v2.ablation_config finetune <task_id>
      python -m src.instrument_v2.ablation_config counts
Prints shell-evalable KEY=VALUE lines for the given array task.
"""

import sys
from itertools import product

ARMS = ("random", "jepa", "supcon", "hybrid")
TARGETS = ("camera", "camccd")
SEEDS = (0, 1, 2)
BACKBONE_LRS = ("1e-4", "3e-4", "1e-3")
HYBRID_WEIGHTS = ("0.1", "0.5", "1.0")


def pretrain_tasks():
    """supcon x 3 seeds + hybrid x 3 weights x 3 seeds = 12 tasks."""
    tasks = [{"OBJECTIVE": "supcon", "SEED": s, "CONTRASTIVE_WEIGHT": ""}
             for s in SEEDS]
    tasks += [{"OBJECTIVE": "hybrid", "SEED": s, "CONTRASTIVE_WEIGHT": w}
              for w, s in product(HYBRID_WEIGHTS, SEEDS)]
    return tasks


def finetune_tasks():
    """4 arms x 2 targets x 3 seeds x 3 backbone lrs = 72 tasks."""
    return [{"INIT_ARM": a, "TARGET": t, "SEED": s, "BACKBONE_LR": lr}
            for a, t, s, lr in product(ARMS, TARGETS, SEEDS, BACKBONE_LRS)]


def map_pretrain_task(i):
    tasks = pretrain_tasks()
    if not 0 <= i < len(tasks):
        raise IndexError(f"pretrain task id {i} out of range 0..{len(tasks) - 1}")
    return tasks[i]


def map_finetune_task(i):
    tasks = finetune_tasks()
    if not 0 <= i < len(tasks):
        raise IndexError(f"finetune task id {i} out of range 0..{len(tasks) - 1}")
    return tasks[i]


if __name__ == "__main__":
    kind = sys.argv[1]
    if kind == "counts":
        print(f"PRETRAIN_TASKS={len(pretrain_tasks())}")
        print(f"FINETUNE_TASKS={len(finetune_tasks())}")
    else:
        task = (map_pretrain_task if kind == "pretrain" else map_finetune_task)(int(sys.argv[2]))
        for key, value in task.items():
            print(f"{key}={value}")
