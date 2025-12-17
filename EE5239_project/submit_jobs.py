import os
import subprocess
from config import Config

# ----------------------------
# Define your parameter sweeps
# ----------------------------

default_cfg = vars(Config().parse())

param_list = ["name", "lr", "batch_size","LR_sch","n_train","n_val","peft","lora_rank", "model", "wandb_group", "target_layers"]

n = 256
bs = 2
lr_list = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
lr_def = 5e-4
rank_def = 16
rank_list = [1,8,64,128]
group1 = "LR_search_0"
group2 = "rank_search_0"
configs = []

for i, lr in enumerate(lr_list):
    configs.append({"name": f"full_lr_{lr:.0e}", "peft": "None", "batch_size": bs, "model": "l", "n_train": n, "lr": lr,"wandb_group":group1,})
    configs.append({"name": f"lora_rank_{rank_def}_lr_{lr:.0e}", "peft": "lora", "batch_size": bs, "model": "l", "n_train": n, "lr": lr,"wandb_group":group1,})
    configs.append({"name": f"dora_rank_{rank_def}_lr_{lr:.0e}", "peft": "dora", "batch_size": bs, "model": "l", "n_train": n, "lr": lr,"wandb_group":group1,})
    configs.append({"name": f"pissa_rank_{rank_def}_lr_{lr:.0e}", "peft": "pissa", "batch_size": bs, "model": "l", "n_train": n, "lr": lr,"wandb_group":group1,})

for i, rank in enumerate(rank_list):
    configs.append({"name": f"lora_rank_{rank}_lr_{lr_def:.0e}", "peft": "lora", "batch_size": bs, "model": "l", "n_train": n, "lr": lr_def,"wandb_group":group2,})
    configs.append({"name": f"dora_rank_{rank}_lr_{lr_def:.0e}", "peft": "dora", "batch_size": bs, "model": "l", "n_train": n, "lr": lr_def,"wandb_group":group2,})
    configs.append({"name": f"pissa_rank_{rank}_lr_{lr_def:.0e}", "peft": "pissa", "batch_size": bs, "model": "l", "n_train": n, "lr": lr_def,"wandb_group":group2,})


TEMPLATE = "run.sh"

# Read template once
with open(TEMPLATE, "r") as f:
    template_text = f.read()

os.makedirs("generated_jobs", exist_ok=True)

for cfg in configs:
    job_script = template_text
    for param in param_list:
        replacement = str(cfg[param]) if param in cfg else str(default_cfg[param])
        job_script = job_script.replace(f"{param}_PLACEHOLDER", replacement)

    script_path = f"generated_jobs/{cfg['name']}.sh"

    with open(script_path, "w") as f:
        f.write(job_script)

    # Make runnable
    os.chmod(script_path, 0o755)

    # Submit
    print(f"Submitting job: {cfg['name']}")
    subprocess.run(["sbatch", script_path])


    #Test 2 (small model)
    # {"name": "fft_0", "batch_size": 1},
    # {"name": "fft_1", "batch_size": 8},
    # {"name": "fft_2", "batch_size": 16},
    # {"name": "fft_3", "batch_size": 8, "LR_sch": 1},
    # {"name": "fft_4", "batch_size": 8, "LR_sch": 1, "n_train":300},

    # {"name": "lora_0", "peft": "lora", "lr": 1e-4, "batch_size": 8},
    # {"name": "lora_1", "peft": "lora", "lr": 1e-4, "batch_size": 16},
    # {"name": "lora_2", "peft": "lora", "lr": 1e-4, "batch_size": 8, "LR_sch": 1},
    # {"name": "lora_3", "peft": "lora", "lr": 1e-4, "batch_size": 16, "LR_sch": 1},

    # {"name": "lora_4", "peft": "lora", "lr": 1e-5, "batch_size": 8},
    # {"name": "lora_5", "peft": "lora", "lr": 1e-5, "batch_size": 16},
    # {"name": "lora_6", "peft": "lora", "lr": 1e-5, "batch_size": 8, "LR_sch": 1},
    # {"name": "lora_7", "peft": "lora", "lr": 1e-5, "batch_size": 16, "LR_sch": 1},
    # {"name": "lora_8", "peft": "lora", "lr": 1e-4, "batch_size": 8, "LR_sch": 1, "n_train":300},



    # {"name": "fft_lr_0", "peft": "None", "batch_size": bs, "model": "l", "n_train": n, "lr": 1e-5,},
    # {"name": "fft_lr_1", "peft": "None", "batch_size": bs, "model": "l", "n_train": n, "lr": 5e-5,},
    # {"name": "fft_lr_2", "peft": "None", "batch_size": bs, "model": "l", "n_train": n, "lr": 1e-4,},
    # {"name": "fft_lr_3", "peft": "None", "batch_size": bs, "model": "l", "n_train": n, "lr": 5e-4,},
    # {"name": "fft_lr_4", "peft": "None", "batch_size": bs, "model": "l", "n_train": n, "lr": 1e-3,},
    # {"name": "fft_lr_5", "peft": "None", "batch_size": bs, "model": "l", "n_train": n, "lr": 1e-6,},
    # {"name": "fft_lr_6", "peft": "None", "batch_size": bs, "model": "l", "n_train": n, "lr": 5e-6,},

    # {"name": "lora_lr_0", "peft": "lora", "batch_size": bs, "model": "l", "n_train": n, "lr": 1e-5,},
    # {"name": "lora_lr_1", "peft": "lora", "batch_size": bs, "model": "l", "n_train": n, "lr": 5e-5,},
    # {"name": "lora_lr_2", "peft": "lora", "batch_size": bs, "model": "l", "n_train": n, "lr": 1e-4,},
    # {"name": "lora_lr_3", "peft": "lora", "batch_size": bs, "model": "l", "n_train": n, "lr": 5e-4,},
    # {"name": "lora_lr_4", "peft": "lora", "batch_size": bs, "model": "l", "n_train": n, "lr": 1e-3,},

    # {"name": "dora_lr_0", "peft": "lora", "batch_size": bs, "model": "l", "n_train": n, "lr": 1e-5,},
    # {"name": "dora_lr_1", "peft": "lora", "batch_size": bs, "model": "l", "n_train": n, "lr": 5e-5,},
    # {"name": "dora_lr_2", "peft": "lora", "batch_size": bs, "model": "l", "n_train": n, "lr": 1e-4,},
    # {"name": "dora_lr_3", "peft": "lora", "batch_size": bs, "model": "l", "n_train": n, "lr": 5e-4,},
    # {"name": "dora_lr_4", "peft": "lora", "batch_size": bs, "model": "l", "n_train": n, "lr": 1e-3,},


#Rank search:

    # {"name": "lora_rank_1"  , "peft": "lora", "batch_size": bs, "model": "l", "n_train": n, "lr": 5e-4, "lora_rank":1,   "wandb_group":group2, "target_layers": 0, "LR_sch": 1},
    # {"name": "lora_rank_2"  , "peft": "lora", "batch_size": bs, "model": "l", "n_train": n, "lr": 5e-4, "lora_rank":2,   "wandb_group":group2, "target_layers": 0, "LR_sch": 1},
    # {"name": "lora_rank_4"  , "peft": "lora", "batch_size": bs, "model": "l", "n_train": n, "lr": 5e-4, "lora_rank":4,   "wandb_group":group2, "target_layers": 0, "LR_sch": 1},
    # {"name": "lora_rank_8"  , "peft": "lora", "batch_size": bs, "model": "l", "n_train": n, "lr": 5e-4, "lora_rank":8,   "wandb_group":group2, "target_layers": 0, "LR_sch": 1},
    # {"name": "lora_rank_16" , "peft": "lora", "batch_size": bs, "model": "l", "n_train": n, "lr": 5e-4, "lora_rank":16,  "wandb_group":group2, "target_layers": 0, "LR_sch": 1},
    # {"name": "lora_rank_32" , "peft": "lora", "batch_size": bs, "model": "l", "n_train": n, "lr": 5e-4, "lora_rank":32,  "wandb_group":group2, "target_layers": 0, "LR_sch": 1},
    # {"name": "lora_rank_64" , "peft": "lora", "batch_size": bs, "model": "l", "n_train": n, "lr": 5e-4, "lora_rank":64,  "wandb_group":group2, "target_layers": 0, "LR_sch": 1},
    # {"name": "lora_rank_128", "peft": "lora", "batch_size": bs, "model": "l", "n_train": n, "lr": 5e-4, "lora_rank":128, "wandb_group":group2, "target_layers": 0, "LR_sch": 1},
    # {"name": "dora_rank_1"  , "peft": "dora", "batch_size": bs, "model": "l", "n_train": n, "lr": 5e-4, "lora_rank":1,   "wandb_group":group2, "target_layers": 0, "LR_sch": 1},
    # {"name": "dora_rank_2"  , "peft": "dora", "batch_size": bs, "model": "l", "n_train": n, "lr": 5e-4, "lora_rank":2,   "wandb_group":group2, "target_layers": 0, "LR_sch": 1},
    # {"name": "dora_rank_4"  , "peft": "dora", "batch_size": bs, "model": "l", "n_train": n, "lr": 5e-4, "lora_rank":4,   "wandb_group":group2, "target_layers": 0, "LR_sch": 1},
    # {"name": "dora_rank_8"  , "peft": "dora", "batch_size": bs, "model": "l", "n_train": n, "lr": 5e-4, "lora_rank":8,   "wandb_group":group2, "target_layers": 0, "LR_sch": 1},
    # {"name": "dora_rank_16" , "peft": "dora", "batch_size": bs, "model": "l", "n_train": n, "lr": 5e-4, "lora_rank":16,  "wandb_group":group2, "target_layers": 0, "LR_sch": 1},
    # {"name": "dora_rank_32" , "peft": "dora", "batch_size": bs, "model": "l", "n_train": n, "lr": 5e-4, "lora_rank":32,  "wandb_group":group2, "target_layers": 0, "LR_sch": 1},
    # {"name": "dora_rank_64" , "peft": "dora", "batch_size": bs, "model": "l", "n_train": n, "lr": 5e-4, "lora_rank":64,  "wandb_group":group2, "target_layers": 0, "LR_sch": 1},
    # {"name": "dora_rank_128", "peft": "dora", "batch_size": bs, "model": "l", "n_train": n, "lr": 5e-4, "lora_rank":128, "wandb_group":group2, "target_layers": 0, "LR_sch": 1},



    #Notes
    """
    LoRa / DoRA / PISSA LR search for 512 dataset
    
    



    """













