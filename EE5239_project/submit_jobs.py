import os
import subprocess
from config import Config

# ----------------------------
# Define your parameter sweeps
# ----------------------------

default_cfg = vars(Config().parse())

param_list = ["name", "lr", "batch_size","LR_sch","n_train","n_val","peft","lora_rank", "model"]

n = 16
bs = 4

configs = [

    {"name": "fft_0", "peft": "None", "batch_size": 16, "model": "l", "n_train": 256, "lr": 5e-5,},
    {"name": "fft_1", "peft": "None", "batch_size": 16, "model": "l", "n_train": 256, "lr": 1e-4,},
    {"name": "fft_2", "peft": "None", "batch_size": 16, "model": "l", "n_train": 256, "lr": 5e-4,},

    {"name": "lora_3", "peft": "lora", "batch_size": 16, "model": "l", "n_train": 256, "lr": 5e-5,},
    {"name": "lora_4", "peft": "lora", "batch_size": 16, "model": "l", "n_train": 256, "lr": 1e-4,},
    {"name": "lora_5", "peft": "lora", "batch_size": 16, "model": "l", "n_train": 256, "lr": 5e-4,},


]

#Saved_results3
#0-3
#n = 64
#bs = 8
#4-7
# n = 32
# bs = 4
#8-11
#n = 256
#bs = 16

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
