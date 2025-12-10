import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from data_preparation import BRATS_dataset, BRATS_dataset_2D
from torch.utils.data import DataLoader, Subset, random_split
from utils import *
from criterion import SoftDiceLoss
from config import Config, save_configs
from peft import LoraConfig, get_peft_model
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts as cosAnnealer
from torch.optim.lr_scheduler import CosineAnnealingLR
import time

cfg = Config().parse()

results_path = Path('results/saved_results3')/cfg.name
results_path.mkdir(parents=True, exist_ok=True)

save_configs(cfg, results_path/"params.txt")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_path = "/users/0/avela019/Desktop/EE5561_project/BRATS20/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/"
sam2_path = "/users/0/avela019/EE5239_project/sam2/"
data_path = "/users/0/avela019/EE5239_project/sam2/EE5239_project/data_long/"

#========= Setup dataset ===========

if cfg.write_data:
    dataset_3D = BRATS_dataset(dataset_path, device, slices_per_volume=cfg.sl_per_vol, num_slices = cfg.n_data)
    write_dataset(dataset_3D, data_path)

full_dataset = BRATS_dataset_2D(data_path, device)

generator = torch.Generator().manual_seed(0)
train_dataset, val_dataset, _ = random_split(full_dataset, [cfg.n_train, cfg.n_val, len(full_dataset) - cfg.n_train - cfg.n_val], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

#========= Get SAM2 weights ===========

if cfg.model == "s":
    sam2_checkpoint = "../checkpoints/sam2.1_hiera_small.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
else: 
    sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)
predictor.model.sam_mask_decoder.train(True) # enable training of mask decoder 
predictor.model.sam_prompt_encoder.train(True) # enable training of prompt encoder

if(cfg.peft == 'lora'):
    print("Using LoRA")
    config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        use_rslora=True
    )
    lora_model = get_peft_model(predictor.model, config)
    predictor.model = lora_model.to(device)

elif(cfg.peft == "pissa"):
    print("Using PISSA")
    lora_config = LoraConfig(
    init_lora_weights="pissa", # Configure the initialization method to "pissa", which may take several minutes to execute SVD on the pre-trained model.
    #init_lora_weights="pissa_niter_4", # Initialize the PiSSA with fast SVD, which completes in just a few seconds.
    r=cfg.lora_rank,
    lora_alpha=32,
    lora_dropout=0, # Since the component of the PiSSA adapter are the principal singular values and vectors, dropout should be set to 0 to avoid random discarding.
    target_modules=["q_proj", "k_proj"],
    )
    lora_model = get_peft_model(predictor.model, lora_config)
    predictor.model = lora_model.to(device)

print_trainable_parameters(predictor.model)

#========= Training setup and loop ===========

optimizer=torch.optim.AdamW(params=predictor.model.parameters(),lr=cfg.lr,weight_decay=4e-5)
scaler = torch.amp.GradScaler(device="cuda")
loss_fun = SoftDiceLoss()

if cfg.LR_sch:
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.n_epochs, eta_min=1e-6) # Minimum learning rate

trn_losses = []
val_losses = []

rng = np.random.default_rng(4)
plot_indices = rng.choice(np.arange(0, len(val_dataset)), size=10, replace=False)
best_epoch = True
best_val_loss = np.inf

torch.cuda.reset_peak_memory_stats()

print("*"*20 + "  Starting training  " + "*"*20)
for epoch in range(cfg.n_epochs):
    
    avg_loss = 0
    with torch.amp.autocast('cuda'): # cast to mix precision

        if(epoch % cfg.val_freq == 0 or epoch == cfg.n_epochs - 1):
            val_loss = test_model(predictor, val_loader)
            val_losses.append([val_loss,epoch])
            np.save(results_path / "val_loss.npy", np.array(val_losses))
            plot_loss_curves(val_losses, trn_losses, results_path)
            best_epoch = val_loss < best_val_loss
            if best_epoch:
                best_val_loss = val_loss
                plot_examples(predictor, val_dataset, plot_indices, results_path)     

        predictor.model.train()
        if(cfg.tqdm):
            train_bar = tqdm(train_loader, desc="[Training]")
        else: train_bar = train_loader

        itr=0
        
        for image, mask, input_point in train_bar:
            if itr%cfg.batch_size == 0:
                batch_loss_list = torch.zeros(cfg.batch_size)

            prd_mask = get_mask(image, input_point, predictor)
            batch_loss_list[itr%cfg.batch_size] = loss_fun(prd_mask, mask)

            if itr%cfg.batch_size == cfg.batch_size-1:

                loss = batch_loss_list.mean()
                predictor.model.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update() # Mix precision

                avg_loss+=loss.item()/len(train_loader)*cfg.batch_size

            itr+=1


    # print("Optimizer state MB:", optimizer_state_size_mb(optimizer))

    trn_losses.append([avg_loss,epoch])
    np.save(results_path / "train_loss.npy", np.array(trn_losses))
    print(f"---Epoch {epoch}: train loss = {avg_loss:.3f}" + "---")
    if cfg.LR_sch:
        scheduler.step()

