import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from data_preparation import BRATS_dataset, BRATS_dataset_2D
from torch.utils.data import DataLoader, Subset, random_split
from utils import *
from criterion import SegmentationLoss
from config import Config, save_configs
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

def main():

    cfg = Config().parse()

    if(cfg.wandb):
        wandb.init(
        project="SAM_fine_tuning",  # The project name
        entity="ee5239",            # The Team (or User) name
        group="LR_search_5",        # Group for organizing multiple runs (e.g., DDP)
        name=cfg.name,              # A readable name for this specific run
        job_type="train",           # Useful for separating 'train' vs 'eval' vs 'preprocess'
        config=vars(cfg),
        notes="Test with traineable image encoder"
        )

    results_path = Path('./results/saved_results_0')/cfg.name
    results_path.mkdir(parents=True, exist_ok=True)

    save_configs(cfg, results_path/"params.txt")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_path = Path("/users/0/avela019/Desktop/EE5561_project/BRATS20/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/")
    data_path = Path("./training_data/")

    #========= Setup dataset ===========

    if cfg.write_data:
        dataset_3D = BRATS_dataset(dataset_path, device, slices_per_volume=cfg.sl_per_vol, num_slices = cfg.n_data)
        write_dataset(dataset_3D, data_path)

    full_dataset = BRATS_dataset_2D(data_path, device)

    generator = torch.Generator().manual_seed(0)
    train_dataset, val_dataset, _ = random_split(full_dataset, [cfg.n_train, cfg.n_val, len(full_dataset) - cfg.n_train - cfg.n_val], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)

    #====== Get SAM2 weights ===========

    if cfg.model == "s":
        sam2_checkpoint = "../checkpoints/sam2.1_hiera_small.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    else: 
        sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint)

    #Setup LORA and variations
    sam2_model = get_lora_model(sam2_model).to(device)

    print_trainable_parameters(sam2_model)
    predictor = SAM2ImagePredictor(sam2_model)

    #========= Training setup and loop ===========

    optimizer=torch.optim.AdamW(params=predictor.model.parameters(),lr=cfg.lr,weight_decay=4e-5)
    #scaler = torch.amp.GradScaler(device="cuda")
    loss_fun = SegmentationLoss()

    if cfg.LR_sch:
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.n_epochs, eta_min=1e-6) # Minimum learning rate

    trn_losses = []
    val_losses = []

    rng = np.random.default_rng(5)
    plot_indices = rng.choice(np.arange(0, len(val_dataset)), size=10, replace=False)
    best_val_loss = np.inf

    torch.cuda.reset_peak_memory_stats()

    print("*"*20 + "  Starting training  " + "*"*20)
    for epoch in range(cfg.n_epochs):
        
        avg_loss = 0
        #with torch.amp.autocast('cuda'): # cast to mix precision
            
        #Validation before training so we can see the performance of the default model
        if cfg.val and (epoch % cfg.val_freq == 0 or epoch == cfg.n_epochs - 1):
            
            val_loss = test_model(predictor, val_loader)
            val_losses.append([val_loss,epoch])
            if(cfg.wandb):
                wandb.log({
                    "val/loss": val_loss,
                })
            np.save(results_path / "val_loss.npy", np.array(val_losses))
            if cfg.plot:
                plot_loss_curves(val_losses, trn_losses, results_path)
                plot_examples(predictor, val_dataset, plot_indices, results_path, epoch)     

        predictor.model.train()

        if(cfg.tqdm):
            train_bar = tqdm(train_loader, desc="[Training]")
        else: train_bar = train_loader

        for image, mask, input_point in train_bar:

            prd_mask = get_batched_mask(image, input_point, predictor)
            loss = loss_fun(prd_mask, mask)
            predictor.model.zero_grad()
            loss.backward()
            optimizer.step()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update() # Mix precision

            avg_loss+=loss.item()/len(train_loader)

        if(cfg.wandb):
            wandb.log({"train/loss": avg_loss})
        print(f"------ Epoch {epoch} ------")
        print("Optimizer state in MB:", optimizer_state_size_mb(optimizer))
        print("Optimizer params:", count_optimizer_params(optimizer))

        trn_losses.append([avg_loss,epoch])
        np.save(results_path / "train_loss.npy", np.array(trn_losses))
        print(f"train loss = {avg_loss:.3f}" + "---")
        if cfg.LR_sch:
            scheduler.step()


    if(cfg.wandb):
        wandb.finish()

if __name__ == "__main__":
    main()

