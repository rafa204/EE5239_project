import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from data_preparation import BRATS_dataset, BRATS_dataset_2D
from torch.utils.data import DataLoader, random_split
from utils import *
from criterion import SoftDiceLoss
from config import Config

cfg = Config().parse()

results_path = Path('saved_results')/cfg.out_path
results_path.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_path = "/users/0/avela019/Desktop/EE5561_project/BRATS20/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/"
sam2_path = "/users/0/avela019/EE5239_project/sam2/"
data_path = "/users/0/avela019/EE5239_project/sam2/EE5239_project/data/"

#========= Setup dataset ===========

dataset_3D = BRATS_dataset(dataset_path, device, slices_per_volume=cfg.sl_per_vol, num_slices = cfg.n_data)
write_dataset(dataset_3D, data_path)
dataset = BRATS_dataset_2D(data_path, device)

train_size = int(cfg.trn_rat*len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size]
)
train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

#========= Get SAM2 weights ===========

sam2_checkpoint = "../checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)
predictor.model.sam_mask_decoder.train(True) # enable training of mask decoder 
predictor.model.sam_prompt_encoder.train(True) # enable training of prompt encoder

#========= Training setup and loop ===========

optimizer=torch.optim.AdamW(params=predictor.model.parameters(),lr=cfg.learning_rate,weight_decay=4e-5)
scaler = torch.amp.GradScaler(device="cuda")

loss_fun = SoftDiceLoss()

trn_losses = []
val_losses = []

train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

print("Starting training")
for epoch in range(cfg.n_epochs):
    
    avg_loss = 0
    with torch.amp.autocast('cuda'): # cast to mix precision

        if(epoch % cfg.val_freq == 0 or epoch == cfg.n_epochs - 1):
            val_loss = test_model(predictor, val_loader)
            val_losses.append([val_loss,epoch])
            np.save(results_path / "val_loss.npy", np.array(val_losses))
            plot_examples(predictor, val_dataset, range(10), results_path)
            plot_loss_curves(val_losses, trn_losses, results_path)

        for image, mask, input_point in train_loader:
            input_label = np.array([1])

            prd_mask = get_mask(image, input_point, input_label, predictor)

            loss = loss_fun(prd_mask, mask)

            predictor.model.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update() # Mix precision

            avg_loss+=loss.item()/len(train_loader)

    trn_losses.append([avg_loss,epoch])
    np.save(results_path / "train_loss.npy", np.array(trn_losses))
    print("-"*20 + f"Epoch {epoch}: train loss = {avg_loss:.3f}" + "-"*20)
    