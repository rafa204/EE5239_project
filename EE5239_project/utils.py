import torch
import numpy as np
import matplotlib.pyplot as plt
from criterion import SoftDiceLoss
from tqdm import tqdm
from pathlib import Path

def get_mask(image, input_point, predictor):

    if torch.is_tensor(image):
        image = image.squeeze().cpu().numpy()
        input_point = input_point.squeeze(dim=0)

    input_label = np.array([1])

    predictor.set_image(image) # apply SAM image encoder to the image

    # prompt encoding
    mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
    sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels),boxes=None,masks=None,)

    # mask decoder
    batched_mode = unnorm_coords.shape[0] > 1 # multi object prediction
    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
    low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse_embeddings,dense_prompt_embeddings=dense_embeddings,multimask_output=True,repeat_image=batched_mode,high_res_features=high_res_features,)
    prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])# Upscale the masks to the original image resolution

    idx = torch.argmax(prd_scores)
    prd_mask = torch.sigmoid(prd_masks[:, idx])# Turn logit map to probability map

    return prd_mask

def get_batched_mask(image, input_point, predictor):

    if torch.is_tensor(image):
        image = [im.squeeze().cpu().numpy() for im in image]
        #input_point = [p.cpu().numpy() for p in input_point]
        input_label = np.ones((len(image),1))
    else:
        image = [image]
        input_point = input_point[np.newaxis, :]
        input_label = np.ones((1,1))

    predictor.set_image_batch(image) # apply SAM image encoder to the image

    # prompt encoding
    mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
    sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels),boxes=None,masks=None,)

    # mask decoder
    batched_mode = unnorm_coords.shape[0] > 1 # multi object prediction
    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
    low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse_embeddings,dense_prompt_embeddings=dense_embeddings,multimask_output=True,repeat_image=batched_mode,high_res_features=high_res_features,)
    prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])# Upscale the masks to the original image resolution

    prd_masks = torch.sigmoid(prd_masks)# Turn logit map to probability map

    return prd_masks.squeeze()


def test_model(predictor, test_loader):
    predictor.model.eval()
    avg_loss = 0
    loss_fun = SoftDiceLoss()
    
    for image, mask, input_point in test_loader:

        prd_mask = get_mask(image, input_point, predictor)

        prd_mask = torch.round(prd_mask)

        avg_loss += loss_fun(prd_mask, mask)
            
    avg_loss = avg_loss.item() / len(test_loader)
    print(f"--- Validation DICE loss: {avg_loss:3f} ---")
    
    return avg_loss

def plot_examples(predictor, test_dataset, slices, save_path):
    predictor.model.eval()
    print("-"*3+"Plotting examples"+"-"*3)
    loss_fun = SoftDiceLoss()
    fs = 16
    j = 0
    for i in slices:
        if(i>=len(test_dataset)):
            continue
        image, mask, input_point = test_dataset[i]

        prd_mask = get_mask(image, input_point, predictor)
        prd_mask = torch.round(prd_mask).squeeze()

        loss = loss_fun(prd_mask, mask)

        fig, ax = plt.subplots(1,3,figsize = (10,4))
        plt.gray()
        ax[0].imshow(image)
        ax[0].set_title("Input image", fontsize = fs)
        ax[1].imshow(mask.cpu().detach())
        ax[1].set_title("True mask", fontsize = fs)
        ax[2].imshow(prd_mask.cpu().detach())
        ax[2].set_title(f"Pred mask | Dice = {1-loss:.2f}", fontsize = fs)

        for ax in fig.get_axes():
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis='x', length=0)
            ax.tick_params(axis='y', length=0)

        fig.tight_layout()
        fig.savefig(save_path / f"out_{j}.png")
        plt.close("all")
        j += 1
        


def plot_loss_curves(val_losses, trn_losses, results_path):

    val_epochs = [l[1] for l in val_losses]
    val_losses = [l[0] for l in val_losses]
    trn_losses = [l[0] for l in trn_losses]
    
    best_val_dice = 1-min(val_losses)

    fig,  ax = plt.subplots(1,1,figsize = (3.5,3.5))

    ax.plot(trn_losses, label='Training')
    ax.plot(val_epochs, val_losses, label='Validation')
    ax.set_xlabel("Epochs", fontsize = 13)
    ax.set_ylabel("Dice loss", fontsize = 13)
    ax.grid()
    ax.legend()
    ax.set_title(f"Best dice coeff = {best_val_dice:3f}")
        

    fig.tight_layout()
    fig.savefig(results_path/"loss_curves.png")
    plt.close("all")


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def write_dataset(dataset, path):
    print("Writing dataset")

    for i in tqdm(range(len(dataset))):

        image, mask = dataset[i]
        folder = path /f"data_{i}/"
        folder.mkdir(exist_ok=True)
        
        np.save(folder / "imgs.npy", image)
        np.save(folder / "mask.npy", mask)


def optimizer_state_size_mb(optimizer):
    total = 0
    for state in optimizer.state.values():
        for v in state.values():
            if torch.is_tensor(v):
                total += v.numel() * v.element_size()
    return total / 1024**2
