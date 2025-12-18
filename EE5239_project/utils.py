import torch
import numpy as np
import matplotlib.pyplot as plt
from criterion import SegmentationLoss
from tqdm import tqdm
from config import Config
from peft import LoraConfig, get_peft_model
import io
import wandb
from PIL import Image
from galore_torch import GaLoreAdamW


def get_batched_mask(image, input_point, predictor):
    """
    Helper function take an image and prompt batch as input and 
    use SAM2 to produce corresponding masks.
    image (torch.Tensor) [nbatches, 3, H, W]: Input image
    input_point (np array) [nbatches, 2]: Prompts
    """

    if image.ndim == 4: #Batched mode
        image = [im.cpu().numpy() for im in image]
        input_label = np.ones((len(image),1))
    else: #Single image mode
        image = [image]
        input_point = input_point[np.newaxis, :]
        input_label = np.ones((1,1))

    predictor.set_image_batch(image) # apply SAM image encoder to the image

    # prompt encoding
    mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
    sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels),boxes=None,masks=None,)
    high_res_feats = predictor._features["high_res_feats"]

    low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
        image_embeddings=predictor._features["image_embed"],
        image_pe = predictor.model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings, 
        multimask_output=False, # args.multimask_output if you want multiple masks
        repeat_image=False,  # the image is already batched
        high_res_features = high_res_feats
    )

    prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])# Upscale the masks to the original image resolution
    #prd_masks = torch.sigmoid(prd_masks)# Turn logit map to probability map
    return prd_masks.squeeze()

layers_list_short = ['q_proj', 'k_proj', 'v_proj', 'qkv']
layers_list_long = [
    'qkv',
    'q_proj', 
    'v_proj', 
    'k_proj', 
    'out_proj', 
    'mlp.layers.0', 
    'mlp.layers.1', 
    'iou_prediction_head.layers.0',
    'iou_prediction_head.layers.1',
    'iou_prediction_head.layers.2', 
    'pred_obj_score_head.layers.0',
    'pred_obj_score_head.layers.1',
    'pred_obj_score_head.layers.2', 
    'obj_ptr_proj.layers.0',
    'obj_ptr_proj.layers.1',
    'obj_ptr_proj.layers.2']

def get_lora_model(model):

    """
    Helper function to apply LoRA or variations to a given model
    Input: 
        model (Torch module): torch model on which we will apply the lora layers
    Output:
        same model with LoRA layers
    """

    cfg = Config().parse()
    if cfg.target_layers:
        target_modules=layers_list_long
    else:
        target_modules=layers_list_short


    if(cfg.peft == 'lora'):
        print("Using LoRA")
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.1,
            use_rslora=True
        )
        model = get_peft_model(model, lora_config)

    elif(cfg.peft == "pissa"):
        print("Using PISSA")
        pissa_config = LoraConfig(
        init_lora_weights="pissa", # Configure the initialization method to "pissa", which may take several minutes to execute SVD on the pre-trained model.
        #init_lora_weights="pissa_niter_4", # Initialize the PiSSA with fast SVD, which completes in just a few seconds.
        r=cfg.lora_rank,
        lora_alpha=32,
        lora_dropout=0, # Since the component of the PiSSA adapter are the principal singular values and vectors, dropout should be set to 0 to avoid random discarding.
        target_modules=target_modules,
        )
        model = get_peft_model(model, pissa_config)


    elif(cfg.peft == 'dora'):
        print("Using DoRA")
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.1,
            use_rslora=True,
            use_dora=True
        )
        model = get_peft_model(model, lora_config)

    elif(cfg.peft == 'galore'):
        print("Using GaLore")
        for name, param in model.named_parameters():
            param.requires_grad = any(key in name for key in target_modules)

    
    return model

def get_galore_optimizer(model):
    """
    If using galore, this gets the Galore optimizer.
    The input model should have frozen all layers on which Galore should not be applied
    """
    cfg = Config().parse()
    galore_params = [p for p in model.parameters() if (p.requires_grad and p.ndim >= 2)]
    regular_params = [p for p in model.parameters() if (p.requires_grad and p.ndim == 1)]
    # then call galore_adamw
    param_groups = [{'params': regular_params}, {'params': galore_params, 'rank': cfg.lora_rank, 'update_proj_gap': 200, 'scale':0.25, 'proj_type': 'std'}]
    
    optimizer = GaLoreAdamW(param_groups, lr=cfg.lr, weight_decay=4e-5)
    return optimizer


def test_model(predictor, test_loader):
    """
    Test SAM2 on a testing set
    """
    predictor.model.eval()
    avg_loss = 0
    loss_fun = SegmentationLoss()
    
    for image, mask, input_point in test_loader:

        prd_mask = get_batched_mask(image, input_point, predictor)

        avg_loss += loss_fun(prd_mask.squeeze(), mask.squeeze()).item()
            
    avg_loss = avg_loss / len(test_loader)
    print(f"--- Validation loss: {avg_loss:3f} ---")
    
    return avg_loss

def plot_examples(predictor, test_dataset, slices, save_path, epoch):
    """Plot example images from the BRATS 2020 dataset"""
    predictor.model.eval()
    cfg = Config().parse()
    print("-"*3+"Plotting examples"+"-"*3)
    loss_fun = SegmentationLoss()
    fs = 16
    j = 0
    for i in slices:
        if(i>=len(test_dataset)):
            continue
        image, mask, input_point = test_dataset[i]

        prd_mask = get_batched_mask(image, input_point, predictor)

        prd_mask = torch.sigmoid(prd_mask)
        prd_mask = torch.round(prd_mask).squeeze()

        loss = loss_fun.dice_loss(prd_mask, mask)

        fig, ax = plt.subplots(1,3,figsize = (10,4))
        plt.gray()
        ax[0].imshow(image)
        ax[0].set_title("Input image", fontsize = fs)
        ax[1].imshow(mask.cpu().detach())
        ax[1].set_title("True mask", fontsize = fs)
        ax[2].imshow(prd_mask.cpu().detach())
        ax[2].set_title(f"Predicted mask (epoch {epoch}) \n Dice loss = {loss:.2f}", fontsize = fs)

        for ax in fig.get_axes():
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis='x', length=0)
            ax.tick_params(axis='y', length=0)

        fig.tight_layout()
        # fig.savefig(save_path / f"out_{j}_{epoch}.png")
        # plt.close("all")
        j += 1

        if cfg.wandb:
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            wandb.log(({"epoch": epoch, f"example_{j}": wandb.Image(Image.open(buf))}))
        
        plt.close("all")
        


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
    return trainable_params, all_param, 100 * trainable_params / all_param

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


def count_optimizer_params(optimizer):
    """
    Calculates the total number of parameters managed by a PyTorch optimizer.
    """
    total_params = sum(p.numel() for group in optimizer.param_groups for p in group['params'] if p.grad is not None)
    return total_params