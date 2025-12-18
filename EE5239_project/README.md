# EE5239 project
### Rafael Avelar, Leonidas Tsigkounakis

Our project involved fine tuning the SAM2 model by META on the BRATS 2020 dataset. This folder contains only our custom code used for the project. To run our code, you must also get the tools from the SAM2 model repository
- Clone the entire project from [here.](https://github.com/rafa204/EE5239_project) This contains both our code and the SAM2 tools, some of which
have been edited for our work. The project folder is called EE5239_project, 
but there is an inner folder called EE5239_project (this folder).
- Install EE5239_project/requirements.txt
- Download the dataset. To run this on MSI (Minnesota Supercomputing Institute), find the folder: 
/scratch.global/training_data copy it to the inner EE5239_project folder.
This data has already been prepared.
- Download the SAM2 pre-trained weights. You can do this by running the checkpoints/download_ckpts.sh file.

# Runnning a fine tuning session:
To fine tune SAM2, run the train.py file. You can see the results by connecting
to you weights and biases account, as the results are logged there.
Change the wandb.init configurations to what is needed. The most important command line parameters are as follows:

--name: name of the folder to which results are saved, and/or name of wandb run <br>
--peft: "None", "lora", "dora", "pissa" or "galore". These are the peft algorithms tested <br>
--lr: learning rate  <br>
--batch_size: batch size  <br>
--LR_sch: 0 or 1, whether to use Cosine annealing lr scheduler   <br>
--lora_rank: rank parameter for low-rank fine tuning algorithms. <br>
--wandb_group: name of weights and biases group to save results to <br>
