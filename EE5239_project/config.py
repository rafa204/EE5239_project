import argparse
import sys

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.conf = None
        
        #Setup parameters
        self.parser.add_argument('--name', type=str, default='test', help='results file directory')
        self.parser.add_argument('--cuda', type=str, default='0', help='CUDA device to use')
        self.parser.add_argument('--write_data', type=int, default=0, help='Rewrite dataset')
        self.parser.add_argument('--peft', type=str, default='None', help='which type of peft')
        self.parser.add_argument('--lora_rank', type=int, default=16, help='Rank of LoRA matrices')
        self.parser.add_argument('--model', type=str, default="s", help='Small or large model')

        # Hyperparameters for leaning
        self.parser.add_argument('--n_data', type=int, default=369, help='number of data points')
        self.parser.add_argument('--sl_per_vol', type=int, default=3, help='number of slices per volume')
        self.parser.add_argument('--n_train', type=int, default=100, help='training data ratio')
        self.parser.add_argument('--n_val', type=int, default=100, help='validation data ratio')
        self.parser.add_argument('--n_test', type=int, default=0, help='testing data ratio')
        self.parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train')
        self.parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
        self.parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        self.parser.add_argument('--LR_sch', type=int, default=0, help='Is there LR schedule or not')
        self.parser.add_argument('--val_freq', type=int, default=5, help='Validation freq')

        #other
        self.parser.add_argument('--tqdm', type=int, default=1, help='Loading bar or not (bar is bad for slurm)')
    
    

    def parse(self, args=None):
        """Parse the configuration"""
        self.conf = self.parser.parse_args(args=args)

        return self.conf
    
def save_configs(cfg, path):
        """
        Save all configuration parameters to a text file.
        Each line will be: <parameter>: <value>
        """
        with open(path, "w") as f:
            for key, value in vars(cfg).items():
                f.write(f"{key}: {value}\n")
    
