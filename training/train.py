import copy
from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader


class CDTrainLoop():
    def __init__(self, teacher_model: nn.Module, target_model: nn.Module, dataloader: DataLoader, experiment_args: Mapping[str, Any]): #type: ignore
        self.current_step = 0 
        self.teacher_model = teacher_model
        self.target_model = target_model
        self.dataloader = dataloader #type: ignore
        self.experiment_args = experiment_args


        self.teacher_model.requires_grad_(False)
        self.teacher_model.eval()
        
        self.target_model.requires_grad_(False)
        self.target_model.train()



    def train(self):

        while self.current_step < self.experiment_args['total_training_steps']:
            batch, cond = next(self.dataloader)
            self.run_step(batch, cond)
    
    def run_step(self, ):
        pass

def initialize_inputs(dataloader: DataLoader, model: nn.Module , teacher_path: str): #type: ignore
    model.train()
    teacher_model = copy.deepcopy(model)
    teacher_model.load_state_dict(torch.load(teacher_path)) #type: ignore
    teacher_model.eval()

    target_model = copy.deepcopy(model)
    target_model.train()

    CDTrainLoop(
        teacher_model,
        target_model,
        dataloader
    ).train()