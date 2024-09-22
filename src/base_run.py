
import os
import json
from jsonschema import validate, ValidationError
import fire
import torch
import torch.optim as optim
import numpy as np
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from utils import ground_truth, BsKnots, BsKnots_derivatives, compute_bspline_derivatives
from modules import ControlPointNet

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define the JSON Schema
schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "T": {"type": "integer"},
        "L": {"type": "integer"},
        "min_T": {"type": "number"},
        "max_T": {"type": "number"},
        "min_X": {"type": "number"},
        "max_X": {"type": "number"},
        "bspline_order": {"type": "integer"},
        "n_points": {"type": "integer"},
        "a": {"type": "number"},
        "lmbda_train": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"type": "number"}
                },
                {
                    "type": "string"
                }
            ]
        },
        "n_ctrl_pts_time": {"type": "integer"},
        "n_ctrl_pts_state": {"type": "integer"},
        "model_params": {
                "type": "object",
                "properties": {
                    "hidden_dim": {"type": "integer"},
                    "hidden_depth":{"type":"integer"},
                    "learning_rate": {"type": "number"},
                    "max_epochs": {"type": "integer"}
                },
                "required": ["hidden_dim", "learning_rate", "max_epochs"]
            }
    },
    
    "required": ["T", "L", "min_T", "max_T", "min_X", "max_X"]
}


class base_run(ABC):
    def __init__(self):
        super(base_run,).__init__()
    
    @abstractmethod
    def pde_residual(**kwargs):
        pass
        
    def setup(self, json_file):
        if json_file is None:
            self.__setattr__("is_configured", False) 
        elif Path(json_file).is_file():            
            with open(json_file,"r") as file:
                cfg = json.load(file)
                
            try:
                validate(instance=cfg, schema=schema)
                print("JSON data is valid.")
            except ValidationError as e:
                print(f"JSON data is invalid: {e.message}")
                
            
            for name, value in cfg.items():
                self.__setattr__(name, value)
            self.__setattr__("is_configured", True)
         
        self.save_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Generate B-spline basis matrices
        self.t = torch.linspace(self.min_T, self.max_T, self.n_points)
        self.x = torch.linspace(self.min_X, self.max_X, self.n_points)
        self.tk_t, self.Ln_t, self.Bit_t = BsKnots(self.n_ctrl_pts_time, 
                                                   self.bspline_order, 
                                                   self.n_points)
        self.tk_x, self.Ln_x, self.Bit_x = BsKnots(self.n_ctrl_pts_state, 
                                                   self.bspline_order, 
                                                   self.n_points)


        # Derivatives of the B-spline basis matrices
        self.Bit_t_derivative, self.Bit_t_second_derivative = BsKnots_derivatives(self.n_ctrl_pts_time,
                                                                        self.bspline_order, 
                                                                        len(self.t), 
                                                                        self.Ln_t, 
                                                                        self.tk_t)
        self.Bit_x_derivative, self.Bit_x_second_derivative = BsKnots_derivatives(self.n_ctrl_pts_state,
                                                                        self.bspline_order, 
                                                                        len(self.x), 
                                                                        self.Ln_x, 
                                                                        self.tk_x)
        
        #create_model
        self.model = ControlPointNet(self.n_ctrl_pts_state,
                                     self.n_ctrl_pts_time, 
                                     self.model_params.get("hidden_dim"), 
                                     self.model_params.get("hidden_depth"))
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.model_params.get("learning_rate"))
        
        
    def print_attributes(self):
        logger.info("--- Print Attributes ---")
        max_len_key = max([len(i) for i in self.__dict__.keys()])
        for name, value in self.__dict__.items():
            l = max_len_key - len(name)
            if type(value)==np.ndarray:
                value =f"ndarray with shape {value.shape} and type {value.dtype}"
            if type(value)==torch.Tensor:
                value =f"tensor with shape {value.shape} and type {value.dtype}"
                
            logger.info(f"{' '*l}{name}:  {value}")   
            
        
    def impose_icbc(self, y):
       
        U = torch.ones((self.n_ctrl_pts_time, self.n_ctrl_pts_state)) 
        for v in self.icbc:
            dim = int(v[0])
            idx = int(v[1])
            val = v[2]
            # Create a slice object for each dimension
            slices = [slice(None)] * U.dim()
            slices[dim] = idx
            U[tuple(slices)] = val 
        
        U[1:, :-1] = y 
        
        return U
    
    @abstractmethod
    def loss(self,y_hat, y_true , lmbda, physics_loss = True, data_loss=True, physics_loss_weight=1.0, data_loss_weight=1.0):
        pass

    def save_checkpoint(self,  loss):
        
        
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        
        # Remove old checkpoint if it exists
        for filename in Path(self.save_path).iterdir():
            if filename.name.startswith(self.save_prefix):
                filename.unlink()
        
        # Save new checkpoint
        checkpoint_filename = f"{self.save_prefix}_loss_{loss:.6f}.pt"
        checkpoint_path = Path(self.save_path) / checkpoint_filename
        torch.save({"model_state_dict":self.model.state_dict(),
                    }, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint):
        checkpoint = torch.load(checkpoint, weights_only =True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        