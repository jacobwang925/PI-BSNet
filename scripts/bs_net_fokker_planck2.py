import fire
import torch
import torch.optim as optim
import numpy as np
import logging
from pathlib import Path
from matplotlib import pyplot as plt
from utils import ground_truth, compute_bspline_derivatives_3d
from modules import ControlPointNet3D
from base_run import base_run
import opt_einsum as oe


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
plt.set_loglevel (level = 'warning')

class bsnet_train(base_run):
    def __init__(self, cfg=None):
        #super(bsnet_train, base_run).__init__()
        self.__setattr__("is_configured", False)
        self.D = 0.1
        self.save=False
        self.ic_ctrl_only = 0

    
    def pde_residual(self, **kwargs):
        res1 = kwargs["B_s_t"] -self.D* kwargs["B_s_xx"]  
        return res1
    
    def make_surface(self, inner_matrix):
        # Generate the B-spline surface with the predicted control points
        if inner_matrix.ndim ==5:
            y = torch.einsum('qijkl,ti,xj,yk,zl->qtxyz', inner_matrix, self.Bit_t, self.Bit_x, self.Bit_y, self.Bit_z)
        if inner_matrix.ndim ==4:
            y = torch.einsum('qjkl,xj,yk,zl->qxyz', inner_matrix, self.Bit_x, self.Bit_y, self.Bit_z)
        return y
    
    def computebsderivatives(self,U_full):
        
        path_info = oe.contract_path('qijkl,ti,xj,yk,zl->qtxyz', U_full, self.Bit_t, self.Bit_x, self.Bit_y, self.Bit_z)[0]
        B_surface = oe.contract('qijkl,ti,xj,yk,zl->qtxyz', U_full, self.Bit_t, self.Bit_x, self.Bit_y, self.Bit_z, optimize=path_info)
        B_surface_t = oe.contract('qijkl,ti,xj,yk,zl->qtxyz', U_full, self.Bit_t_derivative, self.Bit_x, self.Bit_y, self.Bit_z, optimize=path_info)
        B_surface_xx = oe.contract('qijkl,ti,xj,yk,zl->qtxyz', U_full, self.Bit_t, self.Bit_x_second_derivative, self.Bit_y, self.Bit_z, optimize=path_info)
        B_surface_yy = oe.contract('qijkl,ti,xj,yk,zl->qtxyz', U_full, self.Bit_t, self.Bit_x, self.Bit_y_second_derivative, self.Bit_z, optimize=path_info)
        B_surface_zz = oe.contract('qijkl,ti,xj,yk,zl->qtxyz', U_full, self.Bit_t, self.Bit_x, self.Bit_y, self.Bit_z_second_derivative, optimize=path_info)
        # Laplacian (sum of second derivatives in each spatial direction)
        B_surface_laplacian = B_surface_xx + B_surface_yy + B_surface_zz
        return B_surface, B_surface_t, B_surface_laplacian
        
    def loss(self,y_hat, y_ic, ic, lmd):
        y_hat[:,:,:,:,[0,-1]]=1
        y_ic[:,:,:,[0,-1]] =1
        ic[:,:,:,[0,-1]]=1
        
        loss_on_ic_ctrl = torch.mean(torch.sum((y_ic-ic)**2,axis=(1,2,3)))
        
        if loss_on_ic_ctrl>100 and self.ic_ctrl_only<1:
            self.save=False
            return loss_on_ic_ctrl
        
        #ic_loss (after spline)
        ic_surface = self.make_surface(ic)
        y_ic_surface = self.make_surface(y_ic)
        ic_loss = torch.mean(torch.sum(pow(y_ic_surface - ic_surface,2), axis=(1,2,3)))
        
        if ic_loss>50 and self.ic_ctrl_only<2:
            self.save=False
            self.ic_ctrl_only = 1
            return loss_on_ic_ctrl + ic_loss
        
        self.ic_ctrl_only=2
        U_full = y_hat #torch.cat([y_ic.unsqueeze(1),y_hat], dim=1)
        #pde_loss 
        B_surface, B_s_t,  B_s_xx = self.computebsderivatives(U_full) 
        pde_loss = torch.mean(torch.sum(torch.pow(self.pde_residual(B_s_t=B_s_t, B_s_xx=B_s_xx),2), axis=(1,2,3)))
        
        #bc_loss
        neumann_loss_xy = torch.mean(torch.sum(torch.pow(B_s_t[:,:,[0,-1],:,:],2), axis=(2,3,4)))
        neumann_loss_xy +=torch.mean(torch.sum(torch.pow(B_s_t[:,:,:,[0,-1],:],2), axis=(2,3,4)))
        
        dirichlet_loss_z = torch.mean(torch.sum(torch.pow(B_surface[:,:,:,:,[0,-1]]-1,2), axis=(2,3,4)))
        
        print(loss_on_ic_ctrl.detach().numpy(), 
              ic_loss.detach().numpy(), 
              pde_loss.detach().numpy(), 
              neumann_loss_xy.detach().numpy(), 
              dirichlet_loss_z.detach().numpy(), 
              torch.mean(B_surface**2).detach(), 
              torch.std(B_surface**2).detach(),
              torch.mean(B_s_t**2).detach())
        self.save=True
        return (loss_on_ic_ctrl
                + ic_loss
                + neumann_loss_xy
                + 10*pde_loss)

    def initial_conditions_from_lambda(self, C):
        linspace = torch.linspace(0, 1, self.n_ctrl_pts_state)
        grid = torch.meshgrid(linspace, linspace, linspace, indexing='ij')
        C = C.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        gradient = (C[:, 1] * grid[0]) + (C[:, 2] * grid[1]) + (C[:, 3] * grid[2])+0.5+C[:,0]
        return gradient

    def train(self, config, **kwargs):
        logger.info("--- Train ---")
        #setup from config file
        self.setup(config, **kwargs)    
        
        self.model = self.model.train()
        
        if not self.is_configured:
            logger.warning("Is not configured.")
            return False
    
        self.initial_conditions = self.initial_conditions_from_lambda(self.lambda_train)
        
           
        logger.info(" Training loop commences")
        min_loss = 1e12
        isSaved = False
       
        
        for epoch in range(self.model_params.get('max_epochs')):
            self.perm = torch.randperm(len(self.initial_conditions))[0:100]
            if not self.optim_needs_closure:
                self.optimizer.zero_grad()
                tloss = self.closure()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=50000)
                self.optimizer.step()
                
            if self.optim_needs_closure:
                self.optimizer.step(self.closure)
                tloss = self.closure() 
            
            
            print(f" {epoch+1:05d} total_loss = {tloss.item():2.5f},"+
                  f" min_loss {min_loss:2.7f}") 
            
            if min_loss*0.9> (tloss.item()-1e-8) and self.save:
                if epoch> self.model_params.get("max_epochs")//100 or not isSaved:
                    self.save_checkpoint(tloss)
                    isSaved =True
                    min_loss = tloss.item()
                
                
        
        
    def closure(self):
        self.optimizer.zero_grad()
        y_hat, y_ic = self.model(self.lambda_train[self.perm]) 
        tloss = self.loss(y_hat,y_ic, self.initial_conditions[self.perm], self.lambda_train)
        tloss.backward() 
        return tloss 
    

if __name__ =="__main__":
    fire.Fire(bsnet_train)
    logger.info("Done")