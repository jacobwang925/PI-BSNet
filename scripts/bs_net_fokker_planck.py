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


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
plt.set_loglevel (level = 'warning')

# Gaussian PDF function for the initial condition


class bsnet_train(base_run):
    def __init__(self, cfg=None):
        #super(bsnet_train, base_run).__init__()
        self.__setattr__("is_configured", False)
    
    def pde_residual(self, **kwargs):
        res1 = kwargs["B_s_t"] -0.001* kwargs["B_s_xx"]
        #dx = torch.sum(kwargs["B_s_x"][:,:,:,:,0] - kwargs["B_s_x"][:,:,:,:,-1], axis = (2,3))
        #dy = torch.sum(kwargs["B_s_y"][:,:,:,:,0] - kwargs["B_s_y"][:,:,:,:,-1], axis = (2,3))
        #dz = torch.sum(kwargs["B_s_z"][:,:,:,:,0] - kwargs["B_s_z"][:,:,:,:,-1], axis = (2,3))
        #dt = torch.sum(kwargs["B_s_t"], axis=(2,3,4))
        #res2 =  dt - dx -dy - dz
        #        res3 =  kwargs["B_s_t"] - kwargs["B_s_x"] - kwargs["B_s_y"] - kwargs["B_s_z"]

        return res1#, res2
    def gaussian_pdf(self,center, width):
        x = torch.linspace(self.min_X, self.max_X, self.n_ctrl_pts_state)
        grid_x, grid_y, grid_z = torch.meshgrid(x,x,x)
        cx, cy, cz = center
        ctrl_dist = torch.exp(-((grid_x - cx) ** 2 + (grid_y - cy) ** 2 + (grid_z - cz) ** 2) / (2 * width ** 2))
        proj_U_Full = torch.einsum('ijkl,ti,xj,yk,zl->txyz', ctrl_dist.reshape(1, *ctrl_dist.shape), self.Bit_t, self.Bit_x, self.Bit_y, self.Bit_z)[0,:,:,:]
        f = 1/torch.sum(proj_U_Full)
        return f*ctrl_dist
    
    def build_scenario(self, lmbda=None):
        raise NotImplementedError("currently not implemented")
        #ret_gt = []
       # 
       # for li in lmbda:
        #    logger.info(f"   ...building scenario for lambda = {li}")
        #    ret_gt.append(ground_truth(self.x, self.t, self.a, li))
        return ret_gt

    def make_surface(self, inner_matrix):
        # Generate the B-spline surface with the predicted control points
        y = torch.einsum('qijkl,ti,xj,yk,zl->qtxyz', inner_matrix, self.Bit_t, self.Bit_x, self.Bit_y, self.Bit_z)
        return y
    
    def loss(self,y_hat, x):
        U_full = torch.zeros((x.shape[0],
                              self.n_ctrl_pts_time,
                              self.n_ctrl_pts_state, 
                              self.n_ctrl_pts_state, 
                              self.n_ctrl_pts_state))
        U_full[:,0, :, :, :] = x  # Set initial condition at t=0
        U_full[:,1:,:,:, :] = y_hat  # Predicted control points 
        
        # Compute B-spline derivatives (t, x, and x^2 derivatives)
        B_surface, B_s_t, B_s_x, B_s_y, B_s_z, B_s_xx = compute_bspline_derivatives_3d(U_full, 
                                                            self.Bit_t, 
                                                            self.Bit_x, self.Bit_y, self.Bit_z,
                                                            self.Bit_t_derivative,
                                                            self.Bit_x_derivative, self.Bit_y_derivative, self.Bit_z_derivative,
                                                            self.Bit_x_second_derivative, self.Bit_y_second_derivative, self.Bit_z_second_derivative)
        res1 = self.pde_residual(B_s_t=B_s_t, 
                                B_s_x=B_s_x,
                                B_s_y=B_s_y,
                                B_s_z=B_s_z, 
                                B_s_xx= B_s_xx)
        
        physics_loss = torch.mean(torch.sum(torch.pow(res1,2),axis = (2,3,4)))
        #physics_loss+= torch.mean(torch.sum(torch.pow(res2,2),axis = (1)))
        return  physics_loss
            
        
    def train(self, config, **kwargs):
        logger.info("--- Train ---")
        #setup from config file
        self.setup(config, **kwargs)    
        
        self.model = self.model.train()
        
        if not self.is_configured:
            logger.warning("Is not configured.")
            return False
        
        #how often to newline print_outs
        newline_rate = 100
        
        #reshape input and make into torch tensor.
        lmd = torch.tensor(self.lmbda_train, dtype=torch.float32)
        
        # Define the initial condition with the size of control points grid
        rows = torch.unbind(lmd, dim=0)
        initial_condition = list(map(lambda row: self.gaussian_pdf(row[:3], row[3:]), rows))
        initial_condition = torch.stack(initial_condition, dim =0)

        
        logger.info(" Training loop commences")
        min_loss = 1e12
        isSaved = False
       
        def closure():
            self.optimizer.zero_grad()
            
            u_hat = self.model(lmd) 
            tloss = self.loss(u_hat, initial_condition)
            tloss.backward()  
            return tloss
                    
        for epoch in range(self.model_params.get('max_epochs')):
        
            if not self.optim_needs_closure:
                self.optimizer.zero_grad()
                
            u_hat = self.model(lmd)
            tloss = self.loss(u_hat, initial_condition)
                
            if self.optim_needs_closure:
                self.optimizer.step(closure) 
            else:
                tloss.backward()
                self.optimizer.step()
                
            if min_loss*0.9> (tloss.item()-1e-8):
                if epoch> self.model_params.get("max_epochs")//10 or not isSaved:
                    self.save_checkpoint(tloss)
                    isSaved =True
                    min_loss = tloss.item()
                
            if (epoch-1)%newline_rate==0:
                print()
            print(f" {epoch+1:05d} total_loss = {tloss.item():2.5f},"+
                  f" min_loss {min_loss:2.7f}", end="\r")
        print()
        
    def visualize(self, config, vis_lmbda, **kwargs):
        logger.info("--- Visualize  ---")
       
        #setup from config file
        self.setup(config, **kwargs)    
        if not self.is_configured:
            logger.warning("Is not configured.")
            return
       
            
        self.model = self.model.eval()
        #self.load_checkpoint(self.checkpoint)
        
        with torch.no_grad():
            
            # After training, test the network with a different lambda value
            y_hat = self.model(torch.tensor([[vis_lmbda]], dtype=torch.float32))
            U_full = self.impose_icbc(y_hat)
            
            logger.debug(f"y_hat shape {y_hat.shape}")
            logger.debug(f"u_full shape {U_full.shape}")
            y_hat_surface = self.make_surface(U_full)

            # Generate ground truth data for the testing lambda
            y_true_surface = ground_truth(self.x, self.t, self.a, vis_lmbda)

            # Compute the prediction error
            error_surface = np.abs(y_hat_surface - y_true_surface)

            # Plot the predicted B-spline surface, ground truth, and error
            fig = plt.figure(figsize=(18, 6))

            # Plot the ground truth surface
            ax0 = fig.add_subplot(131, projection='3d')
            X, Y = np.meshgrid(self.x, self.t)
            ax0.plot_surface(X, Y, y_true_surface, cmap='viridis', edgecolor='none')
            ax0.set_title('Ground Truth Surface')
            ax0.set_xlabel('x')
            ax0.set_ylabel('T')
            ax0.set_zlabel('F(x,t)')

            # Plot the predicted B-spline surface
            ax1 = fig.add_subplot(132, projection='3d')
            ax1.plot_surface(X, Y, y_hat_surface, cmap='viridis', edgecolor='none')
            ax1.set_title(f'B-spline Surface Prediction with Lambda = {vis_lmbda}')
            ax1.set_xlabel('x')
            ax1.set_ylabel('T')
            ax1.set_zlabel('F(x,t)')

            # Plot the prediction error
            ax2 = fig.add_subplot(133, projection='3d')
            ax2.plot_surface(X, Y, error_surface, cmap='hot', edgecolor='none')
            ax2.set_title('Prediction Error Surface')
            ax2.set_xlabel('x')
            ax2.set_ylabel('T')
            ax2.set_zlabel('Error')

            plt.savefig("test.png")
            plt.show()
            
def test(self, config, **kwargs):
        
       
        #setup from config file
        self.setup(config, **kwargs)    
        if not self.is_configured:
            logger.warning("Is not configured.")
            return
        
        if self.__getattribute__("checkpoint") is None:
            logger.error("no checkpoint for training")
            exit()
            
        model = model.eval()
        
        #build datasets and dataloaders
        
        y_tru_train = torch.stack(self.build_scenario(self.lmbda_train), dim=0)
        y_tru_test = torch.stack(self.build_scenario(self.lmbda_test), dim=0)
        
        train_ds = torch.utils.data.TensorData(self.lmbda_train, y_tru_train)
        test_ds = torch.utils.data.TensorData(self.lmbda_test, y_tru_test)
        
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size = 1, shuffle = False)
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size = 1, shuffle = False) 
        
        mse_y_train_mean = []
        mse_y_train_std = []
        for x, y in train_dl:
            u_hat = self.model(x)
            u_hat = self.impose_icbc(u_hat)
            y_hat = self.make_surface(u_hat)
            
            mse_y_train_mean.append( torch.mean(torch.pow(y-y_hat,2)).detach().numpy())
            mse_y_train_std.append(torch.mean(torch.pow(y-y_hat,2)).detach().numpy())
        
        
        
        mse_y_test_mean = []
        mse_y_test_std = []    
        for x, y in test_dl:
            u_hat = self.model(x)
            u_hat = self.impose_icbc(u_hat)
            y_hat = self.make_surface(u_hat)
            
            mse_y_test_mean.append( torch.mean(torch.pow(y-y_hat,2)).detach().numpy())
            mse_y_test_std.append(torch.mean(torch.pow(y-y_hat,2)).detach().numpy())
            
        fig = plt.figure(figsize=(8, 6))
        ax0 = fig.add_subplot(111)
        
        ax0.errorbar(self.lmbda_train, mse_y_train_mean, yerr=mse_y_train_std, mfc="black", ms=3, label = "Train")
        ax0.errorbar(self.lmbda_test, mse_y_test_mean, yerr=mse_y_test_std, mfc = "red", ms=3, label = "Test")
        ax0.legend()
        ax0.set_xlabel("$\lambda$")
        ax0.set_ylabel("mean square error")
        
        

        
            
            
             
if __name__ =="__main__":
    fire.Fire(bsnet_train)
    logger.info("Done")