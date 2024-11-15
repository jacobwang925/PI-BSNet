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
        self.D = 0.0001
    
    def pde_residual(self, **kwargs):
        res1 = kwargs["B_s_t"] -self.D* kwargs["B_s_xx"]    
        return res1
    
    def gaussian_pdf(self,center, width):
        x = torch.linspace(self.min_X, self.max_X, self.n_ctrl_pts_state)
        grid_x, grid_y, grid_z = torch.meshgrid(x,x,x)
        cx, cy, cz = center[:, 0], center[:, 1], center[:, 2]
        
        # Reshape grid tensors to match the dimensions for broadcasting
        cx = cx.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        cy = cy.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        cz = cz.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        grid_x = grid_x.unsqueeze(0).repeat(center.size(0), 1, 1, 1)  # Shape: (Q, n, n, n)
        grid_y = grid_y.unsqueeze(0).repeat(center.size(0), 1, 1, 1)  # Shape: (Q, n, n, n)
        grid_z = grid_z.unsqueeze(0).repeat(center.size(0), 1, 1, 1)  # Shape: (Q, n, n, n)
        width = width.unsqueeze(1).unsqueeze(2)
        
        r2 =((grid_x - cx) ** 2 +(grid_y - cy) ** 2 + (grid_z - cz) ** 2)
        ctrl_dist = torch.exp(-r2 / (2 * width ** 2)) #control points
        proj_U_Full = torch.einsum('Nijk,xi,yj,zk->Nxyz', ctrl_dist, self.Bit_x, self.Bit_y, self.Bit_z)
        f = 1/torch.sum(proj_U_Full, dim=(1,2,3))
        return torch.einsum("ijkl,i->ijkl", ctrl_dist,f), f
    
    def estimate_true_surface_at_timestep(self, lmbda=None, time_step=1.0):
        width= lmbda[:,3:]
        Q  = torch.sqrt((width**2 + 2 * self.D * time_step))
        P0,f0 = self.gaussian_pdf(lmbda[:,:3],width)
        Pt,ft = self.gaussian_pdf(lmbda[:,:3],Q)
        Pt = torch.einsum("ijkl,i->ijkl", Pt,f0/ft)
        return torch.einsum("Nijk,xi,yj,zk->xyz", Pt, self.Bit_x, self.Bit_y, self.Bit_z)
    
    def make_surface(self, inner_matrix):
        # Generate the B-spline surface with the predicted control points
        if inner_matrix.ndim ==5:
            y = torch.einsum('qijkl,ti,xj,yk,zl->qtxyz', inner_matrix, self.Bit_t, self.Bit_x, self.Bit_y, self.Bit_z)
        if inner_matrix.ndim ==4:
            y = torch.einsum('qjkl,xj,yk,zl->qxyz', inner_matrix, self.Bit_x, self.Bit_y, self.Bit_z)
        return y
    
    def loss_ic_only(self,y_hat, y_ic, ic, lmd):
        #N= ic.shape[0]
        #a,b = torch.max(torch.abs(ic.reshape(N,-1)), axis=1)
        #a = a.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        
        true_surf = self.make_surface(ic)
        pred_surf = self.make_surface(y_ic)
        
        loss1 = torch.mean(((ic - y_ic)**2/(ic+.0001)**2) )
        loss2 = torch.mean(((true_surf - pred_surf)**2/(true_surf+.0001)**2) )
        
        print(loss1.detach().numpy(),loss2.detach().numpy())
        
        return loss2
    
    def computebsderivatives(self,U_full):
        return compute_bspline_derivatives_3d(U_full, 
                                            self.Bit_t, 
                                            self.Bit_x, 
                                            self.Bit_y, 
                                            self.Bit_z,
                                            self.Bit_t_derivative,
                                            self.Bit_x_derivative, 
                                            self.Bit_y_derivative, 
                                            self.Bit_z_derivative,
                                            self.Bit_x_second_derivative, 
                                            self.Bit_y_second_derivative, 
                                            self.Bit_z_second_derivative)
        
    def loss(self,y_hat, y_ic, ic, lmd):
           
        U_full = torch.zeros((ic.shape[0],
                              self.n_ctrl_pts_time,
                              self.n_ctrl_pts_state, 
                              self.n_ctrl_pts_state, 
                              self.n_ctrl_pts_state))
        
        U_full[:,0,:,:,:] = y_ic
        U_full[:,1:,:,:, :] = y_hat # Predicted control points 
        #U_full = y_hat
        
        # Compute B-spline derivatives (t, x, and x^2 derivatives)
        B_surface, B_s_t, B_s_x, B_s_y, B_s_z, B_s_xx = self.computebsderivatives(U_full) 
        res1= self.pde_residual(B_surface= B_surface,
                                 B_s_t=B_s_t, 
                                B_s_x=B_s_x,
                                B_s_y=B_s_y,
                                B_s_z=B_s_z, 
                                B_s_xx= B_s_xx)
        physics_loss = torch.mean(torch.sum(torch.pow(res1,2),axis = (2,3,4)))
        #initial condition loss
        true_ic_surface = self.make_surface(ic)
        pred_ic_surface = self.make_surface(y_ic)
        loss_ic_cpt = torch.mean(torch.sum((y_ic - ic)**2/(y_ic**2+ic**2+.0001),axis=(1,2,3) ))
        loss_ic =  torch.mean(torch.sum(torch.pow((B_surface[:,0,:,:,:] - true_ic_surface)/(true_ic_surface+.0000001),2), axis= (1,2,3)))
        loss_ic_2 =  torch.mean(torch.sum(torch.pow((B_surface[:,0,:,:,:] - pred_ic_surface)/(true_ic_surface+.0000001),2), axis= (1,2,3)))
        loss_ic_3 = torch.mean(torch.sum(torch.pow((pred_ic_surface- true_ic_surface)/(true_ic_surface+.0000001),2), axis= (1,2,3)))
        print(loss_ic.detach().numpy(), loss_ic_2.detach().numpy(), loss_ic_3.detach().numpy())
        print(torch.sum(pred_ic_surface, axis = (1,2,3)))
        print(torch.sum(true_ic_surface, axis = (1,2,3)))
         
        #data loss at a random timestep
        ts = np.random.randint(self.n_points)
        time =self.t[ts]
        true_surface = self.estimate_true_surface_at_timestep(lmd, time) 
        data_loss = torch.mean(torch.sum(torch.pow(B_surface[:,ts,:,:,:] - true_surface,2), axis = (1,2,3))) 
        
        print(f" {physics_loss.detach().numpy():02.7f},"+  
              f" {data_loss.detach().numpy():02.7f}"+
              f" {loss_ic.detach().numpy():02.7f},"+
              f" {loss_ic_cpt.detach().numpy():02.7f},"+  
              f" {torch.sum(B_surface[:,0,:,:,:], axis = (0, 3,1,2)).detach().numpy()}")
        
        loss = loss_ic_cpt #+ data_loss
        
        if torch.isnan(loss):
            print("LOSS IS NAN")
            exit()
        return loss #
            
        
    def train(self, config, **kwargs):
        logger.info("--- Train ---")
        #setup from config file
        self.setup(config, **kwargs)    
        
        self.model = self.model.train()
        
        if not self.is_configured:
            logger.warning("Is not configured.")
            return False
    
        # Define the initial condition with the size of control points grid
        self.initial_conditions, _ = self.gaussian_pdf(self.lambda_train[:,:3], self.lambda_train[:,3:])
        
        logger.info(" Training loop commences")
        min_loss = 1e12
        isSaved = False
       
        for epoch in range(self.model_params.get('max_epochs')):
        
            if not self.optim_needs_closure:
                self.optimizer.zero_grad()
                tloss = self.closure()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=500)
                self.optimizer.step()
            if self.optim_needs_closure:
                self.optimizer.step(self.closure)
                tloss = self.closure() 
                
            if min_loss*0.9> (tloss.item()-1e-8):
                if epoch> self.model_params.get("max_epochs")//100 or not isSaved:
                    self.save_checkpoint(tloss)
                    isSaved =True
                    min_loss = tloss.item()
                
                
            print(f" {epoch+1:05d} total_loss = {tloss.item():2.5f},"+
                  f" min_loss {min_loss:2.7f}")#, end="")
        
        
    def closure(self):
        self.optimizer.zero_grad()
        
        y_hat, y_ic = self.model(self.lambda_train[:1]) 
        tloss = self.loss_ic_only(y_hat,y_ic, self.initial_conditions, self.lambda_train)
        tloss.backward() 
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=500)
        return tloss  
    
    
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