import fire
import torch
import torch.optim as optim
import numpy as np
import logging
from pathlib import Path
from matplotlib import pyplot as plt
from utils import ground_truth, BsKnots, BsKnots_derivatives, compute_bspline_derivatives
from modules import ControlPointNet
from base_run import base_run


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
plt.set_loglevel (level = 'warning')

class bsnet_train(base_run):
    def __init__(self, cfg=None):
        #super(bsnet_train, base_run).__init__()
        self.__setattr__("is_configured", False)
    
    def pde_residual(self, **kwargs):
        pde_residual = kwargs["B_s_t"] - kwargs["lmbda"]*kwargs["B_s_x"] - 0.5*kwargs["B_s_xx"]
        return pde_residual
    
    def build_scenario(self, lmbda=None):
        ret_gt = []
        
        for li in lmbda:
            logger.info(f"   ...building scenario for lambda = {li}")
            ret_gt.append(ground_truth(self.x, self.t, self.max_X, li))
        return ret_gt

    def make_surface(self, inner_matrix):
        # Generate the B-spline surface with the predicted control points
        y_true = torch.matmul(torch.matmul(self.Bit_t, inner_matrix), self.Bit_x.T)
        return y_true
    
            
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
    def loss(self,y_hat, y_true , lmbda, physics_loss = True, data_loss=True, physics_loss_weight=1.0, data_w=1.0):
        loss = 0
        ploss = 0
        dloss = 0
        U_full = torch.stack([self.impose_icbc(y_hat[i]) for i in range(y_hat.shape[0])], dim=0)  
        if physics_loss:
            pde_residuals = []
            for i in range(y_hat.shape[0]):
                # Compute B-spline derivatives (t, x, and x^2 derivatives)
                B_s_t, B_s_x, B_s_xx = compute_bspline_derivatives(U_full[i], 
                                                                    self.Bit_t, 
                                                                    self.Bit_x, 
                                                                    self.Bit_t_derivative,
                                                                    self.Bit_x_derivative,
                                                                    self.Bit_t_second_derivative, 
                                                                    self.Bit_x_second_derivative)
                pde_residuals.append(self.pde_residual(B_s_t=B_s_t, 
                                                B_s_x=B_s_x, 
                                                B_s_xx= B_s_xx, 
                                                lmbda = lmbda[i]))
                
            physics_loss = torch.mean(torch.pow(torch.hstack(pde_residuals),2))
            ploss = physics_loss.detach()
                
            loss +=physics_loss*physics_loss_weight
            
        if data_loss:
            data_loss =0
            for i in range(y_hat.shape[0]):
                m_hat = self.make_surface(U_full[i])
                data_loss += torch.pow(y_true[i]- m_hat,2)
            loss +=torch.mean(data_loss)*data_w
            dloss = torch.mean(data_loss).detach()
            
        return loss, ploss, dloss
        
    def train(self, config, **kwargs):
        logger.info("--- Train ---")
        #setup from config file
        self.setup(config, **kwargs)    
        
        self.model = self.model.train()
        
        if not self.is_configured:
            logger.warning("Is not configured.")
            return
    
        # loss configuration
        phys_params= self.loss_params.get("physics_loss")
        phys_c = phys_params.get("cadence")
        phys_w = phys_params.get("weight")
        data_params= self.loss_params.get("data_loss")
        data_c = data_params.get("cadence")
        data_w = data_params.get("weight")
        
        #how often to newline print_outs
        newline_rate = max(100, phys_c*data_c)
        
        if min(phys_c, data_c)>1:
            logger.warning("At least one of either physics or data must be used for loss every epoch")
            phys_c = 1
        
        #reshape lambdas and make into torch tensor.
        lmd = torch.tensor(self.lmbda_train).reshape(-1,1)
        
        #calculate ground truth
        if data_c<=0 or data_w<=0:
            # do not calculate ground trouth if data loss is not being used.
            # set to zero
            y_tru = [torch.zeros(1,1) for i in self.lmbda_train]
            data_w, data_c = 0, np.pi
        else:
            y_tru = self.build_scenario(self.lmbda_train)
        
        
        logger.info(" Training loop commences")
        min_loss = 1e12
        isSaved = False
                    
        for epoch in range(self.model_params.get('max_epochs')):
        
            use_phys = (epoch%phys_c)==0
            use_data = (epoch%data_c)==0
            if not self.optim_needs_closure:
                self.optimizer.zero_grad()
                
            u_hat = self.model(lmd)
            
            tloss, ploss, dloss = self.loss(u_hat, y_tru, lmd, use_phys, use_data, phys_w, data_w)
                
            if self.optim_needs_closure:
                def closure():
                    self.optimizer.zero_grad()
                    
                    u_hat = self.model(lmd) 
                    tloss, _, _ = self.loss(u_hat,  y_tru, lmd, use_phys,  use_data,  phys_w, data_w)
                    tloss.backward()  
                    return tloss
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
                  f" physics_loss = {ploss:2.7f}, data_loss = {dloss:2.7f}"+
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
            y_hat_surface = self.make_surface(U_full)

            # Generate ground truth data for the testing lambda
            y_true_surface = ground_truth(self.x, self.t, self.max_X, vis_lmbda)

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
            
    def test(self, config, with_train =False, **kwargs):
    
        #setup from config file
        self.setup(config, **kwargs)    
        if not self.is_configured:
            logger.warning("Is not configured.")
            return
        
        if self.__getattribute__("checkpoint") is None:
            logger.error("no checkpoint for training")
            exit()
            
        self.model = self.model.eval()
        
        #build datasets and dataloaders
        fig = plt.figure(figsize=(6, 5))
        ax0 = fig.add_subplot(111)
        if with_train:
            train_lmda = torch.tensor(self.lmbda_train).reshape(-1,1).to(torch.float32)
            y_tru_train = torch.stack(self.build_scenario(self.lmbda_train), dim=0)
            train_ds = torch.utils.data.TensorDataset(train_lmda, y_tru_train)
            train_dl = torch.utils.data.DataLoader(train_ds, batch_size = 1, shuffle = False)
            mse_y_train_mean = []
            mse_y_train_std = []
            for x, y in train_dl:
                u_hat = self.model(x)
                u_hat = self.impose_icbc(u_hat)
                y_hat = self.make_surface(u_hat)
                
                mse_y_train_mean.append( torch.mean(torch.pow(y-y_hat,2)).detach().numpy())
                mse_y_train_std.append(torch.mean(torch.pow(y-y_hat,2)).detach().numpy())
            
            
            ax0.errorbar(self.lmbda_train, mse_y_train_mean, 
                     yerr=mse_y_train_std,ecolor='gray', 
                     capsize=8,
                     ls='none')
            ax0.scatter(self.lmbda_train, mse_y_train_mean, 
                     color = "k", s=12,
                     label = "Train") 
            
        test_lmda = torch.tensor(self.lmbda_test).reshape(-1,1).to(torch.float32)            
        y_tru_test = torch.stack(self.build_scenario(self.lmbda_test), dim=0)
        test_ds = torch.utils.data.TensorDataset(test_lmda, y_tru_test)
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size = 1, shuffle = False) 
        mse_y_test_mean = []
        mse_y_test_std = []    
        for x, y in test_dl:
            u_hat = self.model(x)
            u_hat = self.impose_icbc(u_hat)
            y_hat = self.make_surface(u_hat)
            
            mse_y_test_mean.append( torch.mean(torch.pow(y-y_hat,2)).detach().numpy())
            mse_y_test_std.append(torch.mean(torch.pow(y-y_hat,2)).detach().numpy())
           
        ax0.errorbar(self.lmbda_test, mse_y_test_mean, 
                     yerr=mse_y_test_std,ecolor='gray', 
                     capsize=8,
                     ls='none') 
        
        ax0.scatter(self.lmbda_test, mse_y_test_mean, 
                     color = "red", s=12,
                     label = "Test") 
        ax0.legend()
        ax0.set_xlabel("$\lambda$", fontsize=18)
        ax0.set_ylabel("mean square error", fontsize=18)
        
        plt.show()
        
        

        
            
            
             
if __name__ =="__main__":
    fire.Fire(bsnet_train)
    logger.info("Done")