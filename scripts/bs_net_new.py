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
            ret_gt.append(ground_truth(self.x, self.t, self.a, li))
        return ret_gt

    def make_surface(self, inner_matrix):
        # Generate the B-spline surface with the predicted control points
        y_true = torch.matmul(torch.matmul(self.Bit_t, inner_matrix), self.Bit_x.T)
        return y_true
    
    def run(self, config, cmd = 'train', **kwargs):
        
        #setup from config file
        self.setup(config)    
        
        if type(self.lmbda_train) == str:
            train_lambda_numpy_file = Path(self.lmbda_train)
            if train_lambda_numpy_file.is_file():
                self.lmbda_train = np.load(train_lambda_numpy_file) 
            else:
                raise ValueError(f"training lambda file: {self.lmbda_train} does not exist")
        else:
            self.lmbda_train = np.array(self.lmbda_train)
            
        if not self.is_configured:
            logger.warning("Is not configured.")
            return
        
        #overwrite from command line
        for name, value in kwargs.items():
            self.__setattr__(name, value)
        
        #print to visually inspect.
        self.print_attributes()
        
        if cmd.lower() == 'train':
            self.train()
        elif cmd.lower() == 'visualize':
            self.visualize()
        else:
            logger.warning(f"uninterpretable command {cmd}")
    
    def loss(self,y_hat, y_true , lmbda, physics_loss = True, data_loss=True, physics_loss_weight=1.0, data_loss_weight=1.0):
        loss = 0
        ploss = 0
        dloss = 0
        U_full = self.impose_icbc(y_hat) 
        #print(U_full)
        #print(y_true)
        if physics_loss:
            # Compute B-spline derivatives (t, x, and x^2 derivatives)
            B_s_t, B_s_x, B_s_xx = compute_bspline_derivatives(U_full, 
                                                                self.Bit_t, 
                                                                self.Bit_x, 
                                                                self.Bit_t_derivative,
                                                                self.Bit_x_derivative,
                                                                self.Bit_t_second_derivative, 
                                                                self.Bit_x_second_derivative)
            pde_residual = self.pde_residual(B_s_t=B_s_t, 
                                             B_s_x=B_s_x, 
                                             B_s_xx= B_s_xx, 
                                             lmbda = lmbda)
            physics_loss = torch.mean(torch.pow(pde_residual,2))
            ploss = physics_loss.detach()
            loss +=physics_loss*physics_loss_weight
            
        if data_loss:
            m_hat = self.make_surface(U_full)
            data_loss = torch.mean(torch.pow(y_true- m_hat,2))
            loss +=data_loss*data_loss_weight
            dloss = data_loss.detach()
            
        return loss, ploss, dloss
        
    def train(self):
        logger.info("--- Train ---")
        
        # loss configuration
        phys_loss_params= self.loss_params.get("physics_loss")
        phys_loss_cadenc = phys_loss_params.get("cadence")
        phys_loss_weight = phys_loss_params.get("weight")
        data_loss_params= self.loss_params.get("data_loss")
        data_loss_cadenc = data_loss_params.get("cadence")
        data_loss_weight = data_loss_params.get("weight")
        
        #how often to newline print_outs
        newline_rate = max(100, phys_loss_cadenc*data_loss_cadenc)
        
        if min(phys_loss_cadenc, data_loss_cadenc)>1:
            logger.warning("At least one of either physics or data must be used for loss every epoch")
            phys_loss_cadenc = 1
        
        #reshape lambdas and make into torch tesnor.
        tch_lmbda = torch.tensor(self.lmbda_train).reshape(-1,1)
        
        #calculate ground truth
        if data_loss_cadenc<=0 or data_loss_weight<=0:
            # do not calculate ground trouth if data loss is not being used.
            # set to zero
            scenario_gt = [torch.zeros(1,1) for i in self.lmbda_train]
            data_loss_weight = 0
            data_loss_cadenc = np.pi
        else:
            scenario_gt = self.build_scenario(self.lmbda_train)
        
        
        logger.info(" Training loop commences")
        

        min_loss = 1e12
        isSaved = False
                    
        for epoch in range(self.model_params.get('max_epochs')):
            sum_tloss=0
            sum_ploss=0
            sum_dloss=0
        
            use_physics = (epoch%phys_loss_cadenc)==0
            use_data = (epoch%data_loss_cadenc)==0
            
            for y_true, lmbda in zip(scenario_gt, tch_lmbda ):
                self.optimizer.zero_grad()
                y_hat = self.model(lmbda)
                
                tloss, ploss, dloss = self.loss(y_hat, 
                                  y_true, 
                                  lmbda, 
                                  use_physics, 
                                  use_data, 
                                  phys_loss_weight, 
                                  data_loss_weight)
                sum_tloss += tloss
                sum_ploss += ploss
                sum_dloss += dloss
                

            sum_tloss.backward()
            self.optimizer.step()
            
            if min_loss*0.9> (sum_tloss.item()-1e-8):
                if epoch> self.model_params.get("max_epochs")//10 or not isSaved:
                    self.save_checkpoint(sum_tloss)
                    isSaved =True
                min_loss = sum_tloss.item()
                
            if (epoch-1)%newline_rate==0:
                print()
            print(f" {epoch+1:05d} total_loss = {sum_tloss.item():2.5f},"+
                  f" physics_loss = {sum_ploss:2.7f}, data_loss = {sum_dloss:2.7f}"+
                  f" min_loss {min_loss:2.7f}", end="\r")
        print()
        
    def visualize(self):
        if not self.__getattribute__("vis_lmbda"):
            raise ValueError("need attribute vis_lmbda")
        
        if not self.__getattribute__("checkpoint"):
            raise ValueError("need attribute checkpoint")
        
        
        logger.info("--- Visualize  ---")
        self.model = self.model.eval()
        self.load_checkpoint()
        
        with torch.no_grad():
            
            # After training, test the network with a different lambda value
            y_hat = self.model(torch.tensor([[self.vis_lmbda]], dtype=torch.float32))
            U_full = self.impose_icbc(y_hat)
            y_hat_surface = self.make_surface(U_full)

            # Generate ground truth data for the testing lambda
            y_true_surface = ground_truth(self.x, self.t, self.a, self.vis_lmbda)

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
            ax1.set_title(f'B-spline Surface Prediction with Lambda = {self.vis_lmbda}')
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

    
    

            
if __name__ =="__main__":
    fire.Fire(bsnet_train)
    logger.info("Done")