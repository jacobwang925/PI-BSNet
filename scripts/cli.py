import logging
import lightning as L
from lightning.pytorch.cli import LightningCLI
from bsnet.model import BSNet
from bsnet.data import BSDataModule

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class myCLI(LightningCLI):
    
    def add_arguments_to_parser(self, parser):
        """ defines which arguments are related to 
        each other to reduce redundancy (and mistakes) 
        in defining the configuration.
        """
        parser.link_arguments(source="model.bspline.init_args.n_ctrl_pts_state", target="data.n_ctrl_pts_state")
        parser.link_arguments(source="model.bspline.init_args.n_ctrl_pts_time", target="model.model.init_args.n_ctrl_pts_time")
        parser.link_arguments(source="model.bspline.init_args.n_ctrl_pts_state", target="model.model.init_args.n_ctrl_pts_state")
        parser.link_arguments(source="data.dimension", target="model.bspline.init_args.dimension")
        parser.link_arguments(source="data.dimension", target="model.dimension")
        parser.link_arguments(source="data.dimension", target="model.model.init_args.dimension")
        parser.link_arguments(source="data.input_size", target="model.model.init_args.input_size", apply_on="instantiate")
        
        
        def c(model_name, run_version):
            return f"experiment/{model_name}/{run_version}/logs"
        parser.link_arguments(source=["model.model_name",
                                      "model.run_version"], target="trainer.logger.init_args.save_dir", 
                              apply_on="instantiate", 
                              compute_fn = c)
        




if __name__ == "__main__":
    cli = myCLI(BSNet, BSDataModule)