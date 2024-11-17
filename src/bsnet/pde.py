import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)



class PDE:
    """Object stores PDE parameters and returns the PDE residual
    """
    def __init__(self,D):
        self.D = D
    def pde_residual(self, u, dudt, dudx, dudxdx):
        res1 = dudt - self.D * dudxdx
        return res1

        