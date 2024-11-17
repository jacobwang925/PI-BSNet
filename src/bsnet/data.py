import lightning.pytorch as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class BSDataset(Dataset):
    """ dataset basic
    """
    def __init__(self, data_path: str, n_ctrl_pts_state: int, dimension: int = 3):

        assert dimension == 3, "Dimension must be 3. can expand later"

        # Load your data here
        self.parameters = torch.from_numpy(np.load(data_path).astype(np.float32))
        linspace = torch.linspace(0, 1, n_ctrl_pts_state, dtype=torch.float32)

        space_griding = [linspace for i in range(dimension)]
        grid = torch.meshgrid(*space_griding, indexing="ij")

        C = self.parameters
        for i in range(dimension):
            C = C.unsqueeze(-1)

        # we use these parameters to create the state at t=0
        self.at_time_0 = (
            (C[:, 1] * grid[0])
            + (C[:, 2] * grid[1])
            + (C[:, 3] * grid[2])
            + 0.5
            + C[:, 0]
        )

    def __len__(self):
        return self.parameters.shape[0]

    def __getitem__(self, idx):
        return {"X": self.parameters[idx], "y": self.at_time_0[idx]}


class BSDataModule(L.LightningDataModule):
    """ data module basic
    """
    def __init__(
        self,
        train_data_path,
        val_data_path,
        test_data_path,
        n_ctrl_pts_state,
        batch_size: int = 32,
        dimension: int = 3,
        input_size: int = 4,
    ):
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.n_ctrl_pts_state = n_ctrl_pts_state
        self.batch_size = batch_size
        self.dimension = dimension
        self.input_size = input_size 
        self.train_dataset = BSDataset(
            self.train_data_path, self.n_ctrl_pts_state, self.dimension
        )
        self.val_dataset = BSDataset(
            self.val_data_path, self.n_ctrl_pts_state, self.dimension
        )
        self.test_dataset = BSDataset(
            self.test_data_path, self.n_ctrl_pts_state, self.dimension
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
