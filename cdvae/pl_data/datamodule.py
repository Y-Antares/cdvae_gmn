import random
from typing import Optional, Sequence
from pathlib import Path

import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from cdvae.common.utils import PROJECT_ROOT
from cdvae.common.data_utils import get_scaler_from_data_list, MinMaxScalerTorch


def worker_init_fn(id: int):
    """
    DataLoaders workers init function.
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


class CrystDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        scaler_path=None,
        scaler_type='standard',  # 添加标准化器类型参数
        energy_scaler_type='standard',  # 添加能量标准化器类型参数
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.scaler_type = scaler_type
        self.energy_scaler_type = energy_scaler_type

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

        self.lattice_scaler = None
        self.scaler = None
        self.energy_scaler = None

        self.get_scaler(scaler_path)

    def prepare_data(self) -> None:
        pass

    def get_scaler(self, scaler_path):
        if scaler_path is None:
            train_dataset = hydra.utils.instantiate(self.datasets.train)
            self.lattice_scaler = get_scaler_from_data_list(
                train_dataset.cached_data, key='scaled_lattice', scaler_type=self.scaler_type)
            self.scaler = get_scaler_from_data_list(
                train_dataset.cached_data, key=train_dataset.prop, scaler_type=self.scaler_type)
            
            # 安全获取形成能的标准化器
            try:
                self.energy_scaler = get_scaler_from_data_list(
                    train_dataset.cached_data, key='formation_energy_per_atom', 
                    scaler_type=self.energy_scaler_type)
            except KeyError:
                print("警告：找不到形成能数据，使用默认标准化器")
                if self.energy_scaler_type.lower() == 'minmax':
                    self.energy_scaler = MinMaxScalerTorch(mins=torch.tensor([0.0]), maxs=torch.tensor([1.0]))
                else:
                    from cdvae.common.data_utils import StandardScaler
                    self.energy_scaler = StandardScaler(mean=0.0, std=1.0)
        else:
            self.lattice_scaler = torch.load(Path(scaler_path) / 'lattice_scaler.pt')
            self.scaler = torch.load(Path(scaler_path) / 'prop_scaler.pt')
            
            energy_scaler_path = Path(scaler_path) / 'energy_scaler.pt'
            if energy_scaler_path.exists():
                self.energy_scaler = torch.load(energy_scaler_path)
            else:
                print(f"警告：找不到能量标准化器文件 {energy_scaler_path}，使用默认标准化器")
                if self.energy_scaler_type.lower() == 'minmax':
                    self.energy_scaler = MinMaxScalerTorch(mins=torch.tensor([0.0]), maxs=torch.tensor([1.0]))
                else:
                    from cdvae.common.data_utils import StandardScaler
                    self.energy_scaler = StandardScaler(mean=0.0, std=1.0)
                    
    def setup(self, stage: Optional[str] = None):
        if stage is None or stage == "fit":
            self.train_dataset = hydra.utils.instantiate(self.datasets.train)
            self.val_datasets = [
                hydra.utils.instantiate(dataset_cfg)
                for dataset_cfg in self.datasets.val
            ]

            self.train_dataset.lattice_scaler = self.lattice_scaler
            self.train_dataset.scaler = self.scaler
            self.train_dataset.energy_scaler = self.energy_scaler

            for val_dataset in self.val_datasets:
                val_dataset.lattice_scaler = self.lattice_scaler
                val_dataset.scaler = self.scaler
                val_dataset.energy_scaler = self.energy_scaler

        if stage is None or stage == "test":
            self.test_datasets = [
                hydra.utils.instantiate(dataset_cfg)
                for dataset_cfg in self.datasets.test
            ]
            for test_dataset in self.test_datasets:
                test_dataset.lattice_scaler = self.lattice_scaler
                test_dataset.scaler = self.scaler
                test_dataset.energy_scaler = self.energy_scaler

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                worker_init_fn=worker_init_fn,
            )
            for dataset in self.val_datasets
        ]

    def test_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                worker_init_fn=worker_init_fn,
            )
            for dataset in self.test_datasets
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3")
def main(cfg: omegaconf.DictConfig):
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    datamodule.setup('fit')
    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    main()