from typing import Any, Dict
import hydra
import numpy as np
import omegaconf
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter
from tqdm import tqdm

from cdvae.common.utils import PROJECT_ROOT
from cdvae.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume,
    frac_to_cart_coords, min_distance_sqr_pbc)
from cdvae.pl_modules.embeddings import MAX_ATOMIC_NUM
from cdvae.pl_modules.embeddings import KHOT_EMBEDDINGS


def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers-1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "train_loss_epoch"}


class CrystGNN_Supervise(BaseModule):
    """
    GNN model for fitting the supervised objectives for crystals.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = hydra.utils.instantiate(self.hparams.encoder)
        
        # 检查是否使用多目标预测 (检查 hparams 或根据配置决定)
        self.multi_target = getattr(self.hparams, 'multi_target', False)
        
        # 根据目标数量调整输出层
        if self.multi_target:
            # 适配模型输出维度为2（formation_energy和目标属性）
            self.out_proj = nn.Linear(self.encoder.out_dim, 2)
        else:
            # 保持原有单目标预测
            self.out_proj = nn.Linear(self.encoder.out_dim, 1)
            
        # 添加标准化器（将在模型加载时设置）
        self.lattice_scaler = None
        self.scaler = None
        self.energy_scaler = None

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        x = self.encoder(batch)
        preds = self.out_proj(x)  # (N, 2) 或 (N, 1)
        return preds

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        preds = self(batch)
        
        # 处理多目标情况
        if self.multi_target:
            # 检查 batch.y 的形状和 preds 的兼容性
            if batch.y.dim() == 3 and batch.y.size(2) == 2:
                # batch.y 形状为 [batch_size, 1, 2]，需要调整为 [batch_size, 2]
                target = batch.y.squeeze(1)
            else:
                # 如果不是预期的形状，打印警告并尝试处理
                print(f"警告: batch.y 形状不是预期的 [batch_size, 1, 2]，而是 {batch.y.shape}")
                if batch.y.dim() == 2 and batch.y.size(1) == 2:
                    target = batch.y
                else:
                    # 尝试调整为兼容形状
                    target = batch.y.view(-1, 2)
            
            # 使用MSE损失
            loss = F.mse_loss(preds, target)
            
            # 可以添加加权损失
            energy_weight = getattr(self.hparams, 'energy_weight', 0.5)
            
            # 分别计算形成能和目标属性的损失
            energy_loss = F.mse_loss(preds[:, 0:1], target[:, 0:1])
            target_loss = F.mse_loss(preds[:, 1:2], target[:, 1:2])
            
            # 加权组合
            loss = energy_weight * energy_loss + (1 - energy_weight) * target_loss
            
            self.log_dict({
                'train_loss': loss,
                'train_energy_loss': energy_loss,
                'train_target_loss': target_loss
            }, on_step=True, on_epoch=True, prog_bar=True)
        else:
            # 原始单目标损失
            loss = F.mse_loss(preds, batch.y)
            self.log_dict({'train_loss': loss}, on_step=True, on_epoch=True, prog_bar=True)
            
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        preds = self(batch)
        log_dict, loss = self.compute_stats(batch, preds, prefix='val')
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        preds = self(batch)
        log_dict, loss = self.compute_stats(batch, preds, prefix='test')
        self.log_dict(log_dict)
        return loss

    def compute_stats(self, batch, preds, prefix):
        # 处理多目标情况
        if self.multi_target:
            # 调整目标张量形状以匹配预测
            if batch.y.dim() == 3 and batch.y.size(2) == 2:
                target = batch.y.squeeze(1)  # [batch_size, 2]
            else:
                # 尝试处理其他形状
                if batch.y.dim() == 2 and batch.y.size(1) == 2:
                    target = batch.y
                else:
                    target = batch.y.view(-1, 2)
            
            # 总体MSE损失
            loss = F.mse_loss(preds, target)
            
            # 确保标准化器可用并且在正确的设备上
            if self.scaler is not None:
                self.scaler.match_device(preds)
            if self.energy_scaler is not None:
                self.energy_scaler.match_device(preds)
            
            # 分别计算两个目标的指标
            # 形成能（第一个目标）
            energy_preds = preds[:, 0:1]
            energy_targets = target[:, 0:1]
            
            if self.energy_scaler is not None:
                scaled_energy_preds = self.energy_scaler.inverse_transform(energy_preds)
                scaled_energy_targets = self.energy_scaler.inverse_transform(energy_targets)
                energy_mae = torch.mean(torch.abs(scaled_energy_preds - scaled_energy_targets))
            else:
                # 如果没有标准化器，使用原始值
                energy_mae = torch.mean(torch.abs(energy_preds - energy_targets))
            
            # 目标属性（第二个目标）
            target_preds = preds[:, 1:2]
            target_targets = target[:, 1:2]
            
            if self.scaler is not None:
                scaled_target_preds = self.scaler.inverse_transform(target_preds)
                scaled_target_targets = self.scaler.inverse_transform(target_targets)
                target_mae = torch.mean(torch.abs(scaled_target_preds - scaled_target_targets))
            else:
                # 如果没有标准化器，使用原始值
                target_mae = torch.mean(torch.abs(target_preds - target_targets))
            
            log_dict = {
                f'{prefix}_loss': loss,
                f'{prefix}_energy_mae': energy_mae,
                f'{prefix}_target_mae': target_mae,
            }
            
            # 如果是晶格属性预测，添加相关指标
            if self.hparams.data.prop == 'scaled_lattice':
                pred_lengths = scaled_target_preds[:, :3]
                pred_angles = scaled_target_preds[:, 3:]
                if self.hparams.data.lattice_scale_method == 'scale_length':
                    pred_lengths = pred_lengths * batch.num_atoms.view(-1, 1).float()**(1/3)
                lengths_mae = torch.mean(torch.abs(pred_lengths - batch.lengths))
                angles_mae = torch.mean(torch.abs(pred_angles - batch.angles))
                lengths_mard = mard(batch.lengths, pred_lengths)
                angles_mard = mard(batch.angles, pred_angles)

                pred_volumes = lengths_angles_to_volume(pred_lengths, pred_angles)
                true_volumes = lengths_angles_to_volume(batch.lengths, batch.angles)
                volumes_mard = mard(true_volumes, pred_volumes)
                
                log_dict.update({
                    f'{prefix}_lengths_mae': lengths_mae,
                    f'{prefix}_angles_mae': angles_mae,
                    f'{prefix}_lengths_mard': lengths_mard,
                    f'{prefix}_angles_mard': angles_mard,
                    f'{prefix}_volumes_mard': volumes_mard,
                })
        else:
            # 原始单目标逻辑
            loss = F.mse_loss(preds, batch.y)
            
            if self.scaler is not None:
                self.scaler.match_device(preds)
                scaled_preds = self.scaler.inverse_transform(preds)
                scaled_y = self.scaler.inverse_transform(batch.y)
                mae = torch.mean(torch.abs(scaled_preds - scaled_y))
            else:
                mae = torch.mean(torch.abs(preds - batch.y))

            log_dict = {
                f'{prefix}_loss': loss,
                f'{prefix}_mae': mae,
            }

            if self.hparams.data.prop == 'scaled_lattice':
                pred_lengths = scaled_preds[:, :3]
                pred_angles = scaled_preds[:, 3:]
                if self.hparams.data.lattice_scale_method == 'scale_length':
                    pred_lengths = pred_lengths * batch.num_atoms.view(-1, 1).float()**(1/3)
                lengths_mae = torch.mean(torch.abs(pred_lengths - batch.lengths))
                angles_mae = torch.mean(torch.abs(pred_angles - batch.angles))
                lengths_mard = mard(batch.lengths, pred_lengths)
                angles_mard = mard(batch.angles, pred_angles)

                pred_volumes = lengths_angles_to_volume(pred_lengths, pred_angles)
                true_volumes = lengths_angles_to_volume(batch.lengths, batch.angles)
                volumes_mard = mard(true_volumes, pred_volumes)
                
                log_dict.update({
                    f'{prefix}_lengths_mae': lengths_mae,
                    f'{prefix}_angles_mae': angles_mae,
                    f'{prefix}_lengths_mard': lengths_mard,
                    f'{prefix}_angles_mard': angles_mard,
                    f'{prefix}_volumes_mard': volumes_mard,
                })
                
        return log_dict, loss


class CDVAE(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = hydra.utils.instantiate(
            self.hparams.encoder, num_targets=self.hparams.latent_dim)
        self.decoder = hydra.utils.instantiate(self.hparams.decoder)

        self.fc_mu = nn.Linear(self.hparams.latent_dim,
                               self.hparams.latent_dim)
        self.fc_var = nn.Linear(self.hparams.latent_dim,
                                self.hparams.latent_dim)
        
        # self.hparams.max_atoms = 100 # 防止assertion error
        self.fc_num_atoms = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                      self.hparams.fc_num_layers, self.hparams.max_atoms+1)
        self.fc_lattice = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                    self.hparams.fc_num_layers, 6)
        self.fc_composition = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                        self.hparams.fc_num_layers, MAX_ATOMIC_NUM)
                                        
        # 检查属性预测配置
        self.predict_property = getattr(self.hparams, 'predict_property', False)
        self.optimization_direction = getattr(self.hparams, 'optimization_direction', None)
        # 获取优化方法和边界参数
        self.optimization_method = getattr(self.hparams, 'optimization_method', 'weighted')
        self.boundary_theta = getattr(self.hparams, 'boundary_theta', 5.0)
        
        # 属性预测模块
        if self.predict_property:
            # 使用共享特征层
            self.fc_property_shared = build_mlp(
                self.hparams.latent_dim, 
                self.hparams.hidden_dim,
                self.hparams.fc_num_layers - 1,  # 减少一层，后面用专用层
                self.hparams.hidden_dim
            )
            
            # 形成能预测头
            self.energy_head = nn.Linear(self.hparams.hidden_dim, 1)
            
            # 目标属性预测头
            self.target_head = nn.Linear(self.hparams.hidden_dim, 1)
            
            # 是否使用加权损失
            self.property_weights = None
            if hasattr(self.hparams, 'property_weights'):
                self.property_weights = self.hparams.property_weights
        
        # 添加标准化器（将在模型加载时设置）
        self.lattice_scaler = None
        self.scaler = None
        self.energy_scaler = None

        sigmas = torch.tensor(np.exp(np.linspace(
            np.log(self.hparams.sigma_begin),
            np.log(self.hparams.sigma_end),
            self.hparams.num_noise_level)), dtype=torch.float32)

        self.sigmas = nn.Parameter(sigmas, requires_grad=False)

        type_sigmas = torch.tensor(np.exp(np.linspace(
            np.log(self.hparams.type_sigma_begin),
            np.log(self.hparams.type_sigma_end),
            self.hparams.num_noise_level)), dtype=torch.float32)

        self.type_sigmas = nn.Parameter(type_sigmas, requires_grad=False)

        self.embedding = torch.zeros(100, 92)
        for i in range(100):
            self.embedding[i] = torch.tensor(KHOT_EMBEDDINGS[i + 1])

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, batch):
        """
        encode crystal structures to latents.
        """
        hidden = self.encoder(batch)
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        z = self.reparameterize(mu, log_var)
        return mu, log_var, z

    def decode_stats(self, z, gt_num_atoms=None, gt_lengths=None, gt_angles=None,
                     teacher_forcing=False):
        """
        decode key stats from latent embeddings.
        batch is input during training for teach-forcing.
        """
        if gt_num_atoms is not None:
            num_atoms = self.predict_num_atoms(z)
            lengths_and_angles, lengths, angles = (
                self.predict_lattice(z, gt_num_atoms))
            composition_per_atom = self.predict_composition(z, gt_num_atoms)
            if hasattr(self.hparams, 'teacher_forcing_lattice') and self.hparams.teacher_forcing_lattice and teacher_forcing:
                lengths = gt_lengths
                angles = gt_angles
        else:
            num_atoms = self.predict_num_atoms(z).argmax(dim=-1)
            lengths_and_angles, lengths, angles = (
                self.predict_lattice(z, num_atoms))
            composition_per_atom = self.predict_composition(z, num_atoms)
        return num_atoms, lengths_and_angles, lengths, angles, composition_per_atom

    @torch.no_grad()
    def langevin_dynamics(self, z, ld_kwargs, gt_num_atoms=None, gt_atom_types=None):
        """
        decode crystral structure from latent embeddings.
        ld_kwargs: args for doing annealed langevin dynamics sampling:
            n_step_each:  number of steps for each sigma level.
            step_lr:      step size param.
            min_sigma:    minimum sigma to use in annealed langevin dynamics.
            save_traj:    if <True>, save the entire LD trajectory.
            disable_bar:  disable the progress bar of langevin dynamics.
        gt_num_atoms: if not <None>, use the ground truth number of atoms.
        gt_atom_types: if not <None>, use the ground truth atom types.
        """
        if ld_kwargs.save_traj:
            all_frac_coords = []
            all_pred_cart_coord_diff = []
            all_noise_cart = []
            all_atom_types = []

        # obtain key stats.
        num_atoms, _, lengths, angles, composition_per_atom = self.decode_stats(
            z, gt_num_atoms)
        if gt_num_atoms is not None:
            num_atoms = gt_num_atoms

        # obtain atom types.
        composition_per_atom = F.softmax(composition_per_atom, dim=-1)
        if gt_atom_types is None:
            cur_atom_types = self.sample_composition(
                composition_per_atom, num_atoms)
        else:
            cur_atom_types = gt_atom_types

        # init coords.
        cur_frac_coords = torch.rand((num_atoms.sum(), 3), device=z.device)

        # annealed langevin dynamics.
        for sigma in tqdm(self.sigmas, total=self.sigmas.size(0), disable=ld_kwargs.disable_bar):
            if sigma < ld_kwargs.min_sigma:
                break
            step_size = ld_kwargs.step_lr * (sigma / self.sigmas[-1]) ** 2

            for step in range(ld_kwargs.n_step_each):
                noise_cart = torch.randn_like(
                    cur_frac_coords) * torch.sqrt(step_size * 2)
                pred_cart_coord_diff, pred_atom_types = self.decoder(
                    z, cur_frac_coords, cur_atom_types, num_atoms, lengths, angles)
                cur_cart_coords = frac_to_cart_coords(
                    cur_frac_coords, lengths, angles, num_atoms)
                pred_cart_coord_diff = pred_cart_coord_diff / sigma
                cur_cart_coords = cur_cart_coords + step_size * pred_cart_coord_diff + noise_cart
                cur_frac_coords = cart_to_frac_coords(
                    cur_cart_coords, lengths, angles, num_atoms)

                if gt_atom_types is None:
                    cur_atom_types = torch.argmax(pred_atom_types, dim=1) + 1

                if ld_kwargs.save_traj:
                    all_frac_coords.append(cur_frac_coords)
                    all_pred_cart_coord_diff.append(
                        step_size * pred_cart_coord_diff)
                    all_noise_cart.append(noise_cart)
                    all_atom_types.append(cur_atom_types)

        output_dict = {'num_atoms': num_atoms, 'lengths': lengths, 'angles': angles,
                       'frac_coords': cur_frac_coords, 'atom_types': cur_atom_types,
                       'is_traj': False}

        if ld_kwargs.save_traj:
            output_dict.update(dict(
                all_frac_coords=torch.stack(all_frac_coords, dim=0),
                all_atom_types=torch.stack(all_atom_types, dim=0),
                all_pred_cart_coord_diff=torch.stack(
                    all_pred_cart_coord_diff, dim=0),
                all_noise_cart=torch.stack(all_noise_cart, dim=0),
                is_traj=True))

        return output_dict

    def sample(self, num_samples, ld_kwargs):
        z = torch.randn(num_samples, self.hparams.hidden_dim,
                        device=self.device)
        samples = self.langevin_dynamics(z, ld_kwargs)
        return samples

    def forward(self, batch, teacher_forcing, training):
        # hacky way to resolve the NaN issue. Will need more careful debugging later.
        mu, log_var, z = self.encode(batch)

        (pred_num_atoms, pred_lengths_and_angles, pred_lengths, pred_angles,
         pred_composition_per_atom) = self.decode_stats(
            z, batch.num_atoms, batch.lengths, batch.angles, teacher_forcing)

        # sample noise levels.
        noise_level = torch.randint(0, self.sigmas.size(0),
                                    (batch.num_atoms.size(0),),
                                    device=self.device)
        used_sigmas_per_atom = self.sigmas[noise_level].repeat_interleave(
            batch.num_atoms, dim=0)

        type_noise_level = torch.randint(0, self.type_sigmas.size(0),
                                         (batch.num_atoms.size(0),),
                                         device=self.device)
        used_type_sigmas_per_atom = (
            self.type_sigmas[type_noise_level].repeat_interleave(
                batch.num_atoms, dim=0))

        # add noise to atom types and sample atom types.
        pred_composition_probs = F.softmax(
            pred_composition_per_atom.detach(), dim=-1)
        atom_type_probs = (
            F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM) +
            pred_composition_probs * used_type_sigmas_per_atom[:, None])
        rand_atom_types = torch.multinomial(
            atom_type_probs, num_samples=1).squeeze(1) + 1

        # add noise to the cart coords
        cart_noises_per_atom = (
            torch.randn_like(batch.frac_coords) *
            used_sigmas_per_atom[:, None])
        cart_coords = frac_to_cart_coords(
            batch.frac_coords, pred_lengths, pred_angles, batch.num_atoms)
        cart_coords = cart_coords + cart_noises_per_atom
        noisy_frac_coords = cart_to_frac_coords(
            cart_coords, pred_lengths, pred_angles, batch.num_atoms)

        pred_cart_coord_diff, pred_atom_types = self.decoder(
            z, noisy_frac_coords, rand_atom_types, batch.num_atoms, pred_lengths, pred_angles)

        # compute loss.
        num_atom_loss = self.num_atom_loss(pred_num_atoms, batch)
        lattice_loss = self.lattice_loss(pred_lengths_and_angles, batch)
        composition_loss = self.composition_loss(
            pred_composition_per_atom, batch.atom_types, batch)
        coord_loss = self.coord_loss(
            pred_cart_coord_diff, noisy_frac_coords, used_sigmas_per_atom, batch)
        type_loss = self.type_loss(pred_atom_types, batch.atom_types,
                                   used_type_sigmas_per_atom, batch)

        kld_loss = self.kld_loss(mu, log_var)

        # 计算属性预测损失（如果启用）
        if self.predict_property:
            property_loss, energy_loss, target_loss = self.property_loss(z, batch)
        else:
            property_loss, energy_loss, target_loss = 0., 0., 0.

        return {
            'num_atom_loss': num_atom_loss,
            'lattice_loss': lattice_loss,
            'composition_loss': composition_loss,
            'coord_loss': coord_loss,
            'type_loss': type_loss,
            'kld_loss': kld_loss,
            'property_loss': property_loss,
            'energy_loss': energy_loss,
            'target_loss': target_loss,
            'pred_num_atoms': pred_num_atoms,
            'pred_lengths_and_angles': pred_lengths_and_angles,
            'pred_lengths': pred_lengths,
            'pred_angles': pred_angles,
            'pred_cart_coord_diff': pred_cart_coord_diff,
            'pred_atom_types': pred_atom_types,
            'pred_composition_per_atom': pred_composition_per_atom,
            'target_frac_coords': batch.frac_coords,
            'target_atom_types': batch.atom_types,
            'rand_frac_coords': noisy_frac_coords,
            'rand_atom_types': rand_atom_types,
            'z': z,
        }

    def generate_rand_init(self, pred_composition_per_atom, pred_lengths,
                           pred_angles, num_atoms, batch):
        rand_frac_coords = torch.rand(num_atoms.sum(), 3,
                                      device=num_atoms.device)
        pred_composition_per_atom = F.softmax(pred_composition_per_atom,
                                              dim=-1)
        rand_atom_types = self.sample_composition(
            pred_composition_per_atom, num_atoms)
        return rand_frac_coords, rand_atom_types

    def sample_composition(self, composition_prob, num_atoms):
        """
        Samples composition such that it exactly satisfies composition_prob
        """
        batch = torch.arange(
            len(num_atoms), device=num_atoms.device).repeat_interleave(num_atoms)
        assert composition_prob.size(0) == num_atoms.sum() == batch.size(0)
        composition_prob = scatter(
            composition_prob, index=batch, dim=0, reduce='mean')

        all_sampled_comp = []

        for comp_prob, num_atom in zip(list(composition_prob), list(num_atoms)):
            comp_num = torch.round(comp_prob * num_atom)
            atom_type = torch.nonzero(comp_num, as_tuple=True)[0] + 1
            atom_num = comp_num[atom_type - 1].long()

            sampled_comp = atom_type.repeat_interleave(atom_num, dim=0)

            # if the rounded composition gives less atoms, sample the rest
            if sampled_comp.size(0) < num_atom:
                left_atom_num = num_atom - sampled_comp.size(0)

                left_comp_prob = comp_prob - comp_num.float() / num_atom

                left_comp_prob[left_comp_prob < 0.] = 0.
                left_comp = torch.multinomial(
                    left_comp_prob, num_samples=left_atom_num, replacement=True)
                # convert to atomic number
                left_comp = left_comp + 1
                sampled_comp = torch.cat([sampled_comp, left_comp], dim=0)

            sampled_comp = sampled_comp[torch.randperm(sampled_comp.size(0))]
            sampled_comp = sampled_comp[:num_atom]
            all_sampled_comp.append(sampled_comp)

        all_sampled_comp = torch.cat(all_sampled_comp, dim=0)
        assert all_sampled_comp.size(0) == num_atoms.sum()
        return all_sampled_comp

    def predict_num_atoms(self, z):
        return self.fc_num_atoms(z)

    def predict_property(self, z):
        """
        预测双属性：形成能和目标属性
        """
        # 确保标准化器可用
        if self.scaler is None or self.energy_scaler is None:
            print("警告: 标准化器未设置，无法预测属性")
            return torch.zeros((z.size(0), 2), device=z.device)
            
        # 共享特征提取
        shared_features = self.fc_property_shared(z)
        
        # 应用各自的预测头
        energy_pred = self.energy_head(shared_features)
        target_pred = self.target_head(shared_features)
        
        # 将标准化器设置到正确的设备上
        self.scaler.match_device(z)
        self.energy_scaler.match_device(z)
        
        # 分别对不同属性进行逆标准化
        energy_pred_scaled = self.energy_scaler.inverse_transform(energy_pred)
        target_pred_scaled = self.scaler.inverse_transform(target_pred)
        
        # 返回合并结果 [batch_size, 2]
        return torch.cat([energy_pred_scaled, target_pred_scaled], dim=1)

    def predict_lattice(self, z, num_atoms):
        if self.lattice_scaler is None:
            print("警告: lattice_scaler未设置，使用默认值")
            pred_lengths_and_angles = self.fc_lattice(z)
            pred_lengths = pred_lengths_and_angles[:, :3]
            pred_angles = pred_lengths_and_angles[:, 3:]
            return pred_lengths_and_angles, pred_lengths, pred_angles
            
        self.lattice_scaler.match_device(z)
        pred_lengths_and_angles = self.fc_lattice(z)  # (N, 6)
        scaled_preds = self.lattice_scaler.inverse_transform(
            pred_lengths_and_angles)
        pred_lengths = scaled_preds[:, :3]
        pred_angles = scaled_preds[:, 3:]
        if self.hparams.data.lattice_scale_method == 'scale_length':
            pred_lengths = pred_lengths * num_atoms.view(-1, 1).float()**(1/3)
        # <pred_lengths_and_angles> is scaled.
        return pred_lengths_and_angles, pred_lengths, pred_angles

    def predict_composition(self, z, num_atoms):
        z_per_atom = z.repeat_interleave(num_atoms, dim=0)
        pred_composition_per_atom = self.fc_composition(z_per_atom)
        return pred_composition_per_atom

    def num_atom_loss(self, pred_num_atoms, batch):
        num_classes = pred_num_atoms.shape[-1]  # 模型输出的类别数
        targets = batch.num_atoms
        
        # 调试：打印标签范围和模型输出维度
        # print(f"[DEBUG] 标签范围: {targets.min().item()} ~ {targets.max().item()}, 模型类别数: {num_classes}")
        
        # 检查标签是否合法
        if targets.min() < 0 or targets.max() >= num_classes:
            raise ValueError(
                f"非法标签值！应为 [0, {num_classes-1}], 但实际为 [{targets.min()}, {targets.max()}]"
            )
        
        return F.cross_entropy(pred_num_atoms, targets)

    # Tchebycheff分解 & 边界交叉法 property loss
    def tchebycheff_loss(self, energy_pred, target_pred):
        """
        实现Tchebycheff分解损失函数
        """
        # 获取理想点（每个目标的最小值）
        # 可以使用训练集中的最小值或动态更新
        if not hasattr(self, 'ideal_points'):
            # 初始化理想点，可在训练中更新
            self.register_buffer('ideal_points', torch.tensor([0.0, 0.0], device=self.device))
        
        # 获取权重
        weights = torch.tensor(self.hparams.property_weights, device=energy_pred.device)
        
        # 计算Tchebycheff损失
        energy_term = weights[0] * torch.abs(energy_pred - self.ideal_points[0])
        target_term = weights[1] * torch.abs(target_pred - self.ideal_points[1])
        
        # 取最大值作为损失
        return torch.max(torch.stack([energy_term, target_term], dim=1), dim=1)[0].mean()

    def boundary_intersection_loss(self, energy_pred, target_pred, theta=5.0):
        """
        实现边界交叉法损失函数
        """
        # 获取理想点
        if not hasattr(self, 'ideal_points'):
            self.register_buffer('ideal_points', torch.tensor([0.0, 0.0], device=self.device))
        
        # 获取权重
        weights = torch.tensor(self.hparams.property_weights, device=energy_pred.device)
        
        # 构建当前解向量和理想点向量
        f_z = torch.stack([energy_pred.squeeze(), target_pred.squeeze()], dim=1)
        z_star = self.ideal_points.to(f_z.device)
        
        # 计算向量差
        diff = f_z - z_star
        
        # 计算范数
        norm = torch.norm(diff, dim=1)
        
        # 计算夹角余弦值
        lambda_norm = torch.sqrt(weights[0]**2 + weights[1]**2)
        cos_angle = (weights[0] * diff[:, 0] + weights[1] * diff[:, 1]) / (norm * lambda_norm + 1e-8)
        
        # 计算d1和d2
        d1 = norm * cos_angle
        d2 = norm * torch.sqrt(1 - cos_angle**2 + 1e-8)
        
        return (d1 + theta * d2).mean()

    def update_ideal_points(self, energy_pred, target_pred):
        """
        更新理想点
        """
        if not hasattr(self, 'ideal_points'):
            self.register_buffer('ideal_points', torch.tensor([0.0, 0.0], device=self.device))
        
        with torch.no_grad():
            current_min_energy = energy_pred.min().item()
            current_min_target = target_pred.min().item()
            
            self.ideal_points[0] = min(self.ideal_points[0].item(), current_min_energy)
            self.ideal_points[1] = min(self.ideal_points[1].item(), current_min_target)
    
    def property_loss(self, z, batch):
        """
        正确计算多目标属性预测损失的修正版本。
        """
        if not (self.predict_property and hasattr(batch, 'y') and batch.y is not None):
            return torch.tensor(0.0, device=z.device), torch.tensor(0.0, device=z.device), torch.tensor(0.0, device=z.device)

        # 从 hparams 中安全地获取多目标配置
        multi_obj_config = self.hparams.get('multi_objective', {})
        optimization_direction = multi_obj_config.get('direction', ['min', 'min'])
        optimization_method = multi_obj_config.get('method', 'weighted')
        property_weights = torch.tensor(multi_obj_config.get('weights', [0.5, 0.5]), device=z.device)
        
        # 鲁棒地处理目标张量的形状，确保为 [batch_size, 2]
        target = batch.y.view(-1, 2)
            
        shared_features = self.fc_property_shared(z)
        energy_pred = self.energy_head(shared_features)
        target_pred = self.target_head(shared_features)
        
        energy_target = target[:, 0:1]
        target_target = target[:, 1:2]
        
        # 计算每个样本的损失 (reduction='none')，以便进行逐样本操作
        energy_loss_ind = F.mse_loss(energy_pred, energy_target, reduction='none')
        
        # 根据优化方向（最大化或最小化）处理第二个目标
        if optimization_direction[1] == 'max':
            target_loss_ind = F.mse_loss(-target_pred, -target_target, reduction='none')
        else:
            target_loss_ind = F.mse_loss(target_pred, target_target, reduction='none')
        
        # 计算用于日志记录的平均损失
        energy_loss_mean = energy_loss_ind.mean()
        target_loss_mean = target_loss_ind.mean()

        if optimization_method == 'tchebycheff':
            ideal_points = torch.tensor(multi_obj_config.get('init_ideal_points', [0.0, 0.0]), device=z.device)
            
            # 使用广播机制计算加权的绝对差值
            energy_term = property_weights[0] * torch.abs(energy_loss_ind - ideal_points[0])
            target_term = property_weights[1] * torch.abs(target_loss_ind - ideal_points[1])
            
            # 对每个样本取最大值，然后对整个批次求平均
            property_loss = torch.mean(torch.max(energy_term, target_term))

        elif optimization_method == 'boundary':
            ideal_points = torch.tensor(multi_obj_config.get('init_ideal_points', [0.0, 0.0]), device=z.device)
            theta = multi_obj_config.get('boundary_theta', 5.0)
            
            f_z = torch.cat([energy_loss_ind, target_loss_ind], dim=1)
            diff = f_z - ideal_points
            
            norm = torch.norm(diff, p=2, dim=1)
            
            # 使用广播进行计算
            cos_angle = torch.sum(property_weights * diff, dim=1) / (norm * torch.norm(property_weights) + 1e-8)
            # 确保 cos_angle 在 [-1, 1] 范围内以避免 sqrt 产生 NaN
            cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
            sin_angle = torch.sqrt(1 - cos_angle**2)
             
            d1 = norm * cos_angle
            d2 = norm * sin_angle
             
            property_loss = (d1 + theta * d2).mean()

        else: # 默认使用加权和 (Weighted Sum)
            property_loss = (property_weights[0] * energy_loss_mean + property_weights[1] * target_loss_mean)

        return property_loss, energy_loss_mean, target_loss_mean
        
    def lattice_loss(self, pred_lengths_and_angles, batch):
        if self.lattice_scaler is None:
            print("警告: lattice_scaler未设置，使用原始值计算损失")
            if self.hparams.data.lattice_scale_method == 'scale_length':
                target_lengths = batch.lengths / \
                    batch.num_atoms.view(-1, 1).float()**(1/3)
            target_lengths_and_angles = torch.cat(
                [target_lengths, batch.angles], dim=-1)
            return F.mse_loss(pred_lengths_and_angles, target_lengths_and_angles)
            
        self.lattice_scaler.match_device(pred_lengths_and_angles)
        if self.hparams.data.lattice_scale_method == 'scale_length':
            target_lengths = batch.lengths / \
                batch.num_atoms.view(-1, 1).float()**(1/3)
        target_lengths_and_angles = torch.cat(
            [target_lengths, batch.angles], dim=-1)
        target_lengths_and_angles = self.lattice_scaler.transform(
            target_lengths_and_angles)
        return F.mse_loss(pred_lengths_and_angles, target_lengths_and_angles)

    def composition_loss(self, pred_composition_per_atom, target_atom_types, batch):
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(pred_composition_per_atom,
                               target_atom_types, reduction='none')
        return scatter(loss, batch.batch, reduce='mean').mean()

    def coord_loss(self, pred_cart_coord_diff, noisy_frac_coords,
                   used_sigmas_per_atom, batch):
        noisy_cart_coords = frac_to_cart_coords(
            noisy_frac_coords, batch.lengths, batch.angles, batch.num_atoms)
        target_cart_coords = frac_to_cart_coords(
            batch.frac_coords, batch.lengths, batch.angles, batch.num_atoms)
        _, target_cart_coord_diff = min_distance_sqr_pbc(
            target_cart_coords, noisy_cart_coords, batch.lengths, batch.angles,
            batch.num_atoms, self.device, return_vector=True)

        target_cart_coord_diff = target_cart_coord_diff / \
            used_sigmas_per_atom[:, None]**2
        pred_cart_coord_diff = pred_cart_coord_diff / \
            used_sigmas_per_atom[:, None]

        loss_per_atom = torch.sum(
            (target_cart_coord_diff - pred_cart_coord_diff)**2, dim=1)

        loss_per_atom = 0.5 * loss_per_atom * used_sigmas_per_atom**2
        return scatter(loss_per_atom, batch.batch, reduce='mean').mean()

    def type_loss(self, pred_atom_types, target_atom_types, used_type_sigmas_per_atom, batch):
        # 确保目标原子类型在有效范围内
        n_classes = pred_atom_types.size(1)
        
        # 原子类型减1转换为索引
        target_atom_types = target_atom_types - 1
        
        # 检查并裁剪无效索引
        invalid_mask = (target_atom_types < 0) | (target_atom_types >= n_classes)
        if invalid_mask.any():
            print(f"警告: 发现 {invalid_mask.sum().item()} 个无效原子类型，总共 {target_atom_types.size(0)} 个")
            print(f"最小值: {target_atom_types.min().item()}, 最大值: {target_atom_types.max().item()}, 类别数: {n_classes}")
            # 裁剪到有效范围
            target_atom_types = torch.clamp(target_atom_types, 0, n_classes - 1)
        
        # 计算损失
        loss = F.cross_entropy(pred_atom_types, target_atom_types, reduction='none')
        
        # 按噪声水平重新缩放损失
        loss = loss / used_type_sigmas_per_atom
        return scatter(loss, batch.batch, reduce='mean').mean()

    def kld_loss(self, mu, log_var):
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)
        return kld_loss
        
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        teacher_forcing = (
            self.current_epoch <= self.hparams.teacher_forcing_max_epoch)
        outputs = self(batch, teacher_forcing, training=True)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='train')
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='val')
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='test')
        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, batch, outputs, prefix):
        # 计算所有损失并组合
        num_atom_loss = outputs['num_atom_loss']
        lattice_loss = outputs['lattice_loss']
        composition_loss = outputs['composition_loss']
        coord_loss = outputs['coord_loss']
        type_loss = outputs['type_loss']
        kld_loss = outputs['kld_loss']
        property_loss = outputs['property_loss']
        
        # 各个损失的权重
        num_atom_weight = getattr(self.hparams, 'num_atom_loss_weight', 1.0)
        lattice_weight = getattr(self.hparams, 'lattice_loss_weight', 1.0)
        composition_weight = getattr(self.hparams, 'composition_loss_weight', 1.0)
        coord_weight = getattr(self.hparams, 'coord_loss_weight', 1.0)
        type_weight = getattr(self.hparams, 'type_loss_weight', 1.0)
        kld_weight = getattr(self.hparams, 'kld_loss_weight', 1.0)
        property_weight = getattr(self.hparams, 'property_loss_weight', 0.0)
        
        # 总损失计算
        loss = (
            num_atom_weight * num_atom_loss +
            lattice_weight * lattice_loss +
            composition_weight * composition_loss +
            coord_weight * coord_loss +
            type_weight * type_loss +
            kld_weight * kld_loss
        )
        
        # 如果启用属性预测，添加属性损失
        if self.predict_property and property_weight > 0:
            loss = loss + property_weight * property_loss
            
        # 创建日志字典
        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_num_atom_loss': num_atom_loss,
            f'{prefix}_lattice_loss': lattice_loss,
            f'{prefix}_composition_loss': composition_loss,
            f'{prefix}_coord_loss': coord_loss,
            f'{prefix}_type_loss': type_loss,
            f'{prefix}_kld_loss': kld_loss,
        }
        
        # 如果启用属性预测，添加属性损失到日志
        if self.predict_property and property_weight > 0:
            log_dict[f'{prefix}_property_loss'] = property_loss
            log_dict[f'{prefix}_energy_loss'] = outputs['energy_loss']
            log_dict[f'{prefix}_target_loss'] = outputs['target_loss']
            
            # 如果有标准化器，添加反标准化后的MAE指标
            if hasattr(batch, 'y') and batch.y is not None and self.scaler is not None and self.energy_scaler is not None:
                # 调整目标张量形状
                if batch.y.dim() == 3 and batch.y.size(2) == 2:
                    target = batch.y.squeeze(1)
                else:
                    if batch.y.dim() == 2 and batch.y.size(1) == 2:
                        target = batch.y
                    else:
                        target = batch.y.view(-1, 2)
                
                # 共享特征提取并预测属性
                z = outputs['z']
                shared_features = self.fc_property_shared(z)
                energy_pred = self.energy_head(shared_features)
                target_pred = self.target_head(shared_features)
                
                # 获取目标值
                energy_target = target[:, 0:1]
                target_target = target[:, 1:2]
                
                # 应用逆标准化计算MAE
                self.energy_scaler.match_device(energy_pred)
                self.scaler.match_device(target_pred)
                
                scaled_energy_pred = self.energy_scaler.inverse_transform(energy_pred)
                scaled_energy_target = self.energy_scaler.inverse_transform(energy_target)
                energy_mae = torch.mean(torch.abs(scaled_energy_pred - scaled_energy_target))
                
                scaled_target_pred = self.scaler.inverse_transform(target_pred)
                scaled_target_target = self.scaler.inverse_transform(target_target)
                target_mae = torch.mean(torch.abs(scaled_target_pred - scaled_target_target))
                
                log_dict[f'{prefix}_energy_mae'] = energy_mae
                log_dict[f'{prefix}_target_mae'] = target_mae
        
        return log_dict, loss