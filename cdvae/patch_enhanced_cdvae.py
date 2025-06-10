#!/usr/bin/env python3
"""
修补 enhanced_cdvae.py 中的 compute_losses 方法
"""

def patch_enhanced_cdvae_file():
    """修补 enhanced_cdvae.py 文件"""
    
    file_path = "cdvae/pl_modules/enhanced_cdvae.py"
    
    # 读取原文件
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 备份原文件
    with open(file_path + ".backup", 'w') as f:
        f.write(content)
    
    # 替换有问题的行
    old_line = "        base_losses = super().compute_losses(batch, outputs, kld_weight)"
    
    new_compute_losses_method = '''        # 计算基础损失（手动调用父类的各个损失函数）
        base_losses = {}
        
        # KLD 损失
        if 'z_mu' in outputs and 'z_log_var' in outputs:
            base_losses['kld'] = self.kld_loss(outputs['z_mu'], outputs['z_log_var']) * kld_weight
        
        # 其他基础损失
        if 'pred_num_atoms' in outputs:
            base_losses['num_atom'] = self.num_atom_loss(outputs['pred_num_atoms'], batch)
        
        if 'pred_lengths_and_angles' in outputs:
            base_losses['lattice'] = self.lattice_loss(outputs['pred_lengths_and_angles'], batch)'''
    
    # 替换内容
    if old_line in content:
        content = content.replace(old_line, new_compute_losses_method)
        
        # 写回文件
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"✅ Successfully patched {file_path}")
        print(f"📄 Backup saved as {file_path}.backup")
        return True
    else:
        print(f"❌ Could not find the line to replace in {file_path}")
        return False

if __name__ == "__main__":
    patch_enhanced_cdvae_file()