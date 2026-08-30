#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配体生成器 (ligand_generator.py)
功能：序列 → 3D构象（含交联剂）→ PDBQT
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))

import config

# 导入MCTS日志模块
try:
    from mcts_logger import get_logger, log_crosslinker_debug
except ImportError:
    get_logger = None
    log_crosslinker_debug = None


# 【修复】使用项目目录下的临时文件夹，避免硬编码 /tmp
TEMP_DIR = config.BASE_DIR / "temp" / "ligand_generator"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# 氨基酸SMILES（N端游离，C端羧基）
AA_SMILES = {
    'A': '[N][C@@H](C)C(=O)O',
    'C': '[N][C@@H](CS)C(=O)O',  # Cys有硫原子S
    'D': '[N][C@@H](CC(=O)O)C(=O)O',
    'E': '[N][C@@H](CCC(=O)O)C(=O)O',
    'F': '[N][C@@H](Cc1ccccc1)C(=O)O',
    'G': '[N]CC(=O)O',
    'H': '[N][C@@H](Cc1c[nH]cn1)C(=O)O',
    'I': '[N][C@@H](C(C)CC)C(=O)O',
    'K': '[N][C@@H](CCCCN)C(=O)O',
    'L': '[N][C@@H](CC(C)C)C(=O)O',
    'M': '[N][C@@H](CCSC)C(=O)O',
    'N': '[N][C@@H](CC(=O)N)C(=O)O',
    'P': 'N1CCCC1C(=O)O',
    'Q': '[N][C@@H](CCC(=O)N)C(=O)O',
    'R': '[N][C@@H](CCCNC(=N)N)C(=O)O',
    'S': '[N][C@@H](CO)C(=O)O',
    'T': '[N][C@@H](C(C)O)C(=O)O',
    'V': '[N][C@@H](C(C)C)C(=O)O',
    'W': '[N][C@@H](Cc1c[nH]c2ccccc12)C(=O)O',
    'Y': '[N][C@@H](Cc1ccc(O)cc1)C(=O)O',
}

# 交联剂SMILES
# 【修复】TBMB使用Kekulé形式（明确指定双键），避免芳香环kekulization问题
CROSSLINKER_SMILES = {
    "TBMB": "C1=CC(CBr)=CC(CBr)=C1CBr",  # Kekulé形式，明确双键
    "TATA": "C(CS)(CS)CS",
    "TBAB": "C1=C(CBr)C=C(CBr)C(CBr)=C1CBr",  # Kekulé形式
}


def find_carboxyl_carbon(mol):
    """找到羧基碳（C端）"""
    from rdkit import Chem
    
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:  # 碳
            o_double = None
            o_single = None
            
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 8:  # 氧
                    bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                    if bond.GetBondType() == Chem.BondType.DOUBLE:
                        o_double = neighbor.GetIdx()
                    elif bond.GetBondType() == Chem.BondType.SINGLE:
                        o_single = neighbor.GetIdx()
            
            if o_double is not None and o_single is not None:
                return atom.GetIdx(), o_single
    
    return None, None


def find_amino_nitrogen(mol):
    """找到氨基氮（N端）"""
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 7:  # 氮
            neighbors = list(atom.GetNeighbors())
            carbon_count = sum(1 for n in neighbors if n.GetAtomicNum() == 6)
            
            if carbon_count >= 1 and carbon_count <= 2:
                return atom.GetIdx()
    
    return None


def build_peptide_with_rdkit(sequence: str) -> 'Chem.Mol':
    """使用RDKit构建肽链"""
    try:
        from rdkit import Chem
    except ImportError:
        raise RuntimeError("RDKit未安装")
    
    if not sequence:
        raise ValueError("序列为空")
    
    try:
        if len(sequence) == 1:
            smiles = AA_SMILES.get(sequence[0], AA_SMILES['A'])
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise RuntimeError(f"无法解析氨基酸: {sequence[0]}")
            return mol
        
        # 创建第一个氨基酸
        first_aa = sequence[0]
        first_smiles = AA_SMILES.get(first_aa, AA_SMILES['A'])
        mol = Chem.MolFromSmiles(first_smiles)
        if mol is None:
            raise RuntimeError(f"无法解析第一个氨基酸: {first_aa}")
        
        # 逐个添加氨基酸
        for i in range(1, len(sequence)):
            aa = sequence[i]
            aa_smiles = AA_SMILES.get(aa, AA_SMILES['A'])
            
            next_mol = Chem.MolFromSmiles(aa_smiles)
            if next_mol is None:
                print(f"【警告】无法解析氨基酸 {aa}，跳过")
                continue
            
            n_prev = mol.GetNumAtoms()
            
            c_atom_idx, oh_atom_idx = find_carboxyl_carbon(mol)
            n_atom_idx = find_amino_nitrogen(next_mol)
            
            if c_atom_idx is None or oh_atom_idx is None or n_atom_idx is None:
                print(f"【警告】第{i}个氨基酸：找不到连接点，简单合并")
                mol = Chem.CombineMols(mol, next_mol)
                continue
            
            # 合并分子
            combined = Chem.CombineMols(mol, next_mol)
            editable = Chem.EditableMol(combined)
            
            # 删除羟基（OH）
            editable.RemoveAtom(oh_atom_idx)
            
            # 调整氮原子索引
            if n_atom_idx > oh_atom_idx:
                n_atom_idx -= 1
            
            # 创建肽键（C-N）
            editable.AddBond(c_atom_idx, n_atom_idx + n_prev - (1 if oh_atom_idx < n_prev else 0), 
                           Chem.BondType.SINGLE)
            
            mol = editable.GetMol()
        
        # Sanitize
        try:
            Chem.SanitizeMol(mol)
        except:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ 
                           Chem.SanitizeFlags.SANITIZE_KEKULIZE
            )
        
        return mol
        
    except Exception as e:
        raise RuntimeError(f"构建肽链失败: {e}")


def find_cys_sulfur_atoms(mol, sequence):
    """
    找到所有Cys的硫原子索引
    
    Args:
        mol: 肽分子
        sequence: 氨基酸序列
    
    Returns:
        List[硫原子索引]
    """
    from rdkit import Chem
    
    sulfur_indices = []
    
    # 遍历分子中的所有硫原子
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 16:  # S
            # 确认这是Cys的硫（连接在CB上）
            # Cys结构: N-CA-CB-S
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 6:  # 碳（CB）
                    sulfur_indices.append(atom.GetIdx())
                    break
    
    return sulfur_indices


def add_crosslinker(mol: 'Chem.Mol', 
                    crosslinker_type: str, 
                    cys_positions: List[int],
                    sequence: str) -> 'Chem.Mol':
    """
    添加交联剂到分子
    
    【修复】正确找到Cys的硫原子并创建C-S键
    
    Args:
        mol: 肽分子
        crosslinker_type: 交联剂类型
        cys_positions: Cys在序列中的位置列表（0-based）
        sequence: 氨基酸序列（用于找到正确的Cys）
    
    Returns:
        含交联剂的分子
    """
    from rdkit import Chem
    
    if crosslinker_type not in CROSSLINKER_SMILES:
        print(f"【警告】未知交联剂类型: {crosslinker_type}，跳过添加")
        return mol
    
    # 获取交联剂SMILES
    xlinker_smiles = CROSSLINKER_SMILES[crosslinker_type]
    xlinker_mol = Chem.MolFromSmiles(xlinker_smiles)
    
    if xlinker_mol is None:
        xlinker_mol = Chem.MolFromSmiles(xlinker_smiles, sanitize=False)
        if xlinker_mol is None:
            print(f"【警告】无法解析交联剂SMILES: {xlinker_smiles}")
            return mol
        
        try:
            Chem.SanitizeMol(xlinker_mol)
        except:
            Chem.SanitizeMol(
                xlinker_mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ 
                           Chem.SanitizeFlags.SANITIZE_KEKULIZE
            )
    
    # 【修复】找到所有Cys的硫原子
    cys_sulfur_indices = find_cys_sulfur_atoms(mol, sequence)
    
    if not cys_sulfur_indices:
        print(f"【警告】找不到Cys的硫原子")
        return mol
    
    # 根据cys_positions选择要连接的硫原子
    # cys_positions是序列位置，我们需要找到对应位置的Cys的硫原子
    selected_sulfur_indices = []
    for pos in cys_positions:
        if pos < len(sequence) and sequence[pos] == 'C':
            # 找到第pos个Cys对应的硫原子
            cys_count = 0
            for i, aa in enumerate(sequence):
                if aa == 'C':
                    if i == pos and cys_count < len(cys_sulfur_indices):
                        selected_sulfur_indices.append(cys_sulfur_indices[cys_count])
                        break
                    cys_count += 1
    
    if not selected_sulfur_indices:
        print(f"【警告】无法找到指定位置的Cys硫原子")
        return mol
    
    # 合并分子
    combo = Chem.CombineMols(mol, xlinker_mol)
    editable = Chem.EditableMol(combo)
    
    # 找到交联剂中的溴原子
    xlinker_start_idx = mol.GetNumAtoms()
    br_indices = []
    for i, atom in enumerate(xlinker_mol.GetAtoms()):
        if atom.GetAtomicNum() == 35:  # Br
            br_indices.append(xlinker_start_idx + i)
    
    # 创建C-S键（Cys的S与交联剂的C，通过删除Br）
    bonds_created = 0
    bond_details = []
    for i, s_idx in enumerate(selected_sulfur_indices[:len(br_indices)]):
        if i >= len(br_indices):
            break
        
        br_idx = br_indices[i]
        br_atom = combo.GetAtomWithIdx(br_idx)
        
        # 找到与Br相连的C（这是要与S连接的C）
        c_idx = None
        for neighbor in br_atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 6:  # C
                c_idx = neighbor.GetIdx()
                break
        
        if c_idx is not None:
            # 删除Br原子
            editable.RemoveAtom(br_idx)
            
            # 调整硫原子索引（如果Br在S之前）
            adjusted_s_idx = s_idx
            if br_idx < s_idx:
                adjusted_s_idx -= 1
            
            # 调整C原子索引
            adjusted_c_idx = c_idx
            if br_idx < c_idx:
                adjusted_c_idx -= 1
            
            # 创建C-S键
            editable.AddBond(adjusted_s_idx, adjusted_c_idx, Chem.BondType.SINGLE)
            bonds_created += 1
            
            bond_details.append({"s": adjusted_s_idx, "c": adjusted_c_idx})
            
            # 更新后续Br的索引（因为删除了一个原子）
            for j in range(i + 1, len(br_indices)):
                if br_indices[j] > br_idx:
                    br_indices[j] -= 1
    
    # 记录调试信息到日志（不打印到控制台）
    if log_crosslinker_debug:
        log_crosslinker_debug(
            cys_sulfur_indices=cys_sulfur_indices,
            selected_sulfur_indices=selected_sulfur_indices,
            br_indices=br_indices,
            bonds_created=bonds_created,
            bond_details=bond_details
        )
    
    result_mol = editable.GetMol()
    
    # Sanitize - 使用更健壮的处理方式
    sanitize_error = None
    try:
        Chem.SanitizeMol(result_mol)
        print(f"【ligand_generator】分子sanitization成功")
    except Exception as e:
        sanitize_error = str(e)
        print(f"【警告】标准sanitization失败: {e}")
        print(f"【警告】尝试跳过kekulization...")
        try:
            Chem.SanitizeMol(
                result_mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ 
                           Chem.SanitizeFlags.SANITIZE_KEKULIZE
            )
            print(f"【ligand_generator】跳过kekulization后sanitization成功")
        except Exception as e2:
            print(f"【错误】跳过kekulization后仍然失败: {e2}")
            print(f"【错误】分子可能有问题，但将继续尝试生成构象")
    
    return result_mol


def generate_3d_conformation(mol: 'Chem.Mol', random_seed: int = 42) -> 'Chem.Mol':
    """生成3D构象"""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    # 【修复】更健壮的sanitization处理
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        print(f"【警告】构象生成前sanitization失败: {e}")
        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ 
                           Chem.SanitizeFlags.SANITIZE_KEKULIZE
            )
            print(f"【警告】跳过kekulization后继续")
        except Exception as e2:
            print(f"【警告】跳过kekulization后仍然失败: {e2}")
            print(f"【警告】将继续尝试生成构象，但可能失败")
    
    # 添加氢原子
    try:
        mol = Chem.AddHs(mol)
    except Exception as e:
        print(f"【错误】添加氢原子失败: {e}")
        raise RuntimeError(f"无法为分子添加氢原子: {e}")
    
    success = False
    
    try:
        from rdkit.Chem import rdDistGeom
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = random_seed
        params.enforceChirality = False
        result = rdDistGeom.EmbedMolecule(mol, params)
        if result == 0:
            success = True
    except Exception as e:
        print(f"【警告】ETKDGv3失败: {e}")
    
    if not success:
        try:
            result = AllChem.EmbedMolecule(mol, randomSeed=random_seed, maxAttempts=100)
            if result == 0:
                success = True
        except Exception as e:
            print(f"【警告】标准EmbedMolecule失败: {e}")
    
    if not success:
        try:
            result = AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=100, randomSeed=random_seed)
            if result == 0:
                success = True
        except Exception as e:
            print(f"【警告】随机坐标EmbedMolecule失败: {e}")
    
    if not success:
        raise RuntimeError("所有构象生成方法都失败")
    
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        except:
            print("【警告】构象优化失败，使用未优化的构象")
    
    # 【修改点1】RDKit计算Gasteiger电荷
    print(f"【ligand_generator】计算Gasteiger电荷...")
    try:
        from rdkit.Chem import AllChem
        AllChem.ComputeGasteigerCharges(mol)
        print(f"【ligand_generator】✓ Gasteiger电荷计算完成")
    except Exception as e:
        print(f"【警告】Gasteiger电荷计算失败: {e}")
        print(f"【警告】将继续生成PDBQT，但电荷可能为0")
    
    return mol


def rdkit_mol_to_pdbqt(mol: 'Chem.Mol', output_path: Path) -> Path:
    """
    【关键修复】直接使用RDKit生成PDBQT格式，保留Gasteiger电荷
    不经过OpenBabel，避免电荷丢失
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # PDBQT原子类型映射
    atom_type_map = {
        1: 'HD',  # 氢（给体）
        6: 'C',   # 碳
        7: 'N',   # 氮
        8: 'OA',  # 氧（受体）
        16: 'S',  # 硫
        17: 'Cl', # 氯
        35: 'Br', # 溴
        53: 'I',  # 碘
    }
    
    # 获取分子中的原子
    atoms = mol.GetAtoms()
    conf = mol.GetConformer()
    
    pdbqt_lines = []
    pdbqt_lines.append("REMARK  Generated by RDKit with Gasteiger charges")
    pdbqt_lines.append("REMARK  " + "-" * 50)
    pdbqt_lines.append("ROOT")
    
    # 写入原子
    # 写入原子（跳过氢原子，Vina使用非极性氢）
    atom_idx = 0
    for atom in atoms:
        # 跳过氢原子（原子序数=1）
        if atom.GetAtomicNum() == 1:
            continue
        
        atom_idx += 1
        pos = conf.GetAtomPosition(atom.GetIdx())
        
        # 获取原子类型
        atomic_num = atom.GetAtomicNum()
        atom_type = atom_type_map.get(atomic_num, 'A')
        
        # 获取Gasteiger电荷
        try:
            charge = atom.GetDoubleProp('_GasteigerCharge')
        except:
            charge = 0.0
        
        # 获取原子名称
        atom_name = atom.GetSymbol()
        
        # PDBQT格式: ATOM 序号 名称 残基 链 残基序号 x y z 占据 温度因子 电荷 类型
        line = f"ATOM  {atom_idx:5d}  {atom_name:3s} UNK A   1    {pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00  0.00    {charge:+.3f} {atom_type:2s}"
        pdbqt_lines.append(line)
    
    pdbqt_lines.append("ENDROOT")
    pdbqt_lines.append("TORSDOF 0")
    
    # 写入文件
    with open(output_path, 'w') as f:
        f.write('\n'.join(pdbqt_lines))
    
    return output_path


def mol_to_pdbqt(mol: 'Chem.Mol', output_path: Path) -> Path:
    """
    将分子转换为PDBQT
    
    【修改点3】优先使用RDKit直接生成，保留电荷信息
    如果失败，回退到OpenBabel方法
    """
    from rdkit import Chem
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 【关键修复】首先尝试使用RDKit直接生成PDBQT（保留电荷）
    try:
        print(f"【ligand_generator】使用RDKit直接生成PDBQT...")
        return rdkit_mol_to_pdbqt(mol, output_path)
    except Exception as e:
        print(f"【警告】RDKit直接生成失败: {e}")
        print(f"【警告】回退到OpenBabel方法...")
    
    # 回退到OpenBabel方法
    temp_sdf = output_path.with_suffix('.temp.sdf')
    
    # 写入SDF（保留完整的分子结构信息，包括键序）
    writer = Chem.SDWriter(str(temp_sdf))
    writer.write(mol)
    writer.close()
    
    obabel_path = config.TOOLS.get("obabel", "obabel")
    
    # 【修改点2】删除obabel的-p参数，因为电荷已在RDKit中计算
    # 【修改点3】从SDF读取而不是PDB，保留键序信息
    cmd = [
        obabel_path,
        str(temp_sdf),
        "-opdbqt",
        # "-p",  # 删除：不再依赖OpenBabel计算电荷
        "-xl",
        "-O", str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            # 【修改点3】如果SDF转换失败，尝试PDB作为fallback
            print(f"【警告】SDF转换失败，尝试PDB格式: {result.stderr}")
            temp_pdb = output_path.with_suffix('.temp.pdb')
            Chem.MolToPDBFile(mol, str(temp_pdb))
            
            cmd_pdb = [
                obabel_path,
                str(temp_pdb),
                "-opdbqt",
                "-xl",
                "-O", str(output_path)
            ]
            result_pdb = subprocess.run(cmd_pdb, capture_output=True, text=True, timeout=60)
            
            if temp_pdb.exists():
                temp_pdb.unlink()
            
            if result_pdb.returncode != 0:
                raise RuntimeError(f"OpenBabel转换失败(SDF和PDB都失败): {result_pdb.stderr}")
        
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("PDBQT文件生成失败")
        
        with open(output_path, 'r') as f:
            content = f.read()
        
        root_count = content.count('\nROOT\n')
        if root_count == 0:
            print(f"【警告】配体PDBQT缺少ROOT标签")
        elif root_count > 1:
            print(f"【错误】配体PDBQT包含多个ROOT标签（{root_count}个）")
            raise RuntimeError(f"配体PDBQT格式错误：包含{root_count}个ROOT标签")
        
    finally:
        if temp_sdf.exists():
            temp_sdf.unlink()
    
    return output_path


def generate_ligand(sequence: str,
                    crosslinker: Optional[str] = None,
                    crosslinker_positions: Optional[List[int]] = None,
                    output_dir: Optional[Path] = None,
                    random_seed: int = 42) -> Path:
    """主函数：序列 → PDBQT"""
    import hashlib
    from rdkit import Chem
    
    if output_dir is None:
        output_dir = TEMP_DIR
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    seq_hash = hashlib.md5(f"{sequence}_{crosslinker}".encode()).hexdigest()[:8]
    output_pdbqt = output_dir / f"peptide_{seq_hash}.pdbqt"
    
    print(f"【ligand_generator】构建肽链: {sequence}")
    
    # 1. 构建肽链
    mol = build_peptide_with_rdkit(sequence)
    
    # 验证肽链
    smiles_check = Chem.MolToSmiles(mol)
    if '.' in smiles_check:
        print(f"【错误】肽链构建失败，存在未连接片段: {smiles_check}")
        raise RuntimeError("肽链构建失败")
    
    # 2. 添加交联剂
    if crosslinker and crosslinker in CROSSLINKER_SMILES:
        print(f"【ligand_generator】添加交联剂: {crosslinker}")
        
        if crosslinker_positions:
            mol = add_crosslinker(mol, crosslinker, crosslinker_positions, sequence)
            
            # 验证交联后的分子
            smiles_check = Chem.MolToSmiles(mol)
            if '.' in smiles_check:
                print(f"【错误】交联剂添加失败，存在未连接片段: {smiles_check}")
                raise RuntimeError("交联剂添加失败")
    
    # 3. 生成3D构象
    print(f"【ligand_generator】生成3D构象...")
    mol = generate_3d_conformation(mol, random_seed)
    
    # 4. 转换为PDBQT
    print(f"【ligand_generator】转换为PDBQT...")
    pdbqt_path = mol_to_pdbqt(mol, output_pdbqt)
    
    print(f"【ligand_generator】✓ 完成: {pdbqt_path}")
    
    return pdbqt_path


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='配体生成器')
    parser.add_argument('-s', '--sequence', type=str, required=True,
                       help='氨基酸序列')
    parser.add_argument('-c', '--crosslinker', type=str, 
                       default=config.CROSSLINKER,
                       help=f'交联剂类型（默认: {config.CROSSLINKER}）')
    parser.add_argument('-o', '--output', type=Path, default=None,
                       help='输出目录')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    try:
        pdbqt_path = generate_ligand(
            sequence=args.sequence,
            crosslinker=args.crosslinker,
            output_dir=args.output,
            random_seed=args.seed
        )
        print(f"✓ PDBQT生成成功: {pdbqt_path}")
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
