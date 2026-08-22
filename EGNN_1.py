#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EGNN数据准备模块 (EGNN_1.py)
功能：序列 + Vina分数 → 原子特征 + 坐标 → 划分数据集

输入：
- sequences.txt: 序列文件（每行一个序列）
- energies.csv: Vina结合能数据（sequence,energy）

处理逻辑：
1. 读取序列和对应Vina分数
2. 对每个序列，用RDKit生成3D构象（施加交联剂约束）
3. 提取原子特征（元素类型、杂化方式、电荷等）
4. 提取原子坐标（x, y, z）
5. 按8:1:1划分训练/验证/测试集

输出：
- egnn/raw/train_data.npz
- egnn/raw/val_data.npz
- egnn/raw/test_data.npz

每个样本包含：
- features: 原子特征矩阵 (n_atoms, n_features)
- coords: 原子坐标矩阵 (n_atoms, 3)
- energy: Vina结合能标量
"""

import os
import sys
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))

import config
from ligand_generator import build_peptide_with_rdkit, generate_3d_conformation


# 原子特征维度定义
N_FEATURES = 20  # 总特征维度

# 元素类型 one-hot (10维)
ELEMENTS = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'other']
ELEMENT_TO_IDX = {e: i for i, e in enumerate(ELEMENTS)}

# 杂化方式 one-hot (4维)
HYBRIDIZATIONS = ['SP', 'SP2', 'SP3', 'other']
HYBRID_TO_IDX = {h: i for i, h in enumerate(HYBRIDIZATIONS)}


@dataclass
class AtomFeatures:
    """原子特征"""
    element: str
    hybridization: str
    formal_charge: float
    is_hbd: bool  # 氢键供体
    is_hba: bool  # 氢键受体
    is_aromatic: bool
    degree: int


def extract_atom_features(mol) -> Tuple[np.ndarray, np.ndarray]:
    """
    从RDKit分子中提取原子特征和坐标
    
    Args:
        mol: RDKit分子对象
    
    Returns:
        features: (n_atoms, n_features) 特征矩阵
        coords: (n_atoms, 3) 坐标矩阵
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    n_atoms = mol.GetNumAtoms()
    features = np.zeros((n_atoms, N_FEATURES))
    coords = np.zeros((n_atoms, 3))
    
    # 获取构象
    if mol.GetNumConformers() == 0:
        raise ValueError("分子没有3D构象")
    
    conf = mol.GetConformer()
    
    for i, atom in enumerate(mol.GetAtoms()):
        # 1. 元素类型 one-hot (10维)
        element = atom.GetSymbol()
        elem_idx = ELEMENT_TO_IDX.get(element, ELEMENT_TO_IDX['other'])
        features[i, elem_idx] = 1.0
        
        # 2. 杂化方式 one-hot (4维)
        hybrid = atom.GetHybridization()
        hybrid_str = str(hybrid)
        if 'SP3' in hybrid_str:
            hybrid_idx = 2
        elif 'SP2' in hybrid_str:
            hybrid_idx = 1
        elif 'SP' in hybrid_str:
            hybrid_idx = 0
        else:
            hybrid_idx = 3
        features[i, 10 + hybrid_idx] = 1.0
        
        # 3. 形式电荷 (1维)
        features[i, 14] = atom.GetFormalCharge()
        
        # 4. 是否是氢键供体 (1维)
        # 简化判断：N或O上有H
        is_hbd = False
        if atom.GetSymbol() in ['N', 'O']:
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == 'H':
                    is_hbd = True
                    break
        features[i, 15] = float(is_hbd)
        
        # 5. 是否是氢键受体 (1维)
        # 简化判断：N或O
        is_hba = atom.GetSymbol() in ['N', 'O']
        features[i, 16] = float(is_hba)
        
        # 6. 是否是芳香族 (1维)
        features[i, 17] = float(atom.GetIsAromatic())
        
        # 7. 原子度数 (1维，归一化）
        features[i, 18] = atom.GetDegree() / 4.0  # 归一化到[0,1]
        
        # 8. 原子质量 (1维，归一化)
        features[i, 19] = atom.GetMass() / 100.0
        
        # 坐标
        pos = conf.GetAtomPosition(i)
        coords[i] = [pos.x, pos.y, pos.z]
    
    return features, coords


def process_sequence(sequence: str,
                     energy: float,
                     crosslinker: Optional[str] = None,
                     crosslinker_positions: Optional[List[int]] = None,
                     random_seed: int = 42) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """
    处理单个序列
    
    Args:
        sequence: 氨基酸序列
        energy: Vina结合能
        crosslinker: 交联剂类型
        crosslinker_positions: Cys连接位置
        random_seed: 随机种子
    
    Returns:
        (features, coords, energy) 或 None（如果失败）
    """
    try:
        # 1. 构建分子（含交联剂）
        from ligand_generator import add_crosslinker
        mol = build_peptide_with_rdkit(sequence)
        
        # 2. 添加交联剂（如果指定）
        if crosslinker and crosslinker_positions:
            mol = add_crosslinker(mol, crosslinker, crosslinker_positions)
        
        # 3. 生成3D构象
        mol = generate_3d_conformation(mol, random_seed)
        
        # 4. 提取特征
        features, coords = extract_atom_features(mol)
        
        return features, coords, energy
        
    except Exception as e:
        print(f"处理序列 {sequence} 失败: {e}")
        return None


def load_data(sequences_file: Path, energies_file: Path) -> List[Tuple[str, float]]:
    """
    加载序列和能量数据
    
    Args:
        sequences_file: 序列文件路径
        energies_file: 能量文件路径
    
    Returns:
        [(序列, 能量), ...]
    """
    # 读取序列
    sequences = []
    with open(sequences_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and ',' not in line:
                sequences.append(line)
    
    # 读取能量
    energies = {}
    with open(energies_file, 'r') as f:
        header = f.readline()  # 跳过表头
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                seq = parts[0]
                try:
                    energy = float(parts[1])
                    energies[seq] = energy
                except ValueError:
                    continue
    
    # 匹配序列和能量
    data = []
    for seq in sequences:
        if seq in energies:
            data.append((seq, energies[seq]))
    
    return data


def split_data(data: List, train_ratio=0.8, val_ratio=0.1) -> Tuple[List, List, List]:
    """
    划分数据集
    
    Args:
        data: 数据列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    
    Returns:
        (train_data, val_data, test_data)
    """
    random.shuffle(data)
    
    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]
    
    return train_data, val_data, test_data


def save_dataset(data: List[Tuple[np.ndarray, np.ndarray, float]],
                 output_file: Path):
    """
    保存数据集为npz格式
    
    Args:
        data: [(features, coords, energy), ...]
        output_file: 输出文件路径
    """
    if not data:
        print(f"警告: 数据集为空，跳过保存 {output_file}")
        return
    
    # 由于每个分子原子数不同，需要保存为列表
    features_list = [item[0] for item in data]
    coords_list = [item[1] for item in data]
    energies = np.array([item[2] for item in data])
    
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_file,
        features=features_list,
        coords=coords_list,
        energies=energies
    )
    
    print(f"✓ 保存数据集: {output_file} ({len(data)} 个样本)")


def main(sequences_file: Optional[Path] = None,
         energies_file: Optional[Path] = None,
         output_dir: Optional[Path] = None):
    """
    主函数
    
    Args:
        sequences_file: 序列文件路径
        energies_file: 能量文件路径
        output_dir: 输出目录
    """
    # 默认路径
    if sequences_file is None:
        sequences_file = config.BASE_DIR / "sequences.txt"
    if energies_file is None:
        energies_file = config.RESULTS_DIR / "1LYZ" / "energies.csv"
    if output_dir is None:
        output_dir = config.BASE_DIR / "egnn" / "raw"
    
    print("="*60)
    print("EGNN数据准备")
    print("="*60)
    
    # 1. 加载数据
    print(f"\n[1/4] 加载数据...")
    data = load_data(sequences_file, energies_file)
    print(f"  加载 {len(data)} 个样本")
    
    if len(data) == 0:
        print("错误: 没有有效数据")
        return
    
    # 2. 处理序列
    print(f"\n[2/4] 处理序列...")
    processed_data = []
    
    crosslinker = config.CROSSLINKER
    crosslinker_positions = config.CROSSLINKER_POSITIONS
    
    for i, (seq, energy) in enumerate(data):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  处理 {i+1}/{len(data)}: {seq}")
        
        result = process_sequence(seq, energy, crosslinker, crosslinker_positions)
        if result:
            processed_data.append(result)
    
    print(f"  成功处理 {len(processed_data)}/{len(data)} 个样本")
    
    if len(processed_data) == 0:
        print("错误: 没有成功处理的样本")
        return
    
    # 3. 划分数据集
    print(f"\n[3/4] 划分数据集...")
    train_data, val_data, test_data = split_data(processed_data)
    print(f"  训练集: {len(train_data)}")
    print(f"  验证集: {len(val_data)}")
    print(f"  测试集: {len(test_data)}")
    
    # 4. 保存数据集
    print(f"\n[4/4] 保存数据集...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_dataset(train_data, output_dir / "train_data.npz")
    save_dataset(val_data, output_dir / "val_data.npz")
    save_dataset(test_data, output_dir / "test_data.npz")
    
    print("\n" + "="*60)
    print("EGNN数据准备完成!")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='EGNN数据准备')
    parser.add_argument('-s', '--sequences', type=Path, default=None,
                       help='序列文件路径')
    parser.add_argument('-e', '--energies', type=Path, default=None,
                       help='能量文件路径')
    parser.add_argument('-o', '--output', type=Path, default=None,
                       help='输出目录')
    
    args = parser.parse_args()
    
    main(args.sequences, args.energies, args.output)
