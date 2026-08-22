# -*- coding: utf-8 -*-
"""
蛋白口袋检测模块
功能：调用 fpocket 识别结合口袋，提取中心坐标和残基列表
输出：
  - results/[target_name]/pocket/pocket.json
  - results/[target_name]/pocket/pocket-for-esmif.pdb
  - results/[target_name]/pocket/pocket_seq.txt
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from config import get_target_dirs, TOOLS


def run_fpocket(cleaned_pdb: Path, output_dir: Path) -> Path:
    """
    运行 fpocket 检测口袋
    
    Args:
        cleaned_pdb: 清洗后的 PDB 文件路径
        output_dir: 输出目录
    
    Returns:
        fpocket 输出目录路径
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # fpocket 会在输入文件所在目录创建输出目录
    # 我们需要先复制到工作目录
    work_dir = output_dir / "fpocket_work"
    work_dir.mkdir(exist_ok=True)
    
    import shutil
    work_pdb = work_dir / cleaned_pdb.name
    shutil.copy2(cleaned_pdb, work_pdb)
    
    print(f"  运行 fpocket...")
    cmd = [TOOLS["fpocket"], "-f", str(work_pdb)]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"  fpocket 错误: {e.stderr}")
        raise RuntimeError("fpocket 执行失败")
    except FileNotFoundError:
        print("【需要安装】fpocket")
        print("请运行: conda install -c conda-forge fpocket 或从源码编译")
        raise RuntimeError("fpocket 未找到")
    
    # fpocket 输出目录: {pdb_name}_out
    fpocket_out = work_dir / f"{work_pdb.stem}_out"
    
    if not fpocket_out.exists():
        raise RuntimeError(f"fpocket 输出目录不存在: {fpocket_out}")
    
    print(f"  ✓ fpocket 完成: {fpocket_out}")
    return fpocket_out


def parse_fpocket_results(fpocket_out: Path, pdb_stem: str) -> List[Dict]:
    """
    解析 fpocket 输出结果
    
    Args:
        fpocket_out: fpocket 输出目录
        pdb_stem: PDB 文件名（不含扩展名）
    
    Returns:
        口袋列表，每个口袋包含：
        - id: 口袋编号
        - score: fpocket 评分
        - volume: 体积（Å³）
        - center: 中心坐标 (x, y, z)
        - residues: 口袋残基列表 [(chain, res_id, res_name), ...]
    """
    pockets = []
    
    # 读取 info.txt 获取评分和体积（fpocket 4.0 格式: {pdb_name}_info.txt）
    info_file = fpocket_out / f"{pdb_stem}_info.txt"
    if not info_file.exists():
        # 尝试旧格式
        info_file = fpocket_out / "info.txt"
    
    if info_file.exists():
        with open(info_file, 'r') as f:
            content = f.read()
            
        # 解析多行格式
        current_pocket = None
        for line in content.split('\n'):
            line = line.strip()
            
            # 新口袋开始: "Pocket 1 :"
            if line.startswith("Pocket") and ':' in line:
                parts = line.split()
                pocket_id = int(parts[1])
                current_pocket = {
                    "id": pocket_id,
                    "score": None,
                    "volume": None,
                    "center": None,
                    "residues": []
                }
                pockets.append(current_pocket)
            
            # 解析 Score
            elif line.startswith("Score :") and current_pocket is not None:
                try:
                    current_pocket["score"] = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
            
            # 解析 Volume
            elif line.startswith("Volume :") and current_pocket is not None:
                try:
                    current_pocket["volume"] = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
    
    # 读取每个口袋的 PDB 文件获取残基信息（fpocket 4.0 在 pockets/ 子目录）
    pockets_dir = fpocket_out / "pockets"
    for pocket in pockets:
        # 尝试新格式（pockets/ 子目录）
        pocket_pdb = pockets_dir / f"pocket{pocket['id']}_atm.pdb"
        if not pocket_pdb.exists():
            # 尝试旧格式（根目录）
            pocket_pdb = fpocket_out / f"pocket{pocket['id']}_atm.pdb"
        
        if not pocket_pdb.exists():
            print(f"    警告: 口袋 {pocket['id']} 的 PDB 文件不存在: {pocket_pdb}")
            continue
        
        residues = set()
        atoms = []
        
        with open(pocket_pdb, 'r') as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    # 解析 ATOM 记录
                    chain_id = line[21].strip()
                    res_seq = line[22:26].strip()
                    res_name = line[17:20].strip()
                    
                    residues.add((chain_id, res_seq, res_name))
                    
                    # 解析坐标
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        atoms.append((x, y, z))
                    except ValueError:
                        continue
        
        # 计算几何中心
        if atoms:
            center_x = sum(a[0] for a in atoms) / len(atoms)
            center_y = sum(a[1] for a in atoms) / len(atoms)
            center_z = sum(a[2] for a in atoms) / len(atoms)
            pocket["center"] = (round(center_x, 3), round(center_y, 3), round(center_z, 3))
        
        pocket["residues"] = sorted(list(residues))
    
    # 按评分排序（分数越低越好）
    pockets.sort(key=lambda x: x["score"])
    
    return pockets


def select_best_pocket(pockets: List[Dict], top_n: int = 1) -> List[Dict]:
    """
    选择最佳口袋
    
    策略：
    1. 优先选择 fpocket 评分最低的（最可成药）
    2. 体积适中（100-1000 Å³）
    """
    # 过滤掉没有 score 或 volume 的口袋
    valid_pockets = [
        p for p in pockets 
        if p.get("score") is not None and p.get("volume") is not None
        and 100 <= p["volume"] <= 1000
    ]
    
    if not valid_pockets:
        # 如果没有体积合格的，返回有 score 的
        valid_pockets = [p for p in pockets if p.get("score") is not None]
    
    if not valid_pockets:
        valid_pockets = pockets
    
    # 按评分排序
    valid_pockets.sort(key=lambda x: x.get("score", float('inf')))
    
    return valid_pockets[:top_n]


def create_pocket_pdb(cleaned_pdb: Path, pocket: Dict, output_pdb: Path):
    """
    创建包含口袋残基的 PDB 文件（用于 ESM-IF）
    
    提取口袋残基对应的所有原子
    """
    pocket_residues = set((r[0], r[1]) for r in pocket["residues"])  # (chain, res_id)
    
    with open(cleaned_pdb, 'r') as f_in, open(output_pdb, 'w') as f_out:
        for line in f_in:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                chain_id = line[21].strip()
                res_seq = line[22:26].strip()
                
                if (chain_id, res_seq) in pocket_residues:
                    f_out.write(line)
            elif line.startswith("TER") or line.startswith("END"):
                f_out.write(line)
    
    print(f"  ✓ 口袋 PDB: {output_pdb}")


def create_pocket_sequence(pocket: Dict, output_txt: Path):
    """
    创建口袋残基序列文件
    """
    # 按残基编号排序
    residues = sorted(pocket["residues"], key=lambda x: (x[0], int(x[1])))
    
    # 三字母转单字母
    three_to_one = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    }
    
    with open(output_txt, 'w') as f:
        f.write(f">Pocket_{pocket['id']} Score={pocket['score']:.2f} Volume={pocket['volume']:.1f}\n")
        
        seq = ""
        for chain, res_id, res_name in residues:
            aa = three_to_one.get(res_name, "X")
            seq += aa
            f.write(f"{chain}:{res_id} {res_name} {aa}\n")
        
        f.write(f"\nSequence: {seq}\n")
    
    print(f"  ✓ 口袋序列: {output_txt}")


def detect_pockets(target_name: str, top_n: int = 1) -> Dict:
    """
    主入口：检测蛋白口袋
    
    Args:
        target_name: 靶点名称
        top_n: 返回前 N 个口袋
    
    Returns:
        最佳口袋信息字典
    """
    print(f"\n{'='*60}")
    print(f"检测口袋: {target_name}")
    print(f"{'='*60}")
    
    dirs = get_target_dirs(target_name)
    cleaned_pdb = dirs["cleaned"] / "cleaned.pdb"
    
    if not cleaned_pdb.exists():
        raise FileNotFoundError(f"请先运行 pdb_cleaner.py: {cleaned_pdb}")
    
    # 1. 运行 fpocket
    fpocket_out = run_fpocket(cleaned_pdb, dirs["pocket"])
    
    # 2. 解析结果
    print(f"  解析 fpocket 结果...")
    pockets = parse_fpocket_results(fpocket_out, cleaned_pdb.stem)
    print(f"  发现 {len(pockets)} 个口袋")
    
    # 3. 选择最佳口袋
    best_pockets = select_best_pocket(pockets, top_n)
    
    if not best_pockets:
        raise RuntimeError("未找到合适的口袋")
    
    best_pocket = best_pockets[0]
    print(f"\n  最佳口袋 #{best_pocket['id']}:")
    print(f"    评分: {best_pocket['score']:.2f}" if best_pocket.get('score') else "    评分: N/A")
    print(f"    体积: {best_pocket['volume']:.1f} Å³" if best_pocket.get('volume') else "    体积: N/A")
    if best_pocket.get('center'):
        print(f"    中心: ({best_pocket['center'][0]:.3f}, {best_pocket['center'][1]:.3f}, {best_pocket['center'][2]:.3f})")
    else:
        print(f"    中心: N/A")
    print(f"    残基数: {len(best_pocket.get('residues', []))}")
    
    # 4. 保存结果
    pocket_dir = dirs["pocket"]
    
    # pocket.json
    pocket_json = pocket_dir / "pocket.json"
    with open(pocket_json, 'w') as f:
        json.dump({
            "target": target_name,
            "best_pocket": best_pocket,
            "all_pockets": pockets[:5]  # 保存前5个
        }, f, indent=2)
    print(f"  ✓ 口袋 JSON: {pocket_json}")
    
    # pocket-for-esmif.pdb
    pocket_pdb = pocket_dir / "pocket-for-esmif.pdb"
    # 使用原始清洗后的 PDB 提取口袋残基
    original_cleaned = dirs["cleaned"] / "cleaned.pdb"
    create_pocket_pdb(original_cleaned, best_pocket, pocket_pdb)
    
    # pocket_seq.txt
    pocket_seq = pocket_dir / "pocket_seq.txt"
    create_pocket_sequence(best_pocket, pocket_seq)
    
    return best_pocket


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="检测蛋白结合口袋")
    parser.add_argument("target", help="靶点名称（如 1LYZ）")
    parser.add_argument("--top-n", type=int, default=1, help="返回前 N 个口袋")
    
    args = parser.parse_args()
    
    try:
        pocket = detect_pockets(args.target, args.top_n)
        print(f"\n✓ 口袋检测完成")
        print(f"  中心坐标: ({pocket['center'][0]:.3f}, {pocket['center'][1]:.3f}, {pocket['center'][2]:.3f})")
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
