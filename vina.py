#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vina对接集成模块 (vina.py)
功能：序列 → 3D构象 → PDBQT → Vina对接 → 结合能

支持：
1. 实时对接进度输出（使用Popen实时读取）
2. 多核CPU并行（单进程多核 + 多进程并行）

输入：
- sequence: 完整氨基酸序列
- target_name: 靶点名称（用于查找受体和配置）
- crosslinker: 交联剂类型（可选，默认从config读取）

输出：
- 结合能（kcal/mol，负值表示结合越强）

集成：sim_2 + sim_3 + sim_4
"""

import os
import sys
import subprocess
import tempfile
import threading
import queue
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))

import config
from ligand_generator import generate_ligand


@dataclass
class VinaResult:
    """Vina对接结果"""
    binding_energy: float  # kcal/mol
    output_file: Optional[Path] = None
    success: bool = True
    error_message: str = ""
    sequence: str = ""  # 添加序列信息


def get_vina_paths(target_name: str) -> Dict[str, Path]:
    """
    获取Vina所需的文件路径
    
    Returns:
        {
            'receptor': 受体PDBQT路径,
            'config': Vina配置文件路径
        }
    """
    dirs = config.get_target_dirs(target_name)
    
    receptor_pdbqt = dirs["vina"] / "vina-receptor.pdbqt"
    vina_config = dirs["vina"] / "vina_config.txt"
    
    if not receptor_pdbqt.exists():
        raise FileNotFoundError(f"受体文件不存在: {receptor_pdbqt}")
    if not vina_config.exists():
        raise FileNotFoundError(f"配置文件不存在: {vina_config}")
    
    return {
        'receptor': receptor_pdbqt,
        'config': vina_config
    }


def run_vina_with_progress(ligand_pdbqt: Path,
                           receptor_pdbqt: Path,
                           vina_config: Path,
                           output_pdbqt: Optional[Path] = None,
                           timeout: int = 300,
                           n_cpu: int = 1,
                           verbose: bool = True,
                           sequence: str = "") -> VinaResult:
    """
    运行Vina对接，实时输出进度
    
    Args:
        ligand_pdbqt: 配体PDBQT路径
        receptor_pdbqt: 受体PDBQT路径
        vina_config: Vina配置文件路径
        output_pdbqt: 输出文件路径
        timeout: 超时时间（秒）
        n_cpu: 使用的CPU核心数
        verbose: 是否打印详细输出
        sequence: 序列信息（用于日志）
    
    Returns:
        VinaResult
    """
    ligand_pdbqt = Path(ligand_pdbqt)
    receptor_pdbqt = Path(receptor_pdbqt)
    vina_config = Path(vina_config)
    
    # 检查文件
    if not ligand_pdbqt.exists():
        return VinaResult(0, None, False, f"配体文件不存在: {ligand_pdbqt}", sequence)
    if not receptor_pdbqt.exists():
        return VinaResult(0, None, False, f"受体文件不存在: {receptor_pdbqt}", sequence)
    if not vina_config.exists():
        return VinaResult(0, None, False, f"配置文件不存在: {vina_config}", sequence)
    
    # 输出文件
    if output_pdbqt is None:
        output_pdbqt = ligand_pdbqt.parent / f"{ligand_pdbqt.stem}_docked.pdbqt"
    else:
        output_pdbqt = Path(output_pdbqt)
    
    # Vina路径
    vina_exe = config.TOOLS.get("vina", "vina")
    
    # 构建命令（使用多核）
    cmd = [
        vina_exe,
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--config", str(vina_config),
        "--out", str(output_pdbqt),
        "--cpu", str(n_cpu)
    ]
    
    if verbose:
        seq_info = f"[{sequence}] " if sequence else ""
        print(f"\n{seq_info}启动Vina对接 (CPU={n_cpu})...")
        print(f"  命令: {' '.join(cmd)}")
    
    # 设置环境变量
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(n_cpu)
    
    try:
        # 使用Popen实时读取输出
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        stdout_lines = []
        
        # 实时读取输出
        if verbose:
            seq_info = f"[{sequence}] " if sequence else ""
            print(f"{seq_info}Vina输出:")
            print("-" * 50)
        
        for line in iter(proc.stdout.readline, ''):
            line = line.rstrip()
            stdout_lines.append(line)
            if verbose:
                print(f"  {line}")
        
        proc.stdout.close()
        returncode = proc.wait(timeout=timeout)
        
        if verbose:
            print("-" * 50)
        
        if returncode != 0:
            error_msg = "\n".join(stdout_lines[-10:]) if stdout_lines else "无错误输出"
            return VinaResult(0, None, False, f"Vina返回错误码 {returncode}: {error_msg}", sequence)
        
        # 解析结合能
        binding_energy = None
        stdout_text = '\n'.join(stdout_lines)
        
        for line in stdout_lines:
            if 'REMARK VINA RESULT:' in line:
                try:
                    parts = line.split()
                    binding_energy = float(parts[3])
                    break
                except (ValueError, IndexError):
                    continue
        
        if binding_energy is None:
            return VinaResult(0, None, False, "无法从Vina输出解析结合能", sequence)
        
        if verbose:
            seq_info = f"[{sequence}] " if sequence else ""
            print(f"{seq_info}✓ 结合能: {binding_energy:.4f} kcal/mol")
        
        return VinaResult(binding_energy, output_pdbqt, True, "", sequence)
        
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return VinaResult(0, None, False, f"Vina对接超时（{timeout}秒）", sequence)
    except Exception as e:
        if 'proc' in locals():
            try:
                proc.kill()
                proc.communicate()
            except:
                pass
        return VinaResult(0, None, False, f"Vina执行异常: {e}", sequence)


def vina_dock(sequence: str,
              target_name: str,
              crosslinker: Optional[str] = None,
              crosslinker_positions: Optional[list] = None,
              output_dir: Optional[Path] = None,
              timeout: int = 300,
              n_cpu: int = 1,
              verbose: bool = False) -> float:
    """
    主函数：序列 → Vina结合能
    
    Args:
        sequence: 完整氨基酸序列
        target_name: 靶点名称
        crosslinker: 交联剂类型（默认从config读取）
        crosslinker_positions: Cys连接位置（默认从config读取）
        output_dir: 输出目录（可选）
        timeout: 超时时间（秒）
        n_cpu: 使用的CPU核心数
        verbose: 是否打印详细信息
    
    Returns:
        结合能（kcal/mol），失败时返回0
    
    完整流程：
    sequence → ligand_generator → PDBQT → Vina → binding_energy
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Vina对接: {sequence}")
        print(f"靶点: {target_name}")
        print(f"{'='*60}")
    
    # 使用默认交联剂配置
    if crosslinker is None:
        crosslinker = config.CROSSLINKER
    if crosslinker_positions is None:
        crosslinker_positions = config.CROSSLINKER_POSITIONS
    
    try:
        # 1. 获取Vina路径
        vina_paths = get_vina_paths(target_name)
        
        # 2. 生成分子（ligand_generator = sim_2 + sim_3）
        if verbose:
            print(f"\n[1/2] 生成分子...")
        
        pdbqt_path = generate_ligand(
            sequence=sequence,
            crosslinker=crosslinker,
            crosslinker_positions=crosslinker_positions,
            output_dir=output_dir
        )
        
        if verbose:
            print(f"  ✓ PDBQT: {pdbqt_path}")
        
        # 3. Vina对接（sim_4）
        if verbose:
            print(f"\n[2/2] Vina对接...")
        
        result = run_vina_with_progress(
            ligand_pdbqt=pdbqt_path,
            receptor_pdbqt=vina_paths['receptor'],
            vina_config=vina_paths['config'],
            timeout=timeout,
            n_cpu=n_cpu,
            verbose=verbose,
            sequence=sequence
        )
        
        if not result.success:
            if verbose:
                print(f"  ✗ 失败: {result.error_message}")
            return 0.0
        
        if verbose:
            print(f"{'='*60}\n")
        
        return result.binding_energy
        
    except Exception as e:
        if verbose:
            print(f"  ✗ 异常: {e}")
        return 0.0


def dock_single_worker(args):
    """
    单进程工作函数（用于多进程并行）
    
    Args:
        args: (sequence, target_name, crosslinker, crosslinker_positions, output_dir, timeout, n_cpu)
    
    Returns:
        (sequence, energy)
    """
    sequence, target_name, crosslinker, crosslinker_positions, output_dir, timeout, n_cpu = args
    
    energy = vina_dock(
        sequence=sequence,
        target_name=target_name,
        crosslinker=crosslinker,
        crosslinker_positions=crosslinker_positions,
        output_dir=output_dir,
        timeout=timeout,
        n_cpu=n_cpu,
        verbose=True  # 每个进程都输出进度
    )
    
    return sequence, energy


def batch_vina_dock_parallel(sequences: List[str],
                             target_name: str,
                             output_file: Optional[Path] = None,
                             n_workers: Optional[int] = None,
                             n_cpu_per_worker: int = 1,
                             timeout: int = 300,
                             verbose: bool = True) -> Dict[str, float]:
    """
    批量对接多个序列（多进程并行）
    
    Args:
        sequences: 序列列表
        target_name: 靶点名称
        output_file: 结果保存路径（可选）
        n_workers: 并行工作进程数（默认使用CPU核心数）
        n_cpu_per_worker: 每个Vina进程使用的CPU数
        timeout: 每个分子对接超时时间（秒）
        verbose: 是否打印详细信息
    
    Returns:
        {序列: 结合能} 字典
    """
    import multiprocessing
    
    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), len(sequences))
    
    print("="*60)
    print(f"批量Vina对接（并行模式）")
    print("="*60)
    print(f"总序列数: {len(sequences)}")
    print(f"并行进程数: {n_workers}")
    print(f"每进程CPU数: {n_cpu_per_worker}")
    print(f"总CPU使用: {n_workers * n_cpu_per_worker}")
    print("="*60)
    
    # 准备参数
    crosslinker = config.CROSSLINKER
    crosslinker_positions = config.CROSSLINKER_POSITIONS
    output_dir = config.BASE_DIR / "temp" / "vina_dock"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    args_list = [
        (seq, target_name, crosslinker, crosslinker_positions, output_dir, timeout, n_cpu_per_worker)
        for seq in sequences
    ]
    
    results = {}
    completed = 0
    failed = 0
    
    # 使用进程池并行执行
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(dock_single_worker, args): args[0] for args in args_list}
        
        for future in as_completed(futures):
            sequence = futures[future]
            try:
                seq, energy = future.result()
                results[seq] = energy
                completed += 1
                if energy == 0:
                    failed += 1
            except Exception as e:
                results[sequence] = 0.0
                completed += 1
                failed += 1
                print(f"✗ [{sequence}] 异常: {e}")
            
            # 打印进度
            progress = completed / len(sequences) * 100
            print(f"\n>>> 总进度: {completed}/{len(sequences)} ({progress:.1f}%) | 成功: {completed-failed} | 失败: {failed}\n")
    
    # 保存结果
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("sequence,energy\n")
            for seq, energy in results.items():
                f.write(f"{seq},{energy}\n")
        
        print(f"\n✓ 结果保存至: {output_file}")
    
    print("="*60)
    print(f"批量对接完成: 成功 {completed-failed}/{len(sequences)}")
    print("="*60)
    
    return results


def batch_vina_dock(sequences: List[str],
                    target_name: str,
                    output_file: Optional[Path] = None,
                    parallel: bool = False,
                    n_workers: Optional[int] = None,
                    n_cpu_per_worker: int = 1,
                    timeout: int = 300,
                    verbose: bool = False) -> Dict[str, float]:
    """
    批量对接（支持串行和并行模式）
    
    Args:
        sequences: 序列列表
        target_name: 靶点名称
        output_file: 结果保存路径
        parallel: 是否使用并行模式
        n_workers: 并行工作进程数
        n_cpu_per_worker: 每个Vina进程使用的CPU数
        timeout: 超时时间
        verbose: 是否打印详细信息
    
    Returns:
        {序列: 结合能} 字典
    """
    if parallel:
        return batch_vina_dock_parallel(
            sequences, target_name, output_file,
            n_workers, n_cpu_per_worker, timeout, verbose
        )
    else:
        # 串行模式
        results = {}
        for i, seq in enumerate(sequences):
            if verbose:
                print(f"\n[{i+1}/{len(sequences)}] 处理序列: {seq}")
            
            energy = vina_dock(seq, target_name, verbose=verbose)
            results[seq] = energy
        
        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                f.write("sequence,energy\n")
                for seq, energy in results.items():
                    f.write(f"{seq},{energy}\n")
            
            if verbose:
                print(f"\n✓ 结果保存至: {output_file}")
        
        return results


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Vina对接集成')
    parser.add_argument('-s', '--sequence', type=str, default=None,
                       help='氨基酸序列（单分子模式）')
    parser.add_argument('-l', '--list', type=str, default=None,
                       help='序列列表文件（批量模式）')
    parser.add_argument('-t', '--target', type=str, required=True,
                       help='靶点名称')
    parser.add_argument('-c', '--crosslinker', type=str, default=None,
                       help='交联剂类型')
    parser.add_argument('-o', '--output', type=Path, default=None,
                       help='输出文件')
    parser.add_argument('--timeout', type=int, default=300,
                       help='超时时间（秒）')
    parser.add_argument('--cpu', type=int, default=1,
                       help='单进程使用的CPU数')
    parser.add_argument('--parallel', action='store_true',
                       help='使用并行模式')
    parser.add_argument('--workers', type=int, default=None,
                       help='并行工作进程数')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='详细输出')
    
    args = parser.parse_args()
    
    if args.sequence:
        # 单分子模式
        energy = vina_dock(
            sequence=args.sequence,
            target_name=args.target,
            crosslinker=args.crosslinker,
            timeout=args.timeout,
            n_cpu=args.cpu,
            verbose=args.verbose
        )
        print(f"\n结合能: {energy:.4f} kcal/mol")
    
    elif args.list:
        # 批量模式
        # 读取序列列表
        sequences = []
        with open(args.list, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and ',' not in line:
                    sequences.append(line)
        
        results = batch_vina_dock(
            sequences=sequences,
            target_name=args.target,
            output_file=args.output,
            parallel=args.parallel,
            n_workers=args.workers,
            n_cpu_per_worker=args.cpu,
            timeout=args.timeout,
            verbose=args.verbose
        )
        
        # 打印统计
        energies = [e for e in results.values() if e != 0]
        if energies:
            print(f"\n统计:")
            print(f"  成功: {len(energies)}/{len(sequences)}")
            print(f"  最佳结合能: {min(energies):.4f} kcal/mol")
            print(f"  平均结合能: {sum(energies)/len(energies):.4f} kcal/mol")
    
    else:
        print("错误: 请指定-s（单序列）或-l（序列列表文件）")


if __name__ == "__main__":
    main()
