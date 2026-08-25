#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vina对接集成模块 (vina.py)
功能：序列 → 3D构象 → PDBQT → Vina对接 → 结合能

【修复】所有Vina参数默认从 config.VINA_CONFIG 读取，确保配置一致性

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
        print(f"\n{'='*60}")
        print(f"【Vina错误】受体文件不存在")
        print(f"{'='*60}")
        print(f"  期望路径: {receptor_pdbqt}")
        print(f"  靶点名称: {target_name}")
        print(f"\n  可能原因:")
        print(f"    1. 阶段一未运行，受体文件未生成")
        print(f"    2. 靶点名称拼写错误")
        print(f"    3. 受体文件被删除或移动")
        print(f"\n  解决方案:")
        print(f"    1. 先运行阶段一准备受体:")
        print(f"       python run_phase1.py -t {target_name}")
        print(f"    2. 检查靶点名称是否正确")
        print(f"    3. 检查 results/{target_name}/vina/ 目录")
        print(f"{'='*60}\n")
        raise FileNotFoundError(f"受体文件不存在: {receptor_pdbqt}")
    
    if not vina_config.exists():
        print(f"\n{'='*60}")
        print(f"【Vina错误】配置文件不存在")
        print(f"{'='*60}")
        print(f"  期望路径: {vina_config}")
        print(f"  靶点名称: {target_name}")
        print(f"\n  可能原因:")
        print(f"    1. 阶段一未运行，配置文件未生成")
        print(f"    2. 配置文件被删除")
        print(f"\n  解决方案:")
        print(f"    1. 先运行阶段一准备受体:")
        print(f"       python run_phase1.py -t {target_name}")
        print(f"    2. 检查 results/{target_name}/vina/ 目录")
        print(f"{'='*60}\n")
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
                           n_cpu: Optional[int] = None,
                           exhaustiveness: Optional[int] = None,
                           num_modes: Optional[int] = None,
                           energy_range: Optional[int] = None,
                           verbose: bool = True,
                           sequence: str = "") -> VinaResult:
    """
    运行Vina对接，实时输出进度
    
    【修复】所有参数默认从 config.VINA_CONFIG 读取
    
    Args:
        ligand_pdbqt: 配体PDBQT路径
        receptor_pdbqt: 受体PDBQT路径
        vina_config: Vina配置文件路径
        output_pdbqt: 输出文件路径
        timeout: 超时时间（秒）
        n_cpu: 使用的CPU核心数（默认从config读取）
        exhaustiveness: 搜索详尽度（默认从config读取）
        num_modes: 输出构象数量（默认从config读取）
        energy_range: 能量范围（默认从config读取）
        verbose: 是否打印详细输出
        sequence: 序列信息（用于日志）
    
    Returns:
        VinaResult
    """
    ligand_pdbqt = Path(ligand_pdbqt)
    receptor_pdbqt = Path(receptor_pdbqt)
    vina_config = Path(vina_config)
    
    # 【修复】从 config 读取默认值
    if n_cpu is None:
        n_cpu = config.VINA_CONFIG.get("cpu", 4)
    if exhaustiveness is None:
        exhaustiveness = config.VINA_CONFIG.get("exhaustiveness", 4)
    if num_modes is None:
        num_modes = config.VINA_CONFIG.get("num_modes", 9)
    if energy_range is None:
        energy_range = config.VINA_CONFIG.get("energy_range", 4)
    
    # 检查文件
    if not ligand_pdbqt.exists():
        print(f"\n{'='*60}")
        print(f"【Vina错误】配体文件不存在")
        print(f"{'='*60}")
        print(f"  期望路径: {ligand_pdbqt}")
        print(f"\n  可能原因:")
        print(f"    1. 配体生成失败，PDBQT文件未创建")
        print(f"    2. 配体生成后文件被删除")
        print(f"    3. 临时目录权限问题")
        print(f"\n  解决方案:")
        print(f"    1. 检查 ligand_generator.py 是否正常工作")
        print(f"    2. 检查临时目录权限")
        print(f"    3. 手动检查文件是否存在: ls -la {ligand_pdbqt.parent}")
        print(f"{'='*60}\n")
        raise FileNotFoundError(f"配体文件不存在: {ligand_pdbqt}")
    
    if not receptor_pdbqt.exists():
        print(f"\n{'='*60}")
        print(f"【Vina错误】受体文件不存在")
        print(f"{'='*60}")
        print(f"  期望路径: {receptor_pdbqt}")
        print(f"\n  可能原因:")
        print(f"    1. 阶段一未运行")
        print(f"    2. 受体文件被删除")
        print(f"\n  解决方案:")
        print(f"    运行: python run_phase1.py -t {target_name}")
        print(f"{'='*60}\n")
        raise FileNotFoundError(f"受体文件不存在: {receptor_pdbqt}")
    
    if not vina_config.exists():
        print(f"\n{'='*60}")
        print(f"【Vina错误】配置文件不存在")
        print(f"{'='*60}")
        print(f"  期望路径: {vina_config}")
        print(f"\n  解决方案:")
        print(f"    运行: python run_phase1.py -t {target_name}")
        print(f"{'='*60}\n")
        raise FileNotFoundError(f"配置文件不存在: {vina_config}")
    
    # 输出文件
    if output_pdbqt is None:
        output_pdbqt = ligand_pdbqt.parent / f"{ligand_pdbqt.stem}_docked.pdbqt"
    else:
        output_pdbqt = Path(output_pdbqt)
    
    # Vina路径
    vina_exe = config.TOOLS.get("vina", "vina")
    
    # 构建命令（使用config中的参数）
    cmd = [
        vina_exe,
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--config", str(vina_config),
        "--out", str(output_pdbqt),
        "--cpu", str(n_cpu),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
        "--energy_range", str(energy_range)
    ]
    
    if verbose:
        seq_info = f"[{sequence}] " if sequence else ""
        print(f"\n{seq_info}启动Vina对接 (CPU={n_cpu}, exhaustiveness={exhaustiveness})...")
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
        
        # 【修复】处理Vina返回码
        # returncode = 0: 成功
        # returncode = -2: 警告（如无法利用所有CPU），但对接可能成功
        # returncode < 0: 其他错误
        if returncode < 0 and returncode != -2:
            error_msg = "\n".join(stdout_lines[-10:]) if stdout_lines else "无错误输出"
            print(f"\n{'='*60}")
            print(f"【Vina错误】Vina进程返回错误码")
            print(f"{'='*60}")
            print(f"  返回码: {returncode}")
            print(f"  序列: {sequence}")
            print(f"  配体: {ligand_pdbqt}")
            print(f"  受体: {receptor_pdbqt}")
            print(f"\n  最后10行输出:")
            for i, line in enumerate(stdout_lines[-10:], 1):
                print(f"    {i}: {line}")
            print(f"\n  可能原因:")
            print(f"    1. Vina命令参数错误")
            print(f"    2. 配体/受体文件格式不兼容")
            print(f"    3. 对接盒子配置错误（中心/大小）")
            print(f"    4. 内存不足")
            print(f"\n  解决方案:")
            print(f"    1. 检查配体PDBQT格式: obabel {ligand_pdbqt} -opdbqt")
            print(f"    2. 检查受体PDBQT格式: obabel {receptor_pdbqt} -opdbqt")
            print(f"    3. 检查Vina配置文件的盒子参数")
            print(f"    4. 尝试减少exhaustiveness参数")
            print(f"{'='*60}\n")
            raise RuntimeError(f"Vina返回错误码 {returncode}: {error_msg}")
        
        # 处理警告（returncode = -2）
        if returncode == -2:
            print(f"\n【Vina警告】返回码-2（警告，但可能成功）")
            print(f"  这通常表示：无法利用所有CPU（exhaustiveness < cpu）")
            print(f"  将继续尝试解析输出...")
        
        # 解析结合能（支持Vina 1.1.2和1.2+格式）
        binding_energy = None
        stdout_text = '\n'.join(stdout_lines)
        
        for line in stdout_lines:
            # Vina 1.2+ 格式: REMARK VINA RESULT: -8.5 0.000 0.000
            if 'REMARK VINA RESULT:' in line:
                try:
                    parts = line.split()
                    binding_energy = float(parts[3])
                    break
                except (ValueError, IndexError):
                    continue
            # Vina 1.1.2 格式:    1         -2.6      0.000      0.000
            # 解析模式行（以数字开头，包含affinity列）
            stripped = line.strip()
            if stripped and stripped[0].isdigit() and 'affinity' not in line.lower():
                parts = stripped.split()
                if len(parts) >= 2:
                    try:
                        # 第一列是mode编号，第二列是affinity
                        mode_num = int(parts[0])
                        energy = float(parts[1])
                        # 只取第一个模式（最佳结合能）
                        if mode_num == 1:
                            binding_energy = energy
                            break
                    except (ValueError, IndexError):
                        continue
        
        if binding_energy is None:
            print(f"\n{'='*60}")
            print(f"【Vina错误】无法从输出中解析结合能")
            print(f"{'='*60}")
            print(f"  序列: {sequence}")
            print(f"  配体: {ligand_pdbqt}")
            print(f"  受体: {receptor_pdbqt}")
            print(f"\n  Vina输出内容（前800字符）:")
            print(f"  {stdout_text[:800]}")
            print(f"\n  可能原因:")
            print(f"    1. Vina输出格式异常（版本不兼容？）")
            print(f"    2. Vina未能成功对接（配体/受体问题）")
            print(f"    3. Vina输出被截断")
            print(f"\n  解决方案:")
            print(f"    1. 检查Vina版本: vina --version")
            print(f"    2. 手动运行Vina查看完整输出")
            print(f"    3. 检查配体和受体文件是否有效")
            print(f"{'='*60}\n")
            raise RuntimeError("无法从Vina输出解析结合能，可能是Vina执行失败或输出格式异常")
        
        # 【关键修复】检查结合能是否为正值（表示对接失败）
        if binding_energy > 0:
            print(f"\n{'='*60}")
            print(f"【Vina警告】对接产生正值结合能 ({binding_energy:.2f} kcal/mol)")
            print(f"{'='*60}")
            print(f"  序列: {sequence}")
            print(f"  配体: {ligand_pdbqt}")
            print(f"  受体: {receptor_pdbqt}")
            print(f"\n  可能原因:")
            print(f"    1. 分子构象生成失败（如'Can't kekulize mol'错误）")
            print(f"    2. 对接盒子设置不正确（中心/大小）")
            print(f"    3. 配体与受体有严重冲突")
            print(f"    4. 分子结构不合理（如键长/键角异常）")
            print(f"\n  解决方案:")
            print(f"    1. 检查ligand_generator.py的分子生成逻辑")
            print(f"    2. 检查Vina配置文件的盒子参数")
            print(f"    3. 使用OpenBabel验证分子: obabel {ligand_pdbqt} -O test.pdb")
            print(f"    4. 手动检查生成的PDBQT文件")
            print(f"{'='*60}\n")
            # 返回失败结果，让上层决定是否继续
            return VinaResult(binding_energy, output_pdbqt, False, 
                            f"正值结合能 ({binding_energy:.2f})，对接失败", sequence)
        
        if verbose:
            seq_info = f"[{sequence}] " if sequence else ""
            print(f"{seq_info}✓ 结合能: {binding_energy:.4f} kcal/mol")
        
        return VinaResult(binding_energy, output_pdbqt, True, "", sequence)
        
    except subprocess.TimeoutExpired:
        if proc is not None:
            proc.kill()
            proc.communicate()
        print(f"\n{'='*60}")
        print(f"【Vina错误】对接超时")
        print(f"{'='*60}")
        print(f"  超时时间: {timeout} 秒")
        print(f"  序列: {sequence}")
        print(f"  配体: {ligand_pdbqt}")
        print(f"  受体: {receptor_pdbqt}")
        print(f"\n  可能原因:")
        print(f"    1. 分子过大，对接计算量过大")
        print(f"    2. exhaustiveness设置过高")
        print(f"    3. CPU资源不足")
        print(f"    4. Vina进程死锁（OpenMP问题）")
        print(f"\n  解决方案:")
        print(f"    1. 增加超时时间: timeout={timeout*2}")
        print(f"    2. 降低exhaustiveness（默认{exhaustiveness}，尝试减半）")
        print(f"    3. 使用更少的CPU核心: n_cpu=1")
        print(f"    4. 检查系统负载: top 或 htop")
        print(f"{'='*60}\n")
        raise RuntimeError(f"对接超时（{timeout}秒）")
    
    except Exception as e:
        if 'proc' in locals() and proc is not None:
            try:
                proc.kill()
                proc.communicate()
            except:
                pass
        print(f"\n{'='*60}")
        print(f"【Vina错误】执行异常")
        print(f"{'='*60}")
        print(f"  异常类型: {type(e).__name__}")
        print(f"  异常信息: {e}")
        print(f"  序列: {sequence}")
        print(f"  配体: {ligand_pdbqt}")
        print(f"  受体: {receptor_pdbqt}")
        print(f"\n  可能原因:")
        print(f"    1. Vina可执行文件不存在或权限不足")
        print(f"    2. 系统资源不足（内存/磁盘）")
        print(f"    3. 输入文件损坏")
        print(f"\n  解决方案:")
        print(f"    1. 检查Vina安装: which vina")
        print(f"    2. 检查磁盘空间: df -h")
        print(f"    3. 检查内存使用: free -h")
        print(f"    4. 验证输入文件完整性")
        print(f"{'='*60}\n")
        raise RuntimeError(f"Vina执行异常: {e}") from e


def vina_dock(sequence: str,
              target_name: str,
              crosslinker: Optional[str] = None,
              crosslinker_positions: Optional[list] = None,
              output_dir: Optional[Path] = None,
              timeout: int = 300,
              n_cpu: Optional[int] = None,
              exhaustiveness: Optional[int] = None,
              verbose: bool = False) -> float:
    """
    主函数：序列 → Vina结合能
    
    【修复】所有参数默认从 config 读取
    
    Args:
        sequence: 完整氨基酸序列
        target_name: 靶点名称
        crosslinker: 交联剂类型（默认从config读取）
        crosslinker_positions: Cys连接位置（默认从config读取）
        output_dir: 输出目录（可选）
        timeout: 超时时间（秒）
        n_cpu: 使用的CPU核心数（默认从config读取）
        exhaustiveness: 搜索详尽度（默认从config读取）
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
    
    # 【修复】从 config 读取所有默认值
    if crosslinker is None:
        crosslinker = config.CROSSLINKER
    if crosslinker_positions is None:
        crosslinker_positions = config.CROSSLINKER_POSITIONS
    if n_cpu is None:
        n_cpu = config.VINA_CONFIG.get("cpu", 4)
    if exhaustiveness is None:
        exhaustiveness = config.VINA_CONFIG.get("exhaustiveness", 4)
    
    if verbose:
        print(f"  配置: CPU={n_cpu}, exhaustiveness={exhaustiveness}")
    
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
            exhaustiveness=exhaustiveness,
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
        # 【关键】不要返回0，而是抛出异常，避免调用者误用
        raise RuntimeError(f"Vina对接失败: {e}") from e


def dock_single_worker(args):
    """
    单进程工作函数（用于多进程并行）
    
    Args:
        args: (sequence, target_name, crosslinker, crosslinker_positions, output_dir, timeout, n_cpu, exhaustiveness)
    
    Returns:
        (sequence, energy)
    """
    sequence, target_name, crosslinker, crosslinker_positions, output_dir, timeout, n_cpu, exhaustiveness = args
    
    energy = vina_dock(
        sequence=sequence,
        target_name=target_name,
        crosslinker=crosslinker,
        crosslinker_positions=crosslinker_positions,
        output_dir=output_dir,
        timeout=timeout,
        n_cpu=n_cpu,
        exhaustiveness=exhaustiveness,
        verbose=True  # 每个进程都输出进度
    )
    
    return sequence, energy


def batch_vina_dock_parallel(sequences: List[str],
                             target_name: str,
                             output_file: Optional[Path] = None,
                             n_workers: Optional[int] = None,
                             n_cpu_per_worker: Optional[int] = None,
                             timeout: int = 300,
                             verbose: bool = True) -> Dict[str, float]:
    """
    批量对接多个序列（多进程并行）
    
    【修复】所有参数默认从 config 读取
    
    Args:
        sequences: 序列列表
        target_name: 靶点名称
        output_file: 结果保存路径（可选）
        n_workers: 并行工作进程数（默认从config读取）
        n_cpu_per_worker: 每个Vina进程使用的CPU数（默认从config读取）
        timeout: 每个分子对接超时时间（秒）
        verbose: 是否打印详细信息
    
    Returns:
        {序列: 结合能} 字典
    """
    import multiprocessing
    
    # 【修复】从 config 读取默认值
    if n_workers is None:
        n_workers = config.PARALLEL_VINA_CONFIG.get("num_workers", multiprocessing.cpu_count())
    if n_cpu_per_worker is None:
        # 【关键修复】使用 VINA_CONFIG["cpu"] 而不是 PARALLEL_VINA_CONFIG["cpu_per_worker"]
        # 确保与单分子对接的CPU设置一致
        n_cpu_per_worker = config.VINA_CONFIG.get("cpu", 4)
    
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
    exhaustiveness = config.VINA_CONFIG.get("exhaustiveness", 4)
    output_dir = config.BASE_DIR / "temp" / "vina_dock"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    args_list = [
        (seq, target_name, crosslinker, crosslinker_positions, output_dir, timeout, n_cpu_per_worker, exhaustiveness)
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
                    n_cpu_per_worker: Optional[int] = None,
                    timeout: int = 300,
                    verbose: bool = False) -> Dict[str, float]:
    """
    批量对接（支持串行和并行模式）
    
    【修复】所有参数默认从 config 读取
    
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
    
    # 【修复】从 config 读取默认值
    default_cpu = config.VINA_CONFIG.get("cpu", 4)
    default_exhaustiveness = config.VINA_CONFIG.get("exhaustiveness", 4)
    
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
    parser.add_argument('--cpu', type=int, default=default_cpu,
                       help=f'单进程使用的CPU数（默认: {default_cpu}）')
    parser.add_argument('--exhaustiveness', type=int, default=default_exhaustiveness,
                       help=f'搜索详尽度（默认: {default_exhaustiveness}）')
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
            exhaustiveness=args.exhaustiveness,
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
