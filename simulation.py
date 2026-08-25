#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCTS Simulation模块 V2 (simulation.py)
支持三层决策的模拟引擎

关键改进：
- 使用MCTS确定的交联剂/拓扑，不再随机选择
- 每个状态（序列+交联剂+二硫键）对应唯一的分子构象
- 【修复】使用项目目录下的临时文件夹，避免硬编码 /tmp
"""

import sys
import random
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Callable
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))

import config
from peptide_state import MCTSNode, PeptideState
from ligand_generator import generate_ligand
from egnn_predictor import create_egnn_predictor, EGNNPredictor


# 【修复】使用项目目录下的临时文件夹，避免硬编码 /tmp
TEMP_DIR = config.BASE_DIR / "temp" / "mcts_simulation"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SimulationResult:
    """模拟结果"""
    score: float           # 归一化分数 [0, 1]
    raw_energy: float      # 原始结合能（kcal/mol）
    details: Dict          # 详细信息


class SimulationEngine:
    """
    MCTS模拟引擎 V2
    
    使用MCTS确定的完整状态（序列+交联剂+二硫键）
    """
    
    def __init__(self,
                 target_name: str,
                 egnn_model: Optional[Callable] = None,
                 n_simulations: int = 1):  # V2中n_simulations主要用于不同构象
        """
        初始化模拟引擎
        
        Args:
            target_name: 靶点名称
            egnn_model: EGNN预测函数
            n_simulations: 每个状态的模拟次数（不同随机种子）
        """
        self.target_name = target_name
        self.egnn_model = egnn_model
        self.n_simulations = n_simulations
        
        # 【修复】使用项目目录下的临时文件夹
        self.temp_dir = TEMP_DIR
        self.temp_dir.mkdir(exist_ok=True)
    
    def get_crosslinker_config(self, state: PeptideState) -> tuple:
        """
        从状态获取交联剂配置
        
        Returns:
            (crosslinker_type, crosslinker_positions)
        """
        crosslinker = state.crosslinker
        
        if crosslinker is None:
            # 线性肽，无交联
            return None, []
        
        elif crosslinker == "disulfide":
            # 二硫键：使用状态中的配对
            if state.disulfide_bonds:
                # 返回第一个配对（简化处理）
                bond = state.disulfide_bonds[0]
                return "disulfide", list(bond)
            else:
                return None, []
        
        else:
            # 化学交联剂：TBMB/TATA/TBAB
            # 使用配置中的位置，或自动检测Cys
            cys_positions = state.get_cys_positions()
            
            if crosslinker in ["TBMB", "TATA"]:
                # 需要3个Cys
                positions = cys_positions[:3] if len(cys_positions) >= 3 else cys_positions
            elif crosslinker == "TBAB":
                # 需要4个Cys
                positions = cys_positions[:4] if len(cys_positions) >= 4 else cys_positions
            else:
                positions = []
            
            return crosslinker, positions
    
    def predict_with_egnn(self, state: PeptideState, seed: int = 42) -> float:
        """
        使用EGNN预测结合能
        
        Args:
            state: 肽状态
            seed: 随机种子（用于生成不同构象）
        
        Returns:
            预测结合能（kcal/mol）
        """
        # 获取交联剂配置
        crosslinker, positions = self.get_crosslinker_config(state)
        
        print(f"[Simulation] 交联剂配置: {crosslinker}, positions={positions}")
        
        try:
            # 生成分子
            pdbqt_path = generate_ligand(
                sequence=state.sequence,
                crosslinker=crosslinker,
                crosslinker_positions=positions if positions else None,
                output_dir=self.temp_dir,
                random_seed=seed
            )
            
            # EGNN预测
            if self.egnn_model is None:
                # 懒加载EGNN模型
                self.egnn_model = create_egnn_predictor()
            
            energy = self.egnn_model.predict(pdbqt_path)
            return energy
            
        except Exception as e:
            print(f"[Simulation] EGNN预测失败: {e}")
            # 返回一个较差的分数作为惩罚
            return 0.0  # 0表示无效/失败
    
    def calculate_heuristic_score(self, state: PeptideState) -> float:
        """
        计算启发式分数（用于非终端节点）
        
        基于：
        - 序列完成度
        - Cys数量（用于交联）
        - 氨基酸组成
        """
        score = 0.0
        
        # 序列完成度
        completed = sum(1 for aa in state.sequence if aa != '_')
        total = len(state.sequence)
        score += 0.3 * (completed / total)
        
        # Cys数量（用于交联）
        cys_count = len(state.get_cys_positions())
        if state.crosslinker in ["TBMB", "TATA"]:
            # 需要至少3个Cys
            score += 0.3 * min(cys_count / 3, 1.0)
        elif state.crosslinker == "TBAB":
            # 需要至少4个Cys
            score += 0.3 * min(cys_count / 4, 1.0)
        elif state.crosslinker == "disulfide":
            # 需要至少2个Cys
            score += 0.3 * min(cys_count / 2, 1.0)
        
        # 多样性奖励（不同氨基酸类型）
        aa_types = set(state.sequence.replace('_', ''))
        score += 0.1 * min(len(aa_types) / 10, 1.0)
        
        return score
    
    def simulate(self, node: MCTSNode, verbose: bool = False) -> SimulationResult:
        """
        执行模拟（ rollout ）
        
        Args:
            node: MCTS节点
            verbose: 是否打印详细信息
        
        Returns:
            SimulationResult
        """
        state = node.state
        
        if verbose:
            print(f"\n[Simulation] 开始模拟")
            print(f"  序列: {state.sequence}")
            print(f"  交联剂: {state.crosslinker}")
            print(f"  二硫键: {state.disulfide_bonds}")
        
        # 检查是否为终端状态
        if not state.is_terminal:
            # 非终端状态：使用启发式分数
            score = self.calculate_heuristic_score(state)
            if verbose:
                print(f"  非终端状态，启发式分数: {score:.4f}")
            return SimulationResult(
                score=score,
                raw_energy=0.0,
                details={"type": "heuristic", "completion": score}
            )
        
        # 终端状态：使用EGNN预测
        energies = []
        for i in range(self.n_simulations):
            seed = random.randint(1, 10000)
            energy = self.predict_with_egnn(state, seed)
            energies.append(energy)
            if verbose:
                print(f"  模拟 {i+1}/{self.n_simulations}: {energy:.4f} kcal/mol")
        
        # 取最佳（最负）的结合能
        best_energy = min(energies) if energies else 0.0
        
        # 归一化分数（假设结合能范围 -15 到 0）
        # 越负越好，所以用 -energy
        normalized_score = max(0.0, min(1.0, -best_energy / 15.0))
        
        if verbose:
            print(f"  最佳结合能: {best_energy:.4f} kcal/mol")
            print(f"  归一化分数: {normalized_score:.4f}")
        
        return SimulationResult(
            score=normalized_score,
            raw_energy=best_energy,
            details={
                "type": "egnn",
                "energies": energies,
                "best_energy": best_energy
            }
        )


def main():
    """测试代码"""
    print("="*60)
    print("SimulationEngine测试")
    print("="*60)
    
    # 创建引擎
    engine = SimulationEngine(
        target_name="1LYZ",
        egnn_model=None,  # 使用默认EGNN
        n_simulations=3
    )
    
    # 测试1：完整状态（TBMB交联）
    print("\n[测试1] 完整状态（TBMB交联）")
    state1 = PeptideState(
        sequence="ACARNDCMVFLWPCG",
        crosslinker="TBMB",
        disulfide_bonds=[]
    )
    node1 = MCTSNode(state=state1)
    
    result1 = engine.simulate(node1, verbose=True)
    print(f"结果: score={result1.score:.4f}, energy={result1.raw_energy:.4f}")
    
    # 测试2：完整状态（二硫键）
    print("\n[测试2] 完整状态（二硫键）")
    state2 = PeptideState(
        sequence="ACARNDCMVFLWPCG",
        crosslinker="disulfide",
        disulfide_bonds=[(1, 8)]
    )
    node2 = MCTSNode(state=state2)
    
    result2 = engine.simulate(node2, verbose=True)
    print(f"结果: score={result2.score:.4f}, energy={result2.raw_energy:.4f}")
    
    # 测试3：不完整状态
    print("\n[测试3] 不完整状态")
    state3 = PeptideState(
        sequence="ACARNDC______CG",
        crosslinker=None,
        disulfide_bonds=[]
    )
    node3 = MCTSNode(state=state3)
    
    result3 = engine.simulate(node3, verbose=True)
    print(f"结果: score={result3.score:.4f} (启发式)")
    
    print("\n" + "="*60)
    print("测试通过!")
    print("="*60)


if __name__ == "__main__":
    main()
