#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCTS Simulation模块 V2 (simulation.py)
支持三层决策的模拟引擎

关键改进：
- 使用MCTS确定的交联剂/拓扑，不再随机选择
- 每个状态（序列+交联剂+二硫键）对应唯一的分子构象
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
        
        self.temp_dir = Path(tempfile.gettempdir()) / "mcts_simulation"
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
            seed: 随机种子
        
        Returns:
            预测结合能（kcal/mol）
        
        Raises:
            RuntimeError: 如果EGNN模型未加载或预测失败
        """
        print(f"[Simulation] 开始EGNN预测: {state}")
        
        # 如果没有EGNN模型，尝试创建
        if self.egnn_model is None:
            print("[Simulation] EGNN模型未加载，尝试创建...")
            self.egnn_model = create_egnn_predictor()
        
        # 如果仍然没有模型（模型不存在或加载失败），返回错误
        if self.egnn_model is None:
            print("[Simulation] 【严重错误】EGNN模型创建失败！")
            print("[Simulation]         可能原因：")
            print("[Simulation]         1. 模型文件不存在: egnn/models/best_model.pt")
            print("[Simulation]         2. PyTorch未安装")
            print("[Simulation]         解决方案：")
            print("[Simulation]         1. 运行 EGNN_1.py 准备数据")
            print("[Simulation]         2. 运行 EGNN_23.py 训练模型")
            raise RuntimeError(
                "EGNN模型未加载。请先运行EGNN训练（EGNN_1.py + EGNN_23.py），"
                "或检查模型路径是否正确。"
            )
        
        print(f"[Simulation] ✓ EGNN模型已加载")
        
        try:
            # 获取交联剂配置
            crosslinker, positions = self.get_crosslinker_config(state)
            print(f"[Simulation] 交联剂配置: {crosslinker}, positions={positions}")
            
            # 生成分子
            print(f"[Simulation] 生成分子: {state.sequence[:20]}...")
            pdbqt_path = generate_ligand(
                sequence=state.sequence,
                crosslinker=crosslinker,
                crosslinker_positions=positions if positions else None,
                output_dir=self.temp_dir,
                random_seed=seed
            )
            print(f"[Simulation] ✓ 分子生成成功: {pdbqt_path}")
            
            # EGNN预测
            print(f"[Simulation] 开始EGNN预测...")
            if isinstance(self.egnn_model, EGNNPredictor):
                # 使用真正的EGNN预测器
                energy = self.egnn_model.predict(pdbqt_path)
            else:
                # 兼容旧的callable接口
                energy = self.egnn_model(pdbqt_path)
            
            # 检查是否为mock值（随机数范围-12到-5）
            if -12 < energy < -5:
                import random
                # 检查是否可能是随机数（通过比较多次调用）
                test_energy = self.egnn_model.predict(pdbqt_path)
                if abs(energy - test_energy) > 0.1:  # 如果两次预测差异大，可能是随机的
                    print(f"[Simulation] 【警告】EGNN预测结果可能是随机数！")
                    print(f"[Simulation]         第一次: {energy:.4f}, 第二次: {test_energy:.4f}")
                    print(f"[Simulation]         请检查egnn_predictor.py是否正确实现")
            
            print(f"[Simulation] ✓ EGNN预测完成: energy={energy:.4f}")
            return energy
            
        except Exception as e:
            print(f"[Simulation] 【错误】EGNN预测失败: {e}")
            print(f"[Simulation]         序列: {state.sequence}")
            print(f"[Simulation]         交联剂: {state.crosslinker}")
            raise  # 重新抛出异常，不返回默认值
    
    def simulate(self, node: MCTSNode, verbose: bool = False) -> SimulationResult:
        """
        模拟节点
        
        使用MCTS确定的完整状态进行评估
        
        Args:
            node: MCTS节点
            verbose: 是否打印详细信息
        
        Returns:
            SimulationResult
        """
        state = node.state
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Simulation: {state}")
            print(f"{'='*60}")
        
        # 检查状态是否完整
        if not state.is_terminal:
            if verbose:
                print(f"警告: 状态不完整，使用启发式评估")
            # 对于非终端节点，使用启发式分数
            return SimulationResult(
                score=0.5,
                raw_energy=-7.0,
                details={'heuristic': True}
            )
        
        # 运行多次模拟（不同随机种子）
        scores = []
        for i in range(self.n_simulations):
            energy = self.predict_with_egnn(state, seed=42 + i)
            scores.append(energy)
        
        # 聚合分数（取平均）
        raw_energy = sum(scores) / len(scores) if scores else 0.0
        
        # 归一化
        normalized_score = self.normalize_score(raw_energy)
        
        if verbose:
            print(f"\n聚合结果:")
            print(f"  原始结合能: {raw_energy:.4f} kcal/mol")
            print(f"  归一化分数: {normalized_score:.4f}")
            print(f"{'='*60}\n")
        
        return SimulationResult(
            score=normalized_score,
            raw_energy=raw_energy,
            details={
                'sequence': state.sequence,
                'crosslinker': state.crosslinker,
                'disulfide_bonds': state.disulfide_bonds,
                'n_simulations': len(scores),
                'individual_scores': scores
            }
        )
    
    def normalize_score(self, energy: float) -> float:
        """
        将结合能归一化到[0, 1]
        
        假设：
        - -15 kcal/mol 为最好 → 1.0
        - 0 kcal/mol 为最差 → 0.0
        """
        best_energy = -15.0
        worst_energy = 0.0
        
        energy = max(worst_energy, min(best_energy, energy))
        normalized = (worst_energy - energy) / (worst_energy - best_energy)
        
        return max(0.0, min(1.0, normalized))


def main():
    """测试代码"""
    print("="*60)
    print("Simulation引擎V2测试")
    print("="*60)
    
    from peptide_state import PeptideState, create_root_node
    
    # 创建模拟引擎
    engine = SimulationEngine(
        target_name="1LYZ",
        egnn_model=None,  # 测试时使用随机分数
        n_simulations=1
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
