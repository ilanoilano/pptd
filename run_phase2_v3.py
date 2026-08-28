#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段二主程序 V3 - 自适应MCTS-EGNN闭环优化（正确实现版）

根据对话整理的正确流程：
1. 每轮选择f(N)个母节点（N=EGNN迭代轮次）
2. 每个母节点生成g(N)个随机填充，EGNN评估得平均亲和度
3. Softmax分配f(N+1)个名额（按平均亲和度绝对值）
4. 每个母节点扩展Top-k子节点（k=分配名额，EGNN预测选Top-k，绝不随机）
5. 收集终端节点，选Top-h(N)个Vina验证
6. 8:1:1划分，微调EGNN
7. 重复直到收敛

关键函数：
- f(N) = min(19 + (N-1)*2, 100)  # 母节点数，递增
- g(N) = max(50 - (N-1)*2, 10)   # 随机填充数，递减
- h(N) = min(40 + (N-1)*3, 200)  # Vina验证数，递增
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

import config_v3 as config

# 导入MCTS模块
from peptide_state import PeptideState, create_root_node, MCTSNode
from selection import PUCTSelector
from expansion import ExpansionEngine
from simulation import SimulationEngine
from backpropagation import BackpropagationEngine
from seq_generator import generate_full_sequence, generate_n_random_fills

# 导入Vina对接
from vina import get_vina_paths, run_vina_with_progress
from ligand_generator import generate_ligand

# 导入EGNN
try:
    from egnn_predictor import create_egnn_predictor, EGNNPredictor
    HAS_EGNN = True
except ImportError:
    HAS_EGNN = False
    print("警告: EGNN模块不可用")

# 导入日志模块
try:
    from mcts_logger import init_logger, log_progress, log_debug, close_logger
    HAS_LOGGER = True
except ImportError:
    HAS_LOGGER = False
    print("警告: 日志模块不可用")


class AdaptiveMCTSEngineV3:
    """
    自适应MCTS-EGNN闭环优化引擎 V3
    
    正确实现：
    - 全局候选池保存所有历史终端节点
    - Softmax分配名额
    - EGNN批量预测选Top-k（绝不随机）
    - f(N), g(N), h(N)动态调整
    """
    
    def __init__(self, target_name: str):
        self.target_name = target_name
        
        # 目录设置
        self.results_dir = config.RESULTS_DIR / target_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # EGNN模型
        self.egnn_model = None
        self.egnn_model_path = config.BASE_DIR / "egnn" / "models" / "best_model.pt"
        
        # 全局候选池（关键：保存所有历史终端节点）
        self.candidate_pool: Dict[str, float] = {}  # {sequence: egnn_score}
        
        # 已Vina验证的序列
        self.vina_validated: Dict[str, float] = {}  # {sequence: vina_energy}
        
        # 测试集管理
        self.test_set: Dict[str, float] = {}
        self.test_mae_history: List[float] = []
        
        # 统计信息
        self.egnn_round = 0  # EGNN迭代轮次
        self.total_mcts_iterations = 0
        
        # 引擎组件
        self.selector = PUCTSelector(c_puct=config.MCTS_CONFIG["c_puct"])
        self.backprop_engine = BackpropagationEngine(verbose=False)
        
        print(f"="*60)
        print(f"自适应MCTS引擎 V3 初始化")
        print(f"靶点: {target_name}")
        print(f"="*60)
        
        # 初始化日志
        if HAS_LOGGER:
            init_logger(target_name)
            log_debug("engine", "自适应MCTS引擎 V3 初始化", {"target": target_name})
    
    # =================================================================
    # 动态参数函数 f(N), g(N), h(N)
    # =================================================================
    
    def f_n(self, n: int) -> int:
        """选取母节点数量（随EGNN轮次递增）"""
        return min(19 + (n - 1) * 2, 100)
    
    def g_n(self, n: int) -> int:
        """每个母节点的随机填充数（随EGNN轮次递减）"""
        return max(50 - (n - 1) * 2, 10)
    
    def h_n(self, n: int) -> int:
        """Vina验证数量（随EGNN轮次递增）"""
        return min(40 + (n - 1) * 3, 200)
    
    # =================================================================
    # EGNN模型管理
    # =================================================================
    
    def load_egnn_model(self) -> bool:
        """加载EGNN模型"""
        if not HAS_EGNN:
            print("错误: EGNN模块不可用")
            return False
        
        if not self.egnn_model_path.exists():
            print(f"错误: EGNN模型不存在: {self.egnn_model_path}")
            return False
        
        try:
            self.egnn_model = create_egnn_predictor()
            print(f"✓ EGNN模型加载成功")
            return True
        except Exception as e:
            print(f"✗ EGNN模型加载失败: {e}")
            return False
    
    def predict_with_egnn(self, sequence: str) -> float:
        """使用EGNN预测单个序列的亲和度"""
        if self.egnn_model is None:
            raise RuntimeError("EGNN模型未加载")
        
        from peptide_state import PeptideState
        state = PeptideState(sequence=sequence, crosslinker=config.CROSSLINKER)
        
        # 生成PDBQT并预测
        pdbqt_path = generate_ligand(
            sequence=state.sequence,
            crosslinker=state.crosslinker or config.CROSSLINKER,
            crosslinker_positions=config.CROSSLINKER_POSITIONS
        )
        
        return self.egnn_model.predict(pdbqt_path)
    
    def batch_predict_with_egnn(self, sequences: List[str]) -> List[float]:
        """批量预测多个序列的亲和度"""
        energies = []
        for seq in sequences:
            try:
                energy = self.predict_with_egnn(seq)
                energies.append(energy)
            except Exception as e:
                print(f"  预测失败 {seq}: {e}")
                energies.append(0.0)  # 失败时返回0
        return energies
    
    # =================================================================
    # 冷启动
    # =================================================================
    
    def cold_start(self, n_sequences: int = 1500) -> bool:
        """
        冷启动：生成初始数据并训练EGNN
        
        Args:
            n_sequences: 初始序列数量（默认1500）
        """
        print("\n" + "="*60)
        print("冷启动：生成初始数据")
        print("="*60)
        
        # 步骤1: 生成随机序列
        print(f"\n[1/4] 生成{n_sequences}个随机序列...")
        # 使用 config.PEPTIDE_TEMPLATE 作为模板
        sequences = generate_n_random_fills(config.PEPTIDE_TEMPLATE, n_sequences)
        print(f"  ✓ 生成完成")
        
        # 步骤2: Vina对接
        print(f"\n[2/4] Vina对接（这可能需要较长时间）...")
        vina_results = self._run_vina_batch(sequences)
        print(f"  ✓ Vina完成: {len(vina_results)}/{len(sequences)} 成功")
        
        if len(vina_results) < 100:
            print("✗ 成功对接的序列太少，冷启动失败")
            return False
        
        # 步骤3: 8:1:1划分
        print(f"\n[3/4] 划分数据集 (8:1:1)...")
        train_data, val_data, test_data = self._split_dataset(vina_results, 0.8, 0.1, 0.1)
        print(f"  ✓ 训练: {len(train_data)} / 验证: {len(val_data)} / 测试: {len(test_data)}")
        
        # 保存测试集
        self.test_set = dict(test_data)
        self.vina_validated = dict(vina_results)
        
        # 步骤4: 训练EGNN
        print(f"\n[4/4] 训练初始EGNN模型...")
        success = self._train_egnn(train_data + val_data)
        if not success:
            return False
        
        # 评估测试集
        self._evaluate_test_set()
        
        self.egnn_round = 1
        print("\n" + "="*60)
        print(f"冷启动完成！EGNN第{self.egnn_round}轮")
        print("="*60)
        return True
    
    def _run_vina_batch(self, sequences: List[str]) -> List[Tuple[str, float]]:
        """批量运行Vina对接"""
        vina_paths = get_vina_paths(self.target_name)
        results = []
        
        for i, seq in enumerate(sequences, 1):
            if i % 1 == 0:
                print(f"  进度: {i}/{len(sequences)}")
            
            try:
                pdbqt_path = generate_ligand(
                    sequence=seq,
                    crosslinker=config.CROSSLINKER,
                    crosslinker_positions=config.CROSSLINKER_POSITIONS
                )
                
                result = run_vina_with_progress(
                    ligand_pdbqt=pdbqt_path,
                    receptor_pdbqt=vina_paths['receptor'],
                    vina_config=vina_paths['config'],
                    n_cpu=config.VINA_CONFIG.get("cpu", 4)
                )
                
                if result.success and result.binding_energy < 0:
                    results.append((seq, result.binding_energy))
                    
                    # 记录Vina成功日志
                    if HAS_LOGGER:
                        log_debug("vina", f"Vina对接成功", {
                            "sequence": seq,
                            "binding_energy": result.binding_energy,
                            "progress": f"{i}/{len(sequences)}"
                        })
                else:
                    # 记录Vina失败日志
                    if HAS_LOGGER:
                        log_debug("vina", f"Vina对接失败", {
                            "sequence": seq,
                            "reason": "energy >= 0 or not success",
                            "progress": f"{i}/{len(sequences)}"
                        })
            except Exception as e:
                # 记录Vina异常日志
                if HAS_LOGGER:
                    log_debug("vina", f"Vina对接异常", {
                        "sequence": seq,
                        "error": str(e),
                        "progress": f"{i}/{len(sequences)}"
                    })
        
        # 记录Vina批次总结
        if HAS_LOGGER:
            log_debug("vina", f"Vina批次完成", {
                "total": len(sequences),
                "success": len(results),
                "success_rate": len(results) / len(sequences) if sequences else 0
            })
        
        return results
    
    def _split_dataset(self, data: List[Tuple[str, float]], 
                       train_ratio: float, val_ratio: float, test_ratio: float):
        """划分数据集"""
        import random
        random.shuffle(data)
        
        n = len(data)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_data = data[:n_train]
        val_data = data[n_train:n_train + n_val]
        test_data = data[n_train + n_val:]
        
        return train_data, val_data, test_data
    
    def _train_egnn(self, data: List[Tuple[str, float]], n_epochs: int = 100) -> bool:
        """训练EGNN模型"""
        if not HAS_EGNN:
            return False
        
        try:
            # 保存数据到文件
            sequences_file = config.BASE_DIR / "sequences.txt"
            energies_file = self.results_dir / "energies.csv"
            
            with open(sequences_file, 'w') as f:
                for seq, _ in data:
                    f.write(f"{seq}\n")
            
            with open(energies_file, 'w', newline='') as f:
                import csv
                writer = csv.writer(f)
                writer.writerow(['sequence', 'energy'])
                for seq, energy in data:
                    writer.writerow([seq, energy])
            
            # 调用训练脚本（egnn_23.py 不需要 --train 参数）
            import subprocess
            result = subprocess.run(
                [sys.executable, "egnn_23.py"],
                cwd=config.BASE_DIR,
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            if result.returncode == 0:
                print("  ✓ EGNN训练完成")
                
                # 记录训练成功日志
                if HAS_LOGGER:
                    log_debug("egnn_train", f"EGNN训练完成", {
                        "n_epochs": n_epochs,
                        "n_data": len(data),
                        "output": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout  # 最后500字符
                    })
                
                return self.load_egnn_model()
            else:
                print(f"  ✗ 训练失败: {result.stderr}")
                
                # 记录训练失败日志
                if HAS_LOGGER:
                    log_debug("egnn_train", f"EGNN训练失败", {
                        "error": result.stderr,
                        "n_data": len(data)
                    })
                
                return False
                
        except Exception as e:
            print(f"  ✗ EGNN训练失败: {e}")
            return False
    
    def _evaluate_test_set(self):
        """评估测试集性能（计算R², MAE, RMSE, Pearson r）"""
        if not self.egnn_model or not self.test_set:
            return
        
        sequences = list(self.test_set.keys())
        true_energies = list(self.test_set.values())
        
        pred_energies = self.batch_predict_with_egnn(sequences)
        
        # 计算各项指标
        mae = np.mean([abs(p - t) for p, t in zip(pred_energies, true_energies)])
        rmse = np.sqrt(np.mean([(p - t) ** 2 for p, t in zip(pred_energies, true_energies)]))
        
        # R²
        ss_res = np.sum([(p - t) ** 2 for p, t in zip(pred_energies, true_energies)])
        ss_tot = np.sum([(t - np.mean(true_energies)) ** 2 for t in true_energies])
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Pearson r
        if len(pred_energies) > 1:
            pearson_r = np.corrcoef(pred_energies, true_energies)[0, 1]
        else:
            pearson_r = 0.0
        
        self.test_mae_history.append(mae)
        
        print(f"  测试集评估:")
        print(f"    MAE:  {mae:.3f} kcal/mol")
        print(f"    RMSE: {rmse:.3f} kcal/mol")
        print(f"    R²:   {r2:.3f}")
        print(f"    Pearson r: {pearson_r:.3f}")
        
        # 记录到日志
        if HAS_LOGGER:
            log_debug("egnn_eval", f"EGNN测试集评估", {
                "egnn_round": self.egnn_round,
                "test_set_size": len(sequences),
                "mae": float(mae),
                "rmse": float(rmse),
                "r2": float(r2),
                "pearson_r": float(pearson_r)
            })
    
    # =================================================================
    # 核心：EGNN第N轮迭代
    # =================================================================
    
    def run_egnn_round(self) -> bool:
        """
        执行EGNN第N轮迭代（核心流程）
        
        流程：
        1. 选择f(N)个母节点
        2. 每个母节点生成g(N)个随机填充，EGNN评估得平均亲和度
        3. Softmax分配f(N+1)个名额
        4. 每个母节点扩展Top-k子节点（EGNN预测选Top-k，绝不随机）
        5. 收集终端节点，选Top-h(N)个Vina验证
        6. 8:1:1划分，微调EGNN
        7. 评估测试集
        
        Returns:
            是否成功完成本轮
        """
        n = self.egnn_round
        print(f"\n{'='*60}")
        print(f"EGNN第{n}轮迭代")
        print(f"{'='*60}")
        print(f"f({n}) = {self.f_n(n)} 个母节点")
        print(f"g({n}) = {self.g_n(n)} 个随机填充")
        print(f"h({n}) = {self.h_n(n)} 个Vina验证")
        
        # 步骤1: 选择f(N)个母节点
        print(f"\n[1/6] 选择{self.f_n(n)}个母节点...")
        parent_nodes = self._select_parent_nodes(self.f_n(n))
        print(f"  ✓ 选中{len(parent_nodes)}个母节点")
        
        # 步骤2: 评估每个母节点（g(N)个随机填充）
        print(f"\n[2/6] 评估母节点（每个{self.g_n(n)}个随机填充）...")
        node_scores = self._evaluate_parent_nodes(parent_nodes, self.g_n(n))
        print(f"  ✓ 评估完成")
        
        # 步骤3: Softmax分配f(N+1)个名额
        next_n = n + 1
        total_slots = self.f_n(next_n)
        print(f"\n[3/6] Softmax分配{total_slots}个名额（给下一轮f({next_n})={total_slots}）...")
        allocations = self._allocate_slots_with_softmax(node_scores, total_slots)
        print(f"  ✓ 分配完成")
        for node_key, slots in list(allocations.items())[:5]:
            print(f"    {node_key}: {slots}个名额")
        
        # 步骤4: 扩展子节点（Top-k选择，绝不随机）
        print(f"\n[4/6] 扩展子节点（EGNN预测选Top-k，绝不随机）...")
        new_candidates = self._expand_children_with_topk(parent_nodes, allocations)
        print(f"  ✓ 扩展完成，新增{len(new_candidates)}个候选")
        
        # 添加到全局候选池
        for seq, score in new_candidates.items():
            if seq not in self.candidate_pool or score < self.candidate_pool[seq]:
                self.candidate_pool[seq] = score
        
        # 步骤5: 选Top-h(N)个终端节点Vina验证
        print(f"\n[5/6] Vina验证Top-{self.h_n(n)}个候选...")
        vina_results = self._select_and_validate_top_h(self.h_n(n))
        print(f"  ✓ Vina验证完成: {len(vina_results)}个")
        
        # 步骤6: 8:1:1划分，微调EGNN
        print(f"\n[6/6] 更新EGNN模型...")
        success = self._update_egnn_with_new_data(vina_results)
        if not success:
            print("  ✗ EGNN更新失败")
            return False
        
        # 轮次+1
        self.egnn_round += 1
        
        print(f"\n{'='*60}")
        print(f"EGNN第{n}轮完成，进入第{self.egnn_round}轮")
        print(f"{'='*60}")
        return True
    
    def _select_parent_nodes(self, n_nodes: int) -> List[MCTSNode]:
        """
        选择N个母节点
        
        策略：
        - 第1轮：从根节点扩展19个子节点作为母节点
        - 后续轮次：从候选池中选择表现最好的序列作为母节点
        """
        if self.egnn_round == 1:
            # 第1轮：从根节点创建19个子节点（第一层）
            root = create_root_node()
            expansion_engine = ExpansionEngine(
                egnn_model=self.egnn_model.predict if self.egnn_model else None,
                use_egnn_prior=True
            )
            
            # 扩展第一层（19个氨基酸）
            children = expansion_engine.expand_level1_amino_acid(root, max_expansions=19)
            return list(children.values())
        else:
            # 后续轮次：从候选池中选择
            if not self.candidate_pool:
                # 候选池为空，重新从根节点开始
                return self._select_parent_nodes_from_root(n_nodes)
            
            # 按EGNN评分排序（越低越好）
            sorted_candidates = sorted(self.candidate_pool.items(), key=lambda x: x[1])
            
            # 选择Top-n_nodes作为母节点
            parent_nodes = []
            for seq, score in sorted_candidates[:n_nodes]:
                # 创建节点
                state = PeptideState(sequence=seq, crosslinker=config.CROSSLINKER)
                node = MCTSNode(state=state, prior_prob=1.0)
                parent_nodes.append(node)
            
            return parent_nodes
    
    def _select_parent_nodes_from_root(self, n_nodes: int) -> List[MCTSNode]:
        """从根节点重新选择母节点"""
        root = create_root_node()
        expansion_engine = ExpansionEngine(
            egnn_model=self.egnn_model.predict if self.egnn_model else None,
            use_egnn_prior=True
        )
        
        children = expansion_engine.expand_level1_amino_acid(root, max_expansions=n_nodes)
        return list(children.values())
    
    def _evaluate_parent_nodes(self, nodes: List[MCTSNode], n_fills: int) -> Dict[str, float]:
        """
        评估每个母节点
        
        对每个母节点：
        1. 生成n_fills个随机填充
        2. EGNN预测每个完整序列
        3. 计算平均亲和度
        
        Returns:
            {node_key: avg_energy}
        """
        node_scores = {}
        
        for i, node in enumerate(nodes, 1):
            seq = node.state.sequence
            print(f"  评估节点 {i}/{len(nodes)}: {seq[:20]}...", end=" ")
            
            # 生成n_fills个随机填充
            filled_sequences = generate_n_random_fills(seq, n_fills)
            
            # EGNN批量预测
            energies = self.batch_predict_with_egnn(filled_sequences)
            
            # 计算平均能量
            avg_energy = sum(energies) / len(energies) if energies else 0.0
            node_scores[node.state.to_key()] = avg_energy
            
            # 打印每个节点的EGNN测试平均值（包含统计信息）
            print(f"avg_energy={avg_energy:.2f}")
            if energies:
                print(f"    [统计] min={min(energies):.2f}, max={max(energies):.2f}, std={np.std(energies):.2f}")
            
            # 记录日志（每节点都记录）
            if HAS_LOGGER:
                log_progress(
                    egnn_iter=self.egnn_round,
                    mcts_depth=seq.count('_'),
                    random_count=i * n_fills,
                    sequence=seq,
                    energy=avg_energy,
                    max_iterations=config.MAX_EGNN_ITERATIONS
                )
                log_debug("mcts", f"母节点评估完成", {
                    "node_index": i,
                    "total_nodes": len(nodes),
                    "sequence": seq,
                    "avg_energy": avg_energy,
                    "min_energy": min(energies) if energies else None,
                    "max_energy": max(energies) if energies else None,
                    "std_energy": float(np.std(energies)) if energies else None,
                    "n_fills": n_fills
                })
        
        return node_scores
    
    def _allocate_slots_with_softmax(self, node_scores: Dict[str, float], 
                                     total_slots: int) -> Dict[str, int]:
        """
        Softmax分配名额
        
        公式：P_i = exp(|E_i|/T) / sum(exp(|E_j|/T))
              slots_i = P_i * total_slots
        """
        import numpy as np
        
        if not node_scores:
            return {}
        
        # 取绝对值（能量越低=绝对值越大=越好）
        abs_energies = np.array([abs(energy) for energy in node_scores.values()])
        
        # Softmax计算
        T = config.SOFTMAX_TEMPERATURE
        exp_energies = np.exp(abs_energies / T)
        probabilities = exp_energies / np.sum(exp_energies)
        
        # 分配名额
        allocations = np.floor(probabilities * total_slots).astype(int)
        
        # 处理剩余名额（按小数部分排序）
        remaining = total_slots - np.sum(allocations)
        if remaining > 0:
            fractional_parts = probabilities * total_slots - allocations
            sorted_indices = np.argsort(fractional_parts)[::-1]
            for i in range(remaining):
                idx = sorted_indices[i % len(sorted_indices)]
                allocations[idx] += 1
        
        # 构建返回字典
        result = {}
        for i, (node_key, _) in enumerate(node_scores.items()):
            result[node_key] = int(allocations[i])
        
        return result
    
    def _expand_children_with_topk(self, parent_nodes: List[MCTSNode], 
                                   allocations: Dict[str, int]) -> Dict[str, float]:
        """
        扩展子节点（Top-k选择，绝不随机）
        
        对每个母节点：
        1. 获取所有可能的子节点（19种氨基酸）
        2. EGNN预测所有子节点
        3. 选择Top-k（k=分配名额）
        """
        expansion_engine = ExpansionEngine(
            egnn_model=self.egnn_model.predict if self.egnn_model else None,
            use_egnn_prior=True
        )
        
        new_candidates = {}
        
        for node in parent_nodes:
            node_key = node.state.to_key()
            allocated_slots = allocations.get(node_key, 0)
            
            if allocated_slots <= 0:
                continue
            
            # 获取下一个可变位置
            next_pos = expansion_engine.get_next_variable_position(node.state.sequence)
            if next_pos is None:
                # 序列已完成，是终端节点
                new_candidates[node.state.sequence] = self.candidate_pool.get(
                    node.state.sequence, 0.0
                )
                continue
            
            # 获取所有可能的氨基酸（排除Cys）
            allowed_aas = expansion_engine.get_allowed_amino_acids(next_pos)
            
            # 生成所有可能的子序列
            child_sequences = []
            for aa in allowed_aas:
                child_seq = expansion_engine.fill_sequence(node.state.sequence, next_pos, aa)
                child_sequences.append((aa, child_seq))
            
            # EGNN批量预测所有子节点
            sequences_only = [seq for _, seq in child_sequences]
            energies = self.batch_predict_with_egnn(sequences_only)
            
            # 按能量排序（越低越好）
            scored_children = list(zip(child_sequences, energies))
            scored_children.sort(key=lambda x: x[1])  # 按能量升序
            
            # 选择Top-k（k=allocated_slots）
            top_k = scored_children[:allocated_slots]
            
            # 添加到候选池
            for (aa, child_seq), energy in top_k:
                new_candidates[child_seq] = energy
        
        return new_candidates
    
    def _select_and_validate_top_h(self, n_validate: int) -> List[Tuple[str, float]]:
        """
        选Top-h(N)个终端节点进行Vina验证
        
        从候选池中选择表现最好的n_validate个序列进行Vina对接
        """
        # 过滤出未验证的终端节点（序列填满）
        terminal_candidates = {}
        for seq, score in self.candidate_pool.items():
            # 检查是否已Vina验证
            if seq in self.vina_validated:
                continue
            # 检查是否是终端节点（无占位符）
            if '_' not in seq and 'x' not in seq and 'X' not in seq:
                terminal_candidates[seq] = score
        
        if not terminal_candidates:
            print("  警告: 没有未验证的终端节点")
            return []
        
        # 按EGNN评分排序（越低越好）
        sorted_candidates = sorted(terminal_candidates.items(), key=lambda x: x[1])
        
        # 选择Top-n_validate
        to_validate = sorted_candidates[:n_validate]
        sequences = [seq for seq, _ in to_validate]
        
        print(f"  选择{len(sequences)}个候选进行Vina验证")
        
        # 运行Vina
        return self._run_vina_batch(sequences)
    
    def _update_egnn_with_new_data(self, vina_results: List[Tuple[str, float]]) -> bool:
        """
        使用新的Vina数据更新EGNN
        
        1. 8:1:1划分新数据
        2. 与历史数据合并
        3. 微调EGNN
        4. 评估测试集
        """
        if not vina_results:
            print("  没有新数据，跳过更新")
            return True
        
        # 更新已验证集合
        for seq, energy in vina_results:
            self.vina_validated[seq] = energy
        
        # 8:1:1划分
        train_data, val_data, test_data = self._split_dataset(
            vina_results, 0.8, 0.1, 0.1
        )
        
        print(f"  划分: 训练{len(train_data)} / 验证{len(val_data)} / 测试{len(test_data)}")
        
        # 更新测试集
        for seq, energy in test_data:
            self.test_set[seq] = energy
        
        # 限制测试集大小（保留最新的100个）
        if len(self.test_set) > 100:
            items = list(self.test_set.items())
            self.test_set = dict(items[-100:])
        
        # 合并所有历史数据用于训练
        all_train_data = list(self.vina_validated.items())
        
        # 微调EGNN（较少epoch）
        print(f"  微调EGNN（使用{len(all_train_data)}个数据）...")
        return self._train_egnn(all_train_data, n_epochs=20)
    
    def check_convergence(self) -> Tuple[bool, str]:
        """
        检查是否收敛
        
        条件：连续5轮测试集MAE无显著改善
        """
        if len(self.test_mae_history) < 6:
            return False, "数据不足"
        
        # 检查最近5轮
        recent_mae = self.test_mae_history[-5:]
        min_improvement = 0.05  # 最小改善阈值
        
        for i in range(1, len(recent_mae)):
            improvement = recent_mae[i-1] - recent_mae[i]
            if improvement > min_improvement:
                return False, f"第{i}轮有改善({improvement:.3f})"
        
        return True, "连续5轮无显著改善"
    
    def _save_checkpoint(self):
        """保存检查点"""
        checkpoint = {
            'egnn_round': self.egnn_round,
            'candidate_pool': self.candidate_pool,
            'vina_validated': self.vina_validated,
            'test_set': self.test_set,
            'test_mae_history': self.test_mae_history,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = self.results_dir / f"checkpoint_round{self.egnn_round}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        print(f"  ✓ 检查点保存: {checkpoint_path}")


# =================================================================
# 主程序入口
# =================================================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="自适应MCTS-EGNN闭环优化 V3")
    parser.add_argument("-t", "--target", required=True, help="靶点名称（如1LYZ）")
    parser.add_argument("--max-rounds", type=int, default=100, help="最大EGNN迭代轮数")
    parser.add_argument("--cold-start-n", type=int, default=1500, help="冷启动序列数")
    
    args = parser.parse_args()
    
    engine = AdaptiveMCTSEngineV3(args.target)
    
    # 检查EGNN模型
    if engine.egnn_model_path.exists():
        print(f"\n检测到已有EGNN模型，跳过冷启动")
        engine.load_egnn_model()
        engine.egnn_round = 1
    else:
        print(f"\n未检测到EGNN模型，执行冷启动...")
        if not engine.cold_start(n_sequences=args.cold_start_n):
            print("冷启动失败！")
            sys.exit(1)
    
    # 运行自适应MCTS-EGNN闭环优化
    print(f"\n开始自适应MCTS-EGNN闭环优化")
    print(f"最大轮数: {args.max_rounds}")
    
    results = []
    
    while engine.egnn_round <= args.max_rounds:
        # 执行一轮EGNN迭代
        success = engine.run_egnn_round()
        if not success:
            print(f"\n第{engine.egnn_round}轮失败，停止")
            break
        
        # 评估测试集
        engine._evaluate_test_set()
        
        # 检查收敛
        is_converged, reason = engine.check_convergence()
        if is_converged:
            print(f"\n{'='*60}")
            print(f"收敛 detected: {reason}")
            print(f"{'='*60}")
            break
        
        # 保存检查点
        engine._save_checkpoint()
    
    # 保存最终结果
    print(f"\n{'='*60}")
    print(f"自适应MCTS-EGNN闭环优化完成")
    print(f"总轮数: {engine.egnn_round}")
    print(f"候选池大小: {len(engine.candidate_pool)}")
    print(f"Vina验证数: {len(engine.vina_validated)}")
    print(f"{'='*60}")
    
    # 关闭日志
    if HAS_LOGGER:
        close_logger()


if __name__ == "__main__":
    main()
