#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段二主程序 (run_phase2.py)
功能：MCTS三层决策闭环搜索 + EGNN训练/更新

三层决策：
1. 氨基酸序列填充
2. 交联剂/环化方式选择（TBMB/TATA/TBAB/disulfide/None）
3. 二硫键配对选择（如果使用disulfide）

工作流程：
1. 冷启动：生成初始序列+拓扑 → Vina对接 → 训练初始EGNN模型
2. 日常循环：MCTS迭代（Selection→Expansion→Simulation→Backpropagation）
3. 定期验证：筛选Top-K → Vina验证 → 更新数据集 → 重训EGNN
4. 输出最终候选

输入：
- config中的靶点名称、肽模板、交联剂配置
- 第一阶段准备好的Vina受体文件

输出：
- results/{target_name}/candidates.csv（候选序列、拓扑、分数）
- egnn/models/best_model.pt（训练好的EGNN模型）
"""

import os
import sys
import json
import time
import random
import pickle
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent))

import config

# =============================================================================
# 调试配置
# =============================================================================
DEBUG_MODE = os.environ.get('MCTS_DEBUG', '0') == '1'

def debug_print(msg: str, level: str = "INFO"):
    """调试输出函数"""
    if DEBUG_MODE or level in ["ERROR", "WARNING", "CRITICAL"]:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {msg}")

def check_not_mock(value, name: str, expected_type=None):
    """
    检查值不是mock值（如随机数、默认值等）
    
    如果检测到mock值，输出警告
    """
    # 检查是否为随机数（在特定范围内的浮点数）
    if isinstance(value, float):
        if -12 < value < -5 and value not in [-6.0, -7.0, -8.0, -9.0, -10.0, -11.0]:
            debug_print(f"【警告】{name} = {value} 可能是随机生成的mock值！", "WARNING")
            debug_print(f"        请确保这是真实的计算结果，而非占位符", "WARNING")
            return False
    
    # 检查是否为0（Vina失败的默认值）
    if value == 0:
        debug_print(f"【警告】{name} = 0，这可能是Vina失败的默认值！", "WARNING")
        return False
    
    # 检查类型
    if expected_type and not isinstance(value, expected_type):
        debug_print(f"【错误】{name} 类型错误: 期望 {expected_type}, 实际 {type(value)}", "ERROR")
        return False
    
    return True

# 导入MCTS模块
from peptide_state import PeptideState, create_root_node, MCTSNode
from selection import PUCTSelector
from expansion import ExpansionEngine
from simulation import SimulationEngine
from backpropagation import BackpropagationEngine

# 其他模块
from seq_generator import generate_full_sequence
from vina import batch_vina_dock

# 导入EGNN模块
from EGNN_1 import main as egnn_data_prep
from EGNN_23 import main as egnn_train
from EGNN_4 import main as egnn_evaluate


@dataclass
class MCTSNodeSnapshot:
    """MCTS节点快照（轻量级，用于序列化）"""
    state_key: str              # PeptideState.to_key()
    visit_count: int
    total_score: float
    children_keys: List[str]    # 子节点标识列表
    decision_level: int
    decision_action: Optional[str]
    prior_prob: float


@dataclass
class MCTSTreeSnapshot:
    """MCTS树快照（轻量级，避免pickle循环引用问题）"""
    nodes: Dict[str, MCTSNodeSnapshot]  # state_key -> snapshot
    root_key: str
    iteration: int
    best_states: List[Tuple[str, float]]  # (state_key, score)
    all_data: Dict[str, float]  # state_key -> energy


@dataclass
class MCTSState:
    """MCTS状态（用于保存/恢复）- 保留用于兼容性"""
    root: MCTSNode
    iteration: int
    best_states: List[Tuple[PeptideState, float]]
    all_data: Dict[str, float]  # state_key -> energy


class Phase2Engine:
    """
    阶段二主引擎 V2 - 支持三层决策
    """
    
    def __init__(self, target_name: str, checkpoint_dir: Optional[Path] = None):
        """
        初始化引擎
        
        Args:
            target_name: 靶点名称
            checkpoint_dir: 检查点保存目录
        """
        self.target_name = target_name
        self.target_dirs = config.get_target_dirs(target_name)
        
        # 检查第一阶段是否完成
        if not (self.target_dirs["vina"] / "vina-receptor.pdbqt").exists():
            raise FileNotFoundError(f"请先运行阶段一")
        
        # 检查点目录
        if checkpoint_dir is None:
            self.checkpoint_dir = config.BASE_DIR / "checkpoints" / target_name
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 结果目录
        self.results_dir = config.RESULTS_DIR / target_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # MCTS组件（V2）
        self.selector = PUCTSelector(c_puct=config.MCTS_CONFIG["c_puct"])
        self.expansion_engine = ExpansionEngine()
        self.backprop_engine = BackpropagationEngine(verbose=False)
        
        # EGNN模型
        self.egnn_model = None
        self.egnn_model_path = config.BASE_DIR / "egnn" / "models" / "best_model.pt"
        
        # 统计信息
        self.iteration = 0
        self.best_states = []  # [(PeptideState, score), ...]
        self.all_data = {}     # {state_key: energy}
        
        # 数据文件
        self.dataset_path = self.results_dir / "dataset.csv"
        self.candidates_path = self.results_dir / "candidates.csv"
        
        self._init_dataset()
    
    def _init_dataset(self):
        """初始化数据集文件"""
        if not self.dataset_path.exists():
            with open(self.dataset_path, 'w') as f:
                f.write("sequence,crosslinker,disulfide_bonds,energy,data_source,iteration,timestamp\n")
        
        if not self.candidates_path.exists():
            with open(self.candidates_path, 'w') as f:
                f.write("sequence,crosslinker,disulfide_bonds,pred_energy,mcts_visits,mcts_avg_score,iteration,timestamp\n")
    
    def cold_start(self, n_sequences: int = 100, use_parallel: bool = True):
        """
        冷启动：生成初始数据并训练EGNN
        
        Args:
            n_sequences: 初始序列数量
            use_parallel: 是否使用并行Vina对接
        """
        print("="*60)
        print("阶段二：冷启动")
        print("="*60)
        
        # 1. 生成随机序列+拓扑
        print(f"\n[1/5] 生成 {n_sequences} 个随机序列+拓扑...")
        states = []
        for _ in range(n_sequences):
            seq = generate_full_sequence()
            # 随机选择交联剂
            xlinker_options = ["TBMB", "TATA", "disulfide", None]
            xlinker = random.choice(xlinker_options)
            
            state = PeptideState(sequence=seq, crosslinker=xlinker)
            
            # 如果是disulfide，随机选择一对
            if xlinker == "disulfide":
                bonds = state.get_possible_disulfide_bonds()
                if bonds:
                    state.disulfide_bonds = [random.choice(bonds)]
                    state._update_completion_status()
            
            states.append(state)
        
        print(f"  生成 {len(states)} 个状态")
        
        # 2. Vina对接
        print(f"\n[2/5] Vina对接（获取真实分数）...")
        
        # 准备对接参数
        from ligand_generator import generate_ligand
        from vina import get_vina_paths, run_vina_with_progress
        
        vina_paths = get_vina_paths(self.target_name)
        results = {}
        
        for i, state in enumerate(states):
            print(f"\n[{i+1}/{len(states)}] {state}")
            
            try:
                # 获取交联剂配置
                if state.crosslinker == "disulfide":
                    xlinker = "disulfide"
                    positions = list(state.disulfide_bonds[0]) if state.disulfide_bonds else None
                else:
                    xlinker = state.crosslinker
                    positions = config.CROSSLINKER_POSITIONS if xlinker else None
                
                # 生成分子
                pdbqt_path = generate_ligand(
                    sequence=state.sequence,
                    crosslinker=xlinker,
                    crosslinker_positions=positions
                )
                
                # Vina对接
                result = run_vina_with_progress(
                    ligand_pdbqt=pdbqt_path,
                    receptor_pdbqt=vina_paths['receptor'],
                    vina_config=vina_paths['config'],
                    n_cpu=1
                )
                
                if result.success:
                    results[state.to_key()] = result.binding_energy
                    print(f"  ✓ 结合能: {result.binding_energy:.4f}")
                else:
                    print(f"  ✗ 失败: {result.error_message}")
                    
            except Exception as e:
                print(f"  ✗ 异常: {e}")
        
        # 3. 保存到数据集
        print(f"\n[3/5] 更新数据集...")
        self._update_dataset(results, "cold_start")
        print(f"  有效数据: {len(results)}/{len(states)}")
        
        # 4. 训练EGNN
        print(f"\n[4/5] 训练初始EGNN模型...")
        self._train_egnn()
        
        # 5. 加载EGNN模型
        print(f"\n[5/5] 加载EGNN模型...")
        self._load_egnn_model()
        
        print("\n" + "="*60)
        print("冷启动完成！")
        print("="*60)
    
    def _update_dataset(self, results: Dict[str, float], source: str):
        """
        更新数据集（过滤无效数据）
        
        Bug 5修复：只保存有效的负结合能数据，避免Vina失败污染训练集
        """
        timestamp = datetime.now().isoformat()
        
        # 过滤无效数据（只保留负结合能）
        valid_results = {}
        filtered_count = 0
        
        for state_key, energy in results.items():
            # 检查能量是否有效（负结合能表示有效结合）
            if energy < 0:
                valid_results[state_key] = energy
            else:
                filtered_count += 1
                print(f"  [过滤] {state_key}: energy={energy} (非负值，可能是Vina失败)")
        
        if filtered_count > 0:
            print(f"  [数据过滤] 过滤了 {filtered_count} 个无效数据")
        
        # 保存有效数据
        with open(self.dataset_path, 'a') as f:
            for state_key, energy in valid_results.items():
                # 解析state_key
                parts = state_key.split('_')
                sequence = parts[0]
                crosslinker = parts[1] if len(parts) > 1 and parts[1] != "None" else "None"
                bonds_str = parts[2] if len(parts) > 2 else ""
                
                f.write(f"{sequence},{crosslinker},{bonds_str},{energy},{source},{self.iteration},{timestamp}\n")
                self.all_data[state_key] = energy
        
        # 更新最佳状态（只使用有效数据）
        for state_key, energy in valid_results.items():
            # 解析state（简化处理）
            parts = state_key.split('_')
            sequence = parts[0]
            crosslinker = parts[1] if len(parts) > 1 and parts[1] != "None" else None
            
            state = PeptideState(sequence=sequence, crosslinker=crosslinker)
            self.best_states.append((state, energy))
        
        # 排序并去重
        self.best_states = sorted(self.best_states, key=lambda x: x[1])[:1000]
    
    def _train_egnn(self):
        """训练EGNN模型"""
        print("  准备EGNN数据...")
        # TODO: 需要修改EGNN_1以支持新的数据格式
        
        print("  训练EGNN模型...")
        egnn_train(
            data_dir=config.BASE_DIR / "egnn" / "raw",
            output_dir=config.BASE_DIR / "egnn" / "models",
            hidden_dim=config.EGNN_CONFIG["hidden_dim"],
            num_layers=config.EGNN_CONFIG["num_layers"],
            num_epochs=config.EGNN_CONFIG["num_epochs"],
            batch_size=config.EGNN_CONFIG["batch_size"],
            lr=config.EGNN_CONFIG["learning_rate"],
            patience=config.EGNN_CONFIG["patience"]
        )
    
    def _load_egnn_model(self):
        """加载EGNN模型"""
        if not self.egnn_model_path.exists():
            print(f"警告: EGNN模型不存在")
            return False
        
        try:
            from EGNN_23 import EGNNModel
            
            self.egnn_model = EGNNModel(
                in_features=20,
                hidden_dim=config.EGNN_CONFIG["hidden_dim"],
                num_layers=config.EGNN_CONFIG["num_layers"]
            )
            
            checkpoint = torch.load(self.egnn_model_path, map_location='cpu')
            self.egnn_model.load_state_dict(checkpoint['model_state_dict'])
            self.egnn_model.eval()
            
            print(f"  ✓ 加载EGNN模型")
            return True
            
        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
            return False
    
    def _egnn_predict(self, state: PeptideState) -> float:
        """
        使用EGNN预测结合能
        
        Args:
            state: 肽状态（包含序列、交联剂、二硫键）
        
        Returns:
            预测结合能（kcal/mol）
        
        Raises:
            RuntimeError: 如果EGNN模型未加载或预测失败
        """
        debug_print(f"开始EGNN预测: {state}", "INFO")
        
        # 如果没有EGNN模型，尝试加载
        if self.egnn_model is None:
            debug_print("EGNN模型未加载，尝试加载...", "INFO")
            from egnn_predictor import create_egnn_predictor
            self.egnn_model = create_egnn_predictor()
            
            if self.egnn_model is None:
                debug_print("【严重错误】EGNN模型加载失败！", "CRITICAL")
                debug_print("        可能原因：", "CRITICAL")
                debug_print("        1. 模型文件不存在: egnn/models/best_model.pt", "CRITICAL")
                debug_print("        2. 模型文件损坏", "CRITICAL")
                debug_print("        3. PyTorch版本不兼容", "CRITICAL")
                debug_print("        解决方案：", "CRITICAL")
                debug_print("        1. 运行冷启动生成训练数据", "CRITICAL")
                debug_print("        2. 运行 EGNN_1.py 准备数据", "CRITICAL")
                debug_print("        3. 运行 EGNN_23.py 训练模型", "CRITICAL")
                raise RuntimeError(
                    "EGNN模型未加载。请先运行冷启动生成训练数据，"
                    "然后运行EGNN_1.py和EGNN_23.py训练模型。"
                )
            else:
                debug_print("✓ EGNN模型加载成功", "INFO")
        
        # 使用EGNN预测
        from ligand_generator import generate_ligand
        import tempfile
        
        # 获取交联剂配置
        if state.crosslinker == "disulfide":
            xlinker = "disulfide"
            positions = list(state.disulfide_bonds[0]) if state.disulfide_bonds else None
        else:
            xlinker = state.crosslinker
            positions = config.CROSSLINKER_POSITIONS if xlinker else None
        
        debug_print(f"生成分子: sequence={state.sequence[:20]}..., xlinker={xlinker}", "INFO")
        
        # 生成分子
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                pdbqt_path = generate_ligand(
                    sequence=state.sequence,
                    crosslinker=xlinker,
                    crosslinker_positions=positions,
                    output_dir=Path(tmpdir)
                )
                debug_print(f"✓ 分子生成成功: {pdbqt_path}", "INFO")
                
                # EGNN预测
                debug_print("开始EGNN预测...", "INFO")
                energy = self.egnn_model.predict(pdbqt_path)
                
                # 检查是否为mock值
                check_not_mock(energy, "EGNN预测能量", float)
                
                debug_print(f"✓ EGNN预测完成: energy={energy:.4f}", "INFO")
                return energy
                
        except Exception as e:
            debug_print(f"【错误】EGNN预测失败: {e}", "ERROR")
            debug_print(f"        序列: {state.sequence}", "ERROR")
            debug_print(f"        交联剂: {state.crosslinker}", "ERROR")
            raise
    
    def mcts_iteration(self, root: MCTSNode, n_iterations: int = 100,
                        max_expansions_per_step: Optional[int] = None) -> MCTSNode:
        """
        运行MCTS迭代（三层决策，逐步扩展）
        
        Args:
            root: 根节点
            n_iterations: 迭代次数
            max_expansions_per_step: 每次扩展的最大子节点数（默认从config读取）
        
        Returns:
            更新后的根节点
        """
        if max_expansions_per_step is None:
            max_expansions_per_step = config.MCTS_CONFIG.get("max_expansions", 5)
        
        sim_engine = SimulationEngine(
            target_name=self.target_name,
            egnn_model=self._egnn_predict,
            n_simulations=config.MCTS_CONFIG["n_simulations"]
        )
        
        for i in range(n_iterations):
            self.iteration += 1
            
            # 1. Selection: 选择路径
            # 当遇到可扩展节点时停止，让expansion处理
            path = self.selector.select_path(
                root,
                can_expand_fn=lambda node: self.expansion_engine.can_expand(node)
            )
            leaf = path[-1]
            
            # 2. Expansion: 逐步扩展（只扩展最多max_expansions_per_step个子节点）
            if not leaf.is_terminal and self.expansion_engine.can_expand(leaf):
                # 逐步扩展：只扩展部分子节点，不是全部
                new_children = self.expansion_engine.expand(
                    leaf, 
                    max_expansions=max_expansions_per_step
                )
                
                # 从新扩展的子节点中选择一个进行模拟
                if new_children:
                    # 策略：选择prior_prob最高的新子节点
                    best_child = max(new_children.values(), key=lambda c: c.prior_prob)
                    path.append(best_child)
                    leaf = best_child
                elif leaf.children:
                    # 如果没有新扩展，但从已有子节点中选择
                    # 选择visit_count最少的（鼓励探索）
                    best_child = min(leaf.children.values(), key=lambda c: c.visit_count)
                    path.append(best_child)
                    leaf = best_child
            
            # 3. Simulation: 评估
            if leaf.is_terminal:
                result = sim_engine.simulate(leaf, verbose=False)
                reward = result.score
            else:
                # 非终端节点使用启发式分数
                reward = 0.5
            
            # 4. Backpropagation: 回溯更新
            self.backprop_engine.backpropagate(path, reward)
            
            # 定期输出进度
            if (i + 1) % 100 == 0:
                # 统计树的大小
                def count_nodes(node):
                    count = 1
                    for child in node.children.values():
                        count += count_nodes(child)
                    return count
                
                tree_size = count_nodes(root)
                print(f"  MCTS迭代: {i+1}/{n_iterations}, 树节点数: {tree_size}")
        
        return root
    
    def extract_candidates(self, root: MCTSNode, n_candidates: int = 100) -> List[Tuple[PeptideState, float]]:
        """
        提取候选状态（迭代式DFS，避免栈溢出）
        
        Args:
            root: 根节点
            n_candidates: 候选数量
        
        Returns:
            List[Tuple[PeptideState, float]] - 候选状态和分数
        """
        candidates = []
        
        # 迭代式DFS（使用栈）
        stack = [root]
        visited = set()
        
        while stack:
            node = stack.pop()
            node_id = id(node)
            
            if node_id in visited:
                continue
            visited.add(node_id)
            
            # 如果是终端节点，添加到候选列表
            if node.is_terminal:
                # 深拷贝状态，避免后续树结构变化影响候选列表
                state_copy = node.state.copy()
                candidates.append((state_copy, node.average_score))
            
            # 将子节点加入栈
            for child in node.children.values():
                if id(child) not in visited:
                    stack.append(child)
        
        # 按分数排序，返回top-N
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:n_candidates]
    
    def _create_tree_snapshot(self, root: MCTSNode) -> MCTSTreeSnapshot:
        """
        创建MCTS树的轻量级快照（避免pickle循环引用）
        
        Bug 4修复：不序列化整个对象，只保存关键数据
        """
        nodes = {}
        
        # 迭代式遍历树
        stack = [root]
        visited = set()
        
        while stack:
            node = stack.pop()
            node_key = node.state.to_key()
            
            if node_key in visited:
                continue
            visited.add(node_key)
            
            # 创建节点快照
            children_keys = [child.state.to_key() for child in node.children.values()]
            snapshot = MCTSNodeSnapshot(
                state_key=node_key,
                visit_count=node.visit_count,
                total_score=node.total_score,
                children_keys=children_keys,
                decision_level=node.decision_level,
                decision_action=node.decision_action,
                prior_prob=node.prior_prob
            )
            nodes[node_key] = snapshot
            
            # 添加子节点到栈
            for child in node.children.values():
                child_key = child.state.to_key()
                if child_key not in visited:
                    stack.append(child)
        
        # 创建树快照
        best_states_keys = [(state.to_key(), score) for state, score in self.best_states]
        
        return MCTSTreeSnapshot(
            nodes=nodes,
            root_key=root.state.to_key(),
            iteration=self.iteration,
            best_states=best_states_keys,
            all_data=self.all_data.copy()
        )
    
    def _rebuild_tree_from_snapshot(self, snapshot: MCTSTreeSnapshot) -> Optional[MCTSNode]:
        """
        从快照重建MCTS树（恢复parent指针）
        
        Bug 4修复：重建时正确设置parent指针，确保backpropagation正常工作
        """
        if not snapshot.nodes or snapshot.root_key not in snapshot.nodes:
            return None
        
        # 第一步：创建所有节点（不设置parent和children）
        node_map = {}
        for key, snap in snapshot.nodes.items():
            # 解析state_key重建PeptideState
            parts = key.split('_')
            sequence = parts[0]
            crosslinker = parts[1] if len(parts) > 1 and parts[1] != "None" else None
            
            state = PeptideState(sequence=sequence, crosslinker=crosslinker)
            
            # 解析二硫键
            if len(parts) > 2 and parts[2]:
                bonds_str = parts[2].split('|')
                for bond_str in bonds_str:
                    if '-' in bond_str:
                        i, j = bond_str.split('-')
                        state.disulfide_bonds.append((int(i), int(j)))
            
            state._update_completion_status()
            
            # 创建节点
            node = MCTSNode(
                state=state,
                visit_count=snap.visit_count,
                total_score=snap.total_score,
                children={},  # 稍后填充
                parent=None,  # 稍后填充
                prior_prob=snap.prior_prob,
                decision_level=snap.decision_level,
                decision_action=snap.decision_action
            )
            node_map[key] = node
        
        # 第二步：恢复children和parent关系
        for key, snap in snapshot.nodes.items():
            node = node_map[key]
            for child_key in snap.children_keys:
                if child_key in node_map:
                    child = node_map[child_key]
                    child.parent = node
                    # 使用decision_action作为key
                    action = child.decision_action if child.decision_action else child_key
                    node.children[action] = child
        
        # 第三步：恢复引擎状态
        self.iteration = snapshot.iteration
        self.all_data = snapshot.all_data.copy()
        
        # 恢复best_states
        self.best_states = []
        for state_key, score in snapshot.best_states:
            if state_key in node_map:
                self.best_states.append((node_map[state_key].state, score))
        
        return node_map.get(snapshot.root_key)
    
    def save_checkpoint(self, root: MCTSNode, filename: Optional[str] = None):
        """
        保存检查点（使用轻量级快照，避免pickle循环引用问题）
        
        Bug 4修复：不pickle整个对象树，只保存关键数据
        """
        if filename is None:
            filename = f"checkpoint_iter{self.iteration}.json"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # 创建快照
        snapshot = self._create_tree_snapshot(root)
        
        # 保存为JSON（更可靠，跨平台兼容）
        import json
        with open(checkpoint_path, 'w') as f:
            json.dump({
                'nodes': {k: {
                    'state_key': v.state_key,
                    'visit_count': v.visit_count,
                    'total_score': v.total_score,
                    'children_keys': v.children_keys,
                    'decision_level': v.decision_level,
                    'decision_action': v.decision_action,
                    'prior_prob': v.prior_prob
                } for k, v in snapshot.nodes.items()},
                'root_key': snapshot.root_key,
                'iteration': snapshot.iteration,
                'best_states': snapshot.best_states,
                'all_data': snapshot.all_data
            }, f, indent=2)
        
        print(f"  ✓ 保存检查点: {checkpoint_path} ({len(snapshot.nodes)} 个节点)")
    
    def load_checkpoint(self, filename: str) -> Optional[MCTSNode]:
        """
        加载检查点（从JSON快照重建树）
        
        Bug 4修复：正确重建parent指针
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        if not checkpoint_path.exists():
            return None
        
        import json
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        
        # 重建快照对象
        nodes = {}
        for k, v in data['nodes'].items():
            nodes[k] = MCTSNodeSnapshot(
                state_key=v['state_key'],
                visit_count=v['visit_count'],
                total_score=v['total_score'],
                children_keys=v['children_keys'],
                decision_level=v['decision_level'],
                decision_action=v['decision_action'],
                prior_prob=v['prior_prob']
            )
        
        snapshot = MCTSTreeSnapshot(
            nodes=nodes,
            root_key=data['root_key'],
            iteration=data['iteration'],
            best_states=data['best_states'],
            all_data=data['all_data']
        )
        
        # 重建树
        root = self._rebuild_tree_from_snapshot(snapshot)
        
        if root:
            print(f"  ✓ 加载检查点: {checkpoint_path} (迭代 {self.iteration}, {len(nodes)} 个节点)")
        
        return root
    
    def run(self, 
            n_mcts_iterations: int = 1000,
            validation_interval: int = 5000,
            max_iterations: int = 50000):
        """运行阶段二主循环"""
        print("="*60)
        print(f"阶段二：MCTS三层决策闭环搜索")
        print(f"靶点: {self.target_name}")
        print("="*60)
        
        # 检查是否需要冷启动
        if not self.egnn_model_path.exists():
            print("\n未检测到EGNN模型，执行冷启动...")
            self.cold_start(n_sequences=100)
        else:
            print("\n检测到已有EGNN模型，加载中...")
            self._load_egnn_model()
        
        # 创建或加载根节点（从检查点）
        root = self.load_checkpoint("latest.json")
        if root is None:
            print("\n创建新的MCTS根节点...")
            root = create_root_node()
        
        print(f"\n开始MCTS搜索（最大迭代: {max_iterations}）...")
        
        while self.iteration < max_iterations:
            print(f"\n--- MCTS迭代 {self.iteration}-{self.iteration + n_mcts_iterations} ---")
            root = self.mcts_iteration(
                root, 
                n_mcts_iterations,
                max_expansions_per_step=config.MCTS_CONFIG.get("max_expansions", 5)
            )
            
            # 保存检查点
            self.save_checkpoint(root, "latest.json")
            
            if self.iteration % validation_interval < n_mcts_iterations:
                print(f"\n--- 触发验证（迭代 {self.iteration}） ---")
                candidates = self.extract_candidates(root, 100)
                print(f"  提取 {len(candidates)} 个候选")
                # TODO: Vina验证和重训
                
                # 保存当前检查点
                self.save_checkpoint(root, f"checkpoint_iter{self.iteration}.json")
        
        print("\n" + "="*60)
        print("阶段二完成！")
        print("="*60)
        
        final_candidates = self.extract_candidates(root, 10)
        print(f"\n最终候选（Top 10）:")
        for i, (state, score) in enumerate(final_candidates, 1):
            print(f"  {i}. {state}")
        
        # 保存最终检查点
        self.save_checkpoint(root, "final.json")
        
        return final_candidates


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='阶段二：MCTS三层决策闭环搜索')
    parser.add_argument('-t', '--target', type=str, required=True,
                       help='靶点名称')
    parser.add_argument('--cold-start', action='store_true',
                       help='强制冷启动')
    parser.add_argument('--n-sequences', type=int, default=100,
                       help='冷启动序列数量')
    parser.add_argument('--mcts-iter', type=int, default=1000,
                       help='每次MCTS迭代数')
    parser.add_argument('--val-interval', type=int, default=5000,
                       help='验证间隔')
    parser.add_argument('--max-iter', type=int, default=50000,
                       help='最大迭代数')
    
    args = parser.parse_args()
    
    engine = Phase2Engine(target_name=args.target)
    
    if args.cold_start:
        engine.cold_start(n_sequences=args.n_sequences)
    
    engine.run(
        n_mcts_iterations=args.mcts_iter,
        validation_interval=args.val_interval,
        max_iterations=args.max_iter
    )


if __name__ == "__main__":
    main()
