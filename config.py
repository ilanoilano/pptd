# -*- coding: utf-8 -*-
"""
MCTS-Directed 环肽设计项目 - 全局配置
所有可调整参数集中在此文件
"""

import os
from pathlib import Path

# =============================================================================
# 项目路径配置
# =============================================================================
BASE_DIR = Path("/mnt/d/code/AA")
PDB_DIR = BASE_DIR / "PDB"
RESULTS_DIR = BASE_DIR / "results"
FINAL_DIR = BASE_DIR / "final"
MODELS_DIR = BASE_DIR / "models"

# EGNN 模型路径
def get_model_dir(target_name: str) -> Path:
    """获取指定靶点的模型目录"""
    return MODELS_DIR / target_name

# 子目录路径（动态生成，基于靶点名称）
def get_target_dirs(target_name: str) -> dict:
    """获取指定靶点的所有工作目录"""
    target_base = RESULTS_DIR / target_name
    return {
        "cleaned": target_base / "cleaned",
        "pocket": target_base / "pocket",
        "vina": target_base / "vina",
        "af3_receptor": target_base / "AF3_receptor",
    }


# 靶点特定的 pocket 文件名配置
# 默认使用 "pocket1.pdb"，可为特定靶点指定不同文件名
TARGET_POCKET_FILES = {
    # 格式: "靶点名称": "pocket文件名"
    # 示例:
    # "5tdn": "pocket1.pdb",
    "1LYZ": "pocket-for-esmif.pdb",
}


def get_pocket_pdb_path(target_name: str) -> Path:
    """
    获取指定靶点的 pocket PDB 文件路径
    
    Args:
        target_name: 靶点名称（如 "1LYZ", "5tdn"）
    
    Returns:
        pocket PDB 文件的完整路径
    """
    pocket_dir = get_target_dirs(target_name)["pocket"]
    # 检查是否有特定配置，否则使用默认 "pocket1.pdb"
    pocket_filename = TARGET_POCKET_FILES.get(target_name, "pocket1.pdb")
    return pocket_dir / pocket_filename

# =============================================================================
# ESM-IF 模型配置
# =============================================================================
# ESM-IF 模型权重文件路径（相对于 BASE_DIR）
ESMIF_MODEL_REL_PATH = Path("models/esm_if1_gvp4_t16_142M_UR50.pt")
ESMIF_MODEL_PATH = BASE_DIR / ESMIF_MODEL_REL_PATH

# ESM-IF 模型下载地址（当本地文件不存在时自动下载）
ESMIF_MODEL_URL = "https://dl.fbaipublicfiles.com/fair-esm/models/esm_if1_gvp4_t16_142M_UR50.pt"

# =============================================================================
# 外部工具路径配置
# =============================================================================
TOOLS = {
    "fpocket": "fpocket",  # fpocket 可执行文件路径
    "obabel": "/usr/bin/obabel",  # Open Babel (AA 环境)
    "vina": "/home/ilano/miniconda3/envs/AA/bin/vina",      # AutoDock Vina (AA 环境)
    "babel_libdir": "/home/ilano/miniconda3/envs/AA/lib/openbabel/2.4.1",  # OpenBabel插件目录
}

# =============================================================================
# 肽模板配置（可自定义）
# =============================================================================

# 肽序列模板
# 格式说明：
# - 大写字母：固定氨基酸（如 A, C, G）
# - x：可变位置，由 MCTS 搜索确定
# 环肽格式：ACX₆CX₆CG（16个氨基酸，3个固定Cys，使用TBMB交联剂形成单环）
PEPTIDE_TEMPLATE = "ACxxxCxxxCG"

# 固定位置映射：{位置索引: 氨基酸}
# 位置从0开始计数
# 环肽 ACX₆CX₆CG：3个Cys用于TBMB交联形成单环
FIXED_POSITIONS = {
    0: "A",   # N端
    1: "C",   # 第一个半胱氨酸（TBMB连接位点1）
    5: "C",   # 第二个半胱氨酸（TBMB连接位点2）
    9: "C",  # C端
    10: "G"
}

# 可变位置（模板中 'x' 的位置）
def get_variable_positions(template: str = PEPTIDE_TEMPLATE) -> list:
    """从模板中提取可变位置索引"""
    return [i for i, aa in enumerate(template) if aa == 'x']

VARIABLE_POSITIONS = get_variable_positions()

# 标准20种氨基酸
AMINO_ACIDS = {
    "A", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"
}

# 可变位置允许的氨基酸集合
# 标准20种氨基酸（包含C，用于TBMB交联剂的第三个Cys）
ALLOWED_AMINO_ACIDS = [
    "A",  # Ala 丙氨酸

    "D",  # Asp 天冬氨酸
    "E",  # Glu 谷氨酸
    "F",  # Phe 苯丙氨酸
    "G",  # Gly 甘氨酸
    "H",  # His 组氨酸
    "I",  # Ile 异亮氨酸
    "K",  # Lys 赖氨酸
    "L",  # Leu 亮氨酸
    "M",  # Met 甲硫氨酸
    "N",  # Asn 天冬酰胺
    "P",  # Pro 脯氨酸
    "Q",  # Gln 谷氨酰胺
    "R",  # Arg 精氨酸
    "S",  # Ser 丝氨酸
    "T",  # Thr 苏氨酸
    "V",  # Val 缬氨酸
    "W",  # Trp 色氨酸
    "Y",  # Tyr 酪氨酸
]

# 每个可变位置的可选氨基酸（默认全部，可针对特定位置限制）
VARIABLE_AA_OPTIONS = {
    pos: ALLOWED_AMINO_ACIDS.copy() for pos in VARIABLE_POSITIONS
}

# 可变位置氨基酸映射（用于MCTS扩展）
VARIABLE_AMINO_ACIDS = {
    pos: ALLOWED_AMINO_ACIDS.copy() for pos in VARIABLE_POSITIONS
}

# =============================================================================
# 环化配置
# =============================================================================

# 二硫键配对列表：[(Cys1位置, Cys2位置), ...]
# 使用TBMB交联剂时，不需要二硫键（交联剂替代二硫键形成环）
DISULFIDE_BONDS = []

# 化学交联剂配置（必需，用于形成环肽）
# 选项：None, "TBMB", "TATA", "TBAB"
# TBMB: 三官能团交联剂，连接3个Cys形成1个环
CROSSLINKER = "TBMB"

# 交联剂连接位置（TBMB需要3个Cys）
# ACX₆CX₆CG格式：Cys位于位置1, 8，第三个Cys需要在可变区域中指定
# 这里配置前两个固定Cys的位置，第三个将在扩展时从可变区域选择
CROSSLINKER_POSITIONS = [1, 5,9]  # 基础位置，第三个Cys在可变区域中确定

# =============================================================================
# 分子量筛选范围（Da）
# =============================================================================
MOLECULAR_WEIGHT_RANGE = (800, 2000)

# =============================================================================
# MCTS 算法参数
# =============================================================================
MCTS_ITERATIONS = 30000         # 默认迭代搜索次数

MCTS_CONFIG = {
    "n_iterations": MCTS_ITERATIONS,  # 使用统一的迭代次数
    "c_puct": 1.414,            # PUCT 探索常数
    "n_simulations": 10,        # 每次迭代模拟次数
    "esmif_top_k": 5,           # ESM-IF 压缩分支因子至 top-k
    "max_expansions": 5,        # 每次扩展的最大子节点数
}

# =============================================================================
# EGNN + MCTS 闭环训练参数
# =============================================================================
EGNN_CONFIG = {
    "hidden_dim": 128,          # EGNN 隐藏层维度
    "num_layers": 4,            # EGNN 层数
    "learning_rate": 1e-3,      # 学习率
    "batch_size": 32,           # 批次大小
    "num_epochs": 100,          # 训练轮数
    "patience": 20,             # 早停耐心值
}

COLD_START_CONFIG = {
    "num_sequences": 10000,     # 冷启动生成的序列数量
    "num_workers": 1,           # 并行工作进程数
}

ACTIVE_LEARNING_CONFIG = {
    "mcts_iterations": 50000,   # 每轮 MCTS 探索迭代数
    "top_n_candidates": 5000,   # 每轮选择的候选数量
    "min_new_data": 2000,       # 触发重新训练的最小新增数据量
    "min_new_ratio": 0.2,       # 触发重新训练的最小新增比例
    "final_iterations": 100000, # 最终搜索的 MCTS 迭代数
    "top_n_final": 10,          # 最终输出的候选数量
}

# =============================================================================
# 并行 Vina 对接配置
# =============================================================================
PARALLEL_VINA_CONFIG = {
    "num_workers": 12,           # 默认并行工作进程数（CPU 核心数）
    # 【注意】cpu_per_worker 已从配置中移除，现在统一使用 VINA_CONFIG["cpu"]
    # 这样可以确保单分子对接和批量对接使用相同的CPU设置
    "timeout": 3000,              # 每个分子对接超时时间（秒）
}

# =============================================================================
# 打分权重配置
# =============================================================================
SCORING_WEIGHTS = {
    "vina_score": 1.0,          # Vina 结合能权重（负值越好）
    "hydrophobicity": 0.1,      # 疏水性评分权重
    "charge_balance": 0.1,      # 电荷平衡权重
    "protease_stability": 0.2,  # 蛋白酶稳定性权重
    "molecular_weight": 0.05,   # 分子量偏离惩罚
}

# =============================================================================
# Vina 对接配置
# =============================================================================
VINA_CONFIG = {
    "exhaustiveness": 2,       # 搜索详尽度
    "num_modes": 9,             # 输出构象数量
    "energy_range": 4,          # 能量范围（kcal/mol）
    "cpu": 10,                   # 使用CPU核心数
}

# 对接盒子默认尺寸（Å），实际从 fpocket 结果计算
VINA_BOX_SIZE = (22, 22,  22)

# =============================================================================
# 氨基酸物理化学性质（用于启发式评分）
# =============================================================================

# Kyte-Doolittle 疏水性标度（正值疏水，负值亲水）
HYDROPATHY = {
    "A": 1.8,   "C": 2.5,   "D": -3.5,  "E": -3.5,
    "F": 2.8,   "G": -0.4,  "H": -3.2,  "I": 4.5,
    "K": -3.9,  "L": 3.8,   "M": 1.9,   "N": -3.5,
    "P": -1.6,  "Q": -3.5,  "R": -4.5,  "S": -0.8,
    "T": -0.7,  "V": 4.2,   "W": -0.9,  "Y": -1.3,
}

# 氨基酸电荷（pH 7.4 近似）
AA_CHARGE = {
    "D": -1,    "E": -1,    "R": +1,    "K": +1,
    "H": 0.5,   # 组氨酸部分带电
}

# 氨基酸分子量（Da）
AA_MOLECULAR_WEIGHT = {
    "A": 89.09,   "C": 121.16,  "D": 133.10,  "E": 147.13,
    "F": 165.19,  "G": 75.07,   "H": 155.16,  "I": 131.17,
    "K": 146.19,  "L": 131.17,  "M": 149.21,  "N": 132.12,
    "P": 115.13,  "Q": 146.15,  "R": 174.20,  "S": 105.09,
    "T": 119.12,  "V": 117.15,  "W": 204.23,  "Y": 181.19,
}

# =============================================================================
# 蛋白酶切割位点预测配置
# =============================================================================
PROTEASE_CONFIG = {
    "trypsin": {
        # 胰蛋白酶：切割 K/R 后的肽键（除非后跟 P）
        "cut_after": ["K", "R"],
        "not_before": ["P"],
    },
    "chymotrypsin": {
        # 胰凝乳蛋白酶：切割 F/Y/W 后的肽键
        "cut_after": ["F", "Y", "W"],
        "not_before": ["P"],
    },
}

# =============================================================================
# 辅助函数
# =============================================================================

def calculate_peptide_weight(sequence: str) -> float:
    """计算肽链分子量（减去水分子）"""
    weight = sum(AA_MOLECULAR_WEIGHT.get(aa, 0) for aa in sequence)
    # 减去 (n-1) 个水分子（肽键形成失去的水）
    weight -= 18.015 * (len(sequence) - 1)
    return weight

def get_net_charge(sequence: str) -> float:
    """计算肽链净电荷（pH 7.4 近似）"""
    charge = sum(AA_CHARGE.get(aa, 0) for aa in sequence)
    # N端氨基 +1，C端羧基 -1
    charge += 1 - 1
    return charge

def is_weight_valid(sequence: str) -> bool:
    """检查分子量是否在允许范围内"""
    weight = calculate_peptide_weight(sequence)
    return MOLECULAR_WEIGHT_RANGE[0] <= weight <= MOLECULAR_WEIGHT_RANGE[1]

# =============================================================================
# 验证配置一致性
# =============================================================================

def validate_config():
    """验证配置参数的一致性"""
    # 检查模板长度
    template_len = len(PEPTIDE_TEMPLATE)
    
    # 检查固定位置是否在范围内
    for pos in FIXED_POSITIONS:
        if pos < 0 or pos >= template_len:
            raise ValueError(f"固定位置 {pos} 超出模板范围 [0, {template_len})")
    
    # 检查二硫键位置
    for cys1, cys2 in DISULFIDE_BONDS:
        if PEPTIDE_TEMPLATE[cys1] != 'C' and cys1 not in FIXED_POSITIONS:
            raise ValueError(f"二硫键位置 {cys1} 不是半胱氨酸")
        if PEPTIDE_TEMPLATE[cys2] != 'C' and cys2 not in FIXED_POSITIONS:
            raise ValueError(f"二硫键位置 {cys2} 不是半胱氨酸")
    
    # 检查可变位置
    var_pos = get_variable_positions()
    for pos in var_pos:
        if pos in FIXED_POSITIONS:
            raise ValueError(f"位置 {pos} 不能同时是固定位置和可变位置")
    
    print("✓ 配置验证通过")
    return True

if __name__ == "__main__":
    validate_config()
    print(f"肽模板: {PEPTIDE_TEMPLATE}")
    print(f"模板长度: {len(PEPTIDE_TEMPLATE)}")
    print(f"可变位置数: {len(VARIABLE_POSITIONS)}")
    print(f"二硫键配对: {DISULFIDE_BONDS}")
