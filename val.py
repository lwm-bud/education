# evaluate.py

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
import pandas as pd
import itertools
from collections import defaultdict
from tqdm import tqdm
import json
import os

# --- 1. 导入训练脚本中的必要组件 ---
# 我们需要复用配置、数据加载、图构建和模型定义
from train import (
    Config, 
    load_and_preprocess_data, 
    build_static_graphs, 
    build_dynamic_video_graph,
    TOME_Model # 确保导入的是我们新的模型
)

# --- 2. 评估主函数 ---
def evaluate(config, k=10):
    # 检查模型权重文件是否存在
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"错误: 模型权重文件未找到于 {config.MODEL_SAVE_PATH}")
        print("请先运行 train.py 来训练并保存模型。")
        return

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载元数据和ID映射
    (vc_data, cv_data, course_map, video_map, 
     concept_map, num_videos) = load_and_preprocess_data(config)

    # 加载验证序列
    try:
        val_sequences = load_sequences_from_json(config.VAL_DATA_JSON, video_map)
    except FileNotFoundError:
        print(f"错误: 找不到验证数据文件 {config.VAL_DATA_JSON}。")
        print("请创建 data/val_data.json 文件用于评估。")
        return

    # 构建静态图
    edge_index_course, edge_index_concept = build_static_graphs(
        vc_data, cv_data, course_map, video_map, concept_map, config
    )
    edge_index_course = edge_index_course.to(device)
    edge_index_concept = edge_index_concept.to(device)

    # 实例化模型
    model = TOME_Model(
        num_videos, 
        config.EMBED_DIM, 
        config.GNN_HIDDEN_DIM, 
        config.LSTM_HIDDEN_DIM
    ).to(device)

    # 加载已保存的权重
    print(f"从 {config.MODEL_SAVE_PATH} 加载模型权重...")
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    print("权重加载成功。")

    # 将模型设置为评估模式
    model.eval()

    # --- 开始评估 ---
    print(f"\n--- 开始在验证集上评估 (Top-{k}) ---")
    
    hits = []
    reciprocal_ranks = []
    
    # 禁用梯度计算
    with torch.no_grad():
        progress_bar = tqdm(val_sequences, desc="评估进度")
        
        for sequence in progress_bar:
            # 将ID加1以匹配嵌入层
            sequence_tensor = torch.tensor(sequence, dtype=torch.long).to(device) + 1
            
            # 至少需要一个历史视频和一个目标视频
            if len(sequence_tensor) < 2:
                continue

            # 我们将使用序列的前半部分来预测后半部分
            # 至少保留一个历史项
            split_point = max(1, len(sequence_tensor) // 2)
            history = sequence_tensor[:split_point]
            targets = sequence_tensor[split_point:]

            # 模拟观看历史，更新LSTM状态
            # 将历史序列包装成batch_size=1的形式
            history_batch = history.unsqueeze(0)
            
            # 前向传播获取整个历史序列的LSTM输出
            # predictions_history: [1, history_len, num_videos+1]
            predictions_history = model(history_batch, edge_index_course, edge_index_concept)
            
            # 我们只需要最后一个时间步的输出来预测下一个视频
            # last_step_logits: [1, num_videos+1]
            last_step_logits = predictions_history[:, -1, :]

            # 遍历所有需要预测的目标
            for target_video in targets:
                # 获取Top-K预测
                _, top_k_indices = torch.topk(last_step_logits, k, dim=1)
                
                # 检查是否命中
                if target_video in top_k_indices:
                    hits.append(1)
                    # 计算倒数排名
                    rank = (top_k_indices == target_video).nonzero(as_tuple=True)[1].item() + 1
                    reciprocal_ranks.append(1.0 / rank)
                else:
                    hits.append(0)
                    reciprocal_ranks.append(0.0)
                
                # 更新历史，为下一次预测做准备
                # 将当前目标加入历史，并重新计算最后一个时间步的输出
                history = torch.cat([history, target_video.unsqueeze(0)])
                history_batch = history.unsqueeze(0)
                predictions_history = model(history_batch, edge_index_course, edge_index_concept)
                last_step_logits = predictions_history[:, -1, :]

    # 计算最终指标
    if not hits:
        print("没有可评估的样本。")
        return

    hr_at_k = sum(hits) / len(hits)
    mrr_at_k = sum(reciprocal_ranks) / len(reciprocal_ranks)

    print("\n--- 评估结果 ---")
    print(f"HR@{k}: {hr_at_k:.4f}")
    print(f"MRR@{k}: {mrr_at_k:.4f}")
    print("------------------")

# --- 辅助函数 (从训练脚本复制) ---
def load_sequences_from_json(filepath, video_map):
    print(f"从 {filepath} 加载序列...")
    try:
        with open(filepath, 'r') as f:
            sequences_original_id = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到数据文件 {filepath}。")
        return None
    
    sequences_mapped = []
    for seq in sequences_original_id:
        mapped_seq = [video_map.get(vid) for vid in seq if video_map.get(vid) is not None]
        if len(mapped_seq) > 1:
            sequences_mapped.append(mapped_seq)
            
    print(f"成功加载并映射了 {len(sequences_mapped)} 条序列。")
    return sequences_mapped

if __name__ == '__main__':
    config = Config()
    evaluate(config, k=10)