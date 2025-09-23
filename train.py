# train.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import GCNConv, GATConv
import pandas as pd
import itertools
from collections import defaultdict
from tqdm import tqdm
import json
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
# --- 1. 配置与超参数 ---
class Config:
    # 数据路径
    DATA_DIR = "./data/"
    VIDEO_CONCEPT_PATH = os.path.join(DATA_DIR, "video_concept.csv")
    USER_VIDEO_PATH = os.path.join(DATA_DIR, "user_video.csv")
    COURSE_VIDEO_PATH = os.path.join(DATA_DIR, "course_video.csv")
    
    # 外部数据集路径
    TRAIN_DATA_JSON = os.path.join(DATA_DIR, "train_data.json")
    VAL_DATA_JSON = os.path.join(DATA_DIR, "val_data.json")
    # 模型权重保存路径
    MODEL_SAVE_PATH = "./tome_model_weights.pth"
    
    # 模型参数
    EMBED_DIM = 128
    GNN_HIDDEN_DIM = 64  # GAT头数*隐藏维度 = 2*64=128
    LSTM_HIDDEN_DIM = 128
    
    # 训练参数
    LEARNING_RATE = 0.001
    EPOCHS = 10
    BATCH_SIZE = 16 # 使用批处理
    SIMILARITY_THRESHOLD = 0.2

# --- 2. 数据加载与预处理 ---
def load_and_preprocess_data(config):
    print("开始加载和预处理元数据...")
    try:
        video_concept_data = pd.read_csv(config.VIDEO_CONCEPT_PATH)
        course_video_data = pd.read_csv(config.COURSE_VIDEO_PATH)
    except FileNotFoundError as e:
        print(f"错误: 找不到数据文件 {e.filename}。请确保数据文件在 'data' 文件夹中。")
        exit()

    all_course_id = sorted(course_video_data['course_id'].unique())
    all_video_id = sorted(video_concept_data['video_id'].unique())
    all_concept_id = sorted(video_concept_data['concept_id'].unique())

    # 创建从0开始的ID映射
    course_map = {cid: i for i, cid in enumerate(all_course_id)}
    video_map = {vid: i for i, vid in enumerate(all_video_id)}
    concept_map = {cid: i for i, cid in enumerate(all_concept_id)}
    
    num_courses = len(all_course_id)
    num_videos = len(all_video_id)
    num_concepts = len(all_concept_id)
    
    print(f"数据统计: {num_videos}个视频, {num_courses}门课程, {num_concepts}个概念")
    
    return (video_concept_data, course_video_data, 
            course_map, video_map, concept_map, num_videos)

def load_sequences_from_json(filepath, video_map):
    print(f"从 {filepath} 加载序列...")
    with open(filepath, 'r') as f:
        sequences_original_id = json.load(f)
    
    sequences_mapped = []
    for seq in sequences_original_id:
        mapped_seq = [video_map.get(vid) for vid in seq if video_map.get(vid) is not None]
        if len(mapped_seq) > 1:
            sequences_mapped.append(mapped_seq)
            
    print(f"成功加载并映射了 {len(sequences_mapped)} 条序列。")
    return sequences_mapped

# --- 3. 图构建 ---
def build_static_graphs(video_concept_data, course_video_data, course_map, video_map, concept_map, config):
    print("开始构建静态图 (Gc 和 Gk)...")
    # 1. 构建课程关系图 (Gc)
    course_rela_map = defaultdict(list)
    for _, row in course_video_data.iterrows():
        course_rela_map[course_map[row['course_id']]].append(video_map[row['video_id']])
    
    course_edges = []
    for _, videos in tqdm(course_rela_map.items(), desc="构建 Gc"):
        for v1, v2 in itertools.combinations(videos, 2):
            course_edges.extend([[v1, v2], [v2, v1]])
    edge_index_course = torch.tensor(course_edges, dtype=torch.long).t().contiguous()
    print(f"课程关系图 (Gc) 构建完毕，边数: {edge_index_course.shape[1]}")

    # 2. 构建知识概念图 (Gk)
    knowledge_concept_rela_map = defaultdict(set)
    for _, row in video_concept_data.iterrows():
        knowledge_concept_rela_map[video_map[row['video_id']]].add(concept_map[row['concept_id']])
        
    def jaccard_similarity(set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0

    video_ids = list(knowledge_concept_rela_map.keys())
    concept_edges = []
    total_combinations = len(video_ids) * (len(video_ids) - 1) // 2
    for v1, v2 in tqdm(itertools.combinations(video_ids, 2), desc="构建 Gk", total=total_combinations):
        sim = jaccard_similarity(knowledge_concept_rela_map[v1], knowledge_concept_rela_map[v2])
        if sim > config.SIMILARITY_THRESHOLD:
            concept_edges.extend([[v1, v2], [v2, v1]])
    edge_index_concept = torch.tensor(concept_edges, dtype=torch.long).t().contiguous()
    print(f"知识概念图 (Gk) 构建完毕，边数: {edge_index_concept.shape[1]}")
    
    return edge_index_course, edge_index_concept

def build_dynamic_video_graph(sequence_batch):
    """根据一个批次的序列动态构建视频关系图 (Gv)"""
    video_edges = []
    for seq in sequence_batch:
        # 过滤掉padding (0)
        valid_seq = [v for v in seq if v != 0]
        if len(valid_seq) < 2:
            continue
        for i in range(len(valid_seq) - 1):
            video_edges.append([valid_seq[i], valid_seq[i+1]])
    
    if not video_edges:
        return torch.empty((2, 0), dtype=torch.long)
        
    return torch.tensor(video_edges, dtype=torch.long).t().contiguous()

# --- 4. 模型定义 ---
class TOME_Model(nn.Module):
    def __init__(self, num_videos, embed_dim, gnn_hidden_dim, lstm_hidden_dim):
        super(TOME_Model, self).__init__()
        
        self.num_videos = num_videos
        self.gnn_output_dim = gnn_hidden_dim * 2
        
        # 1. 视频嵌入层 (padding_idx=0)
        self.video_embedding = nn.Embedding(num_videos + 1, embed_dim, padding_idx=0)

        # 2. GNN层
        self.dgat_course = GATConv(embed_dim, gnn_hidden_dim, heads=2)
        self.dgat_concept = GATConv(embed_dim, gnn_hidden_dim, heads=2)
        self.gcn_video = GCNConv(embed_dim, self.gnn_output_dim)

        # 3. 拼接与MLP融合层
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.gnn_output_dim * 3, self.gnn_output_dim),
            nn.ReLU(),
            nn.Linear(self.gnn_output_dim, lstm_hidden_dim)
        )
        
        # 4. 标准LSTM层
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_dim,
            hidden_size=lstm_hidden_dim,
            batch_first=True
        )

        # 5. 预测层
        self.predictor = nn.Linear(lstm_hidden_dim, num_videos + 1)

    def forward(self, sequences, static_edge_index_c, static_edge_index_k):

        batch_size, seq_len = sequences.shape
        
        all_video_embeds = self.video_embedding.weight
        
        e_c_all_nodes = torch.relu(self.dgat_course(all_video_embeds, static_edge_index_c))
        e_k_all_nodes = torch.relu(self.dgat_concept(all_video_embeds, static_edge_index_k))
        
        dynamic_edge_index_v = build_dynamic_video_graph(sequences.tolist()).to(sequences.device)
        if dynamic_edge_index_v.numel() > 0:
            e_v_all_nodes = torch.relu(self.gcn_video(all_video_embeds, dynamic_edge_index_v))
        else:
            e_v_all_nodes = torch.zeros(self.num_videos + 1, self.gnn_output_dim, device=sequences.device)

        lstm_input_list = []
        for t in range(seq_len):
            current_video_ids = sequences[:, t]
            
            batch_rep_c = e_c_all_nodes[current_video_ids]
            batch_rep_k = e_k_all_nodes[current_video_ids]
            batch_rep_v = e_v_all_nodes[current_video_ids]
            
            concatenated_rep = torch.cat([batch_rep_c, batch_rep_k, batch_rep_v], dim=1)
            fused_rep = self.fusion_mlp(concatenated_rep)
            lstm_input_list.append(fused_rep)
            
        lstm_input = torch.stack(lstm_input_list, dim=1)
        
        lstm_output, _ = self.lstm(lstm_input)
        predictions = self.predictor(lstm_output)
        
        return predictions

# --- 5. Dataset 和 DataLoader ---
class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        # 视频ID已经从0开始，但模型嵌入层大小为num_videos+1，0为padding
        # 所以我们将所有ID加1
        return torch.tensor(self.sequences[idx], dtype=torch.long) + 1

def collate_fn(batch):
    batch.sort(key=len, reverse=True)
    sequences = [item for item in batch]
    inputs = [s[:-1] for s in sequences]
    targets = [s[1:] for s in sequences]
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return padded_inputs, padded_targets

# --- 6. 训练主函数 ---
def train():
    config = Config()
    (v_cou_data, con_v_data, course_map, video_map, concept_map, num_videos) = load_and_preprocess_data(config)

    train_sequences = load_sequences_from_json(config.TRAIN_DATA_JSON, video_map)
    train_dataset = SequenceDataset(train_sequences)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    edge_index_course, edge_index_concept = build_static_graphs( 
        v_cou_data, con_v_data, course_map, video_map, concept_map, config
    )
    edge_index_course = edge_index_course.to(device)
    edge_index_concept = edge_index_concept.to(device)

    model = TOME_Model(
        num_videos, 
        config.EMBED_DIM, 
        config.GNN_HIDDEN_DIM, 
        config.LSTM_HIDDEN_DIM
    ).to(device)
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print("\n--- 开始训练 (使用Concat -> MLP架构) ---")
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            predictions = model(inputs, edge_index_course, edge_index_concept)
            
            loss = loss_fn(predictions.view(-1, num_videos + 1), targets.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            # progress_bar.set_postfix(loss=loss.item())

        avg_epoch_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} 完成, 平均损失: {avg_epoch_loss:.4f}")

    print("\n--- 训练完成 ---")
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print("模型权重保存成功。")

if __name__ == '__main__':
    train()