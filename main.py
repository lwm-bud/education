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
import math
import swanlab

# --- 1. 配置与超参数 ---
class Config:
    DATA_DIR = "./data/"
    VIDEO_CONCEPT_PATH = os.path.join(DATA_DIR, "video_concept.csv")
    USER_VIDEO_PATH = os.path.join(DATA_DIR, "user_video.csv")
    COURSE_VIDEO_PATH = os.path.join(DATA_DIR, "course_video.csv")
    TRAIN_DATA_JSON = os.path.join(DATA_DIR, "train_data.json")
    VAL_DATA_JSON = os.path.join(DATA_DIR, "val_data.json")
    MODEL_SAVE_PATH = "./tome_paper_lstm_weights.pth"
    
    EMBED_DIM = 128
    GNN_HIDDEN_DIM = 64
    LSTM_HIDDEN_DIM = 128
    
    LEARNING_RATE = 0.001
    EPOCHS = 10
    BATCH_SIZE = 16
    SIMILARITY_THRESHOLD = 0.2
    EVAL_K = 10

# --- 2. 数据加载与预处理 (与之前相同) ---
def load_and_preprocess_data(config):
    print("开始加载和预处理元数据...")
    video_concept_data = pd.read_csv(config.VIDEO_CONCEPT_PATH)
    course_video_data = pd.read_csv(config.COURSE_VIDEO_PATH)
    
    all_course_id = sorted(course_video_data['course_id'].unique())
    all_video_id = sorted(video_concept_data['video_id'].unique())
    all_concept_id = sorted(video_concept_data['concept_id'].unique())

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

# --- 3. 图构建 (与之前相同) ---
def build_static_graphs(video_concept_data, course_video_data, course_map, video_map, concept_map, config):
    print("开始构建静态图 (Gc 和 Gk)...")
    course_rela_map = defaultdict(list)
    for _, row in course_video_data.iterrows():
        course_rela_map[course_map[row['course_id']]].append(video_map[row['video_id']])
    course_edges = []
    for _, videos in tqdm(course_rela_map.items(), desc="构建 Gc"):
        for v1, v2 in itertools.combinations(videos, 2):
            course_edges.extend([[v1, v2], [v2, v1]])
    edge_index_course = torch.tensor(course_edges, dtype=torch.long).t().contiguous()
    print(f"课程关系图 (Gc) 构建完毕，边数: {edge_index_course.shape[1]}")

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
    video_edges = []
    for seq in sequence_batch:
        valid_seq = [v for v in seq if v != 0]
        if len(valid_seq) < 2:
            continue
        for i in range(len(valid_seq) - 1):
            video_edges.append([valid_seq[i], valid_seq[i+1]])
    if not video_edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(video_edges, dtype=torch.long).t().contiguous()

# --- 4. 模型定义 (严格按照论文实现) ---
class LSTM_layer(nn.Module):
    # 这个类本身是正确的，无需修改
    def __init__(self, input_dim, context_dim, hidden_dim):
        super(LSTM_layer, self).__init__()
        self.W_i = nn.Linear(input_dim, hidden_dim, bias=True)
        self.U_i = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_ic = nn.Linear(context_dim, hidden_dim, bias=False)
        self.Z_ik = nn.Linear(context_dim, hidden_dim, bias=False)
        self.Q_iv = nn.Linear(context_dim, hidden_dim, bias=False)

        self.W_f = nn.Linear(input_dim, hidden_dim, bias=True)
        self.U_f = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_fc = nn.Linear(context_dim, hidden_dim, bias=False)
        self.Z_fk = nn.Linear(context_dim, hidden_dim, bias=False)
        self.Q_fv = nn.Linear(context_dim, hidden_dim, bias=False)
        
        self.W_o = nn.Linear(input_dim, hidden_dim, bias=True)
        self.U_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_oc = nn.Linear(context_dim, hidden_dim, bias=False)
        self.Z_ok = nn.Linear(context_dim, hidden_dim, bias=False)
        self.Q_ov = nn.Linear(context_dim, hidden_dim, bias=False)
        
        self.W_c = nn.Linear(input_dim, hidden_dim, bias=True)
        self.U_c = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_cc = nn.Linear(context_dim, hidden_dim, bias=False)
        self.Z_ck = nn.Linear(context_dim, hidden_dim, bias=False)
        self.Q_cv = nn.Linear(context_dim, hidden_dim, bias=False)

    def forward(self, x_prev, states, context_vectors):
        h_prev, c_prev = states
        e_c, e_k, e_v = context_vectors
        i_t = torch.sigmoid(self.W_i(x_prev) + self.U_i(h_prev) + self.V_ic(e_c) + self.Z_ik(e_k) + self.Q_iv(e_v))
        f_t = torch.sigmoid(self.W_f(x_prev) + self.U_f(h_prev) + self.V_fc(e_c) + self.Z_fk(e_k) + self.Q_fv(e_v))
        o_t = torch.sigmoid(self.W_o(x_prev) + self.U_o(h_prev) + self.V_oc(e_c) + self.Z_ok(e_k) + self.Q_ov(e_v))
        c_tilde_t = torch.tanh(self.W_c(x_prev) + self.U_c(h_prev) + self.V_cc(e_c) + self.Z_ck(e_k) + self.Q_cv(e_v))
        c_t = f_t * c_prev + i_t * c_tilde_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

class TOME_Paper_LSTM_Model(nn.Module):
    def __init__(self, num_videos, embed_dim, gnn_hidden_dim, lstm_hidden_dim):
        super(TOME_Paper_LSTM_Model, self).__init__()
        self.num_videos = num_videos
        self.context_dim = gnn_hidden_dim * 2  # GAT有2个头

        self.video_embedding = nn.Embedding(num_videos + 1, embed_dim, padding_idx=0)
        self.dgat_course = GATConv(embed_dim, gnn_hidden_dim, heads=2)
        self.dgat_concept = GATConv(embed_dim, gnn_hidden_dim, heads=2)
        self.gcn_video = GCNConv(embed_dim, self.context_dim)
        
        self.lstm_cell = LSTM_layer(
            input_dim=embed_dim, # 输入是前一个视频的原始嵌入
            context_dim=self.context_dim,
            hidden_dim=lstm_hidden_dim
        )
        self.predictor = nn.Linear(lstm_hidden_dim, num_videos + 1)

    def forward(self, sequences, static_edge_index_c, static_edge_index_k):
        batch_size, seq_len = sequences.shape
        device = sequences.device
        
        # 1. 预计算所有节点的GNN表征
        all_video_embeds = self.video_embedding.weight
        e_c_all_nodes = torch.relu(self.dgat_course(all_video_embeds, static_edge_index_c))
        e_k_all_nodes = torch.relu(self.dgat_concept(all_video_embeds, static_edge_index_k))
        
        dynamic_edge_index_v = build_dynamic_video_graph(sequences.tolist()).to(device)
        if dynamic_edge_index_v.numel() > 0:
            e_v_all_nodes = torch.relu(self.gcn_video(all_video_embeds, dynamic_edge_index_v))
        else:
            e_v_all_nodes = torch.zeros(self.num_videos + 1, self.context_dim, device=device)

        # 2. 初始化LSTM状态
        h_state = torch.zeros(batch_size, self.lstm_cell.W_i.out_features, device=device)
        c_state = torch.zeros(batch_size, self.lstm_cell.W_i.out_features, device=device)
        
        outputs = []
        # 3. 逐个时间步处理序列
        for t in range(seq_len):
            current_video_ids = sequences[:, t]
            
            # 获取当前视频的原始嵌入作为LSTM输入 (e^r_t)
            x_t = self.video_embedding(current_video_ids)
            
            # 获取当前视频的节点级上下文表征 (e^C_t, e^K_t, e^V_t)
            context_c = e_c_all_nodes[current_video_ids]
            context_k = e_k_all_nodes[current_video_ids]
            context_v = e_v_all_nodes[current_video_ids]
            
            # 更新LSTM状态
            h_state, c_state = self.lstm_cell(
                x_prev=x_t,
                states=(h_state, c_state),
                context_vectors=(context_c, context_k, context_v)
            )
            outputs.append(h_state)
        
        # 4. 收集所有时间步的输出并进行预测
        lstm_output = torch.stack(outputs, dim=1)
        predictions = self.predictor(lstm_output)
        
        return predictions

# --- 5. Dataset 和 DataLoader (与之前相同) ---
class SequenceDataset(Dataset):
     
    def __init__(self, sequences):
        self.sequences = sequences
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long) + 1

def collate_fn(batch):
     
    batch.sort(key=len, reverse=True)
    sequences = [item for item in batch]
    inputs = [s[:-1] for s in sequences]
    targets = [s[1:] for s in sequences]
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return padded_inputs, padded_targets

# --- 6. 验证函数 (与之前相同) ---
def validate(model, val_loader, static_graphs, loss_fn, device, k):
     
    model.eval()
    total_loss = 0
    hits, ndcgs = [], []
    edge_index_course, edge_index_concept = static_graphs
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs, edge_index_course, edge_index_concept)
            loss = loss_fn(predictions.view(-1, model.num_videos + 1), targets.view(-1))
            total_loss += loss.item()
            for i in range(inputs.shape[0]):
                seq_len = (inputs[i] != 0).sum()
                if seq_len == 0: continue
                pred_seq, target_seq = predictions[i, :seq_len, :], targets[i, :seq_len]
                _, top_k_indices = torch.topk(pred_seq, k, dim=1)
                for j in range(seq_len):
                    target_video = target_seq[j]
                    if target_video in top_k_indices[j]:
                        hits.append(1)
                        rank = (top_k_indices[j] == target_video).nonzero(as_tuple=True)[0].item() + 1
                        ndcgs.append(1.0 / math.log2(rank+1))
                    else:
                        hits.append(0)
                        ndcgs.append(0.0)
    avg_loss = total_loss / len(val_loader)
    hr_at_k = sum(hits) / len(hits) if hits else 0
    ndcg_at_k = sum(ndcgs) / len(ndcgs) if ndcgs else 0
    return avg_loss, hr_at_k, ndcg_at_k

# --- 7. 训练主函数 ---
def main():
    config = Config()
    
    swanlab.init(
        project="MOOC_Recommendation_Paper",
        experiment_name="002"
        # config=config
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    (vc_data, cv_data, course_map, video_map, 
     concept_map, num_videos) = load_and_preprocess_data(config)

    train_sequences = load_sequences_from_json(config.TRAIN_DATA_JSON, video_map)
    val_sequences = load_sequences_from_json(config.VAL_DATA_JSON, video_map)
    
    if not train_sequences or not val_sequences:
        print("训练或验证数据为空，程序退出。")
        return

    train_dataset = SequenceDataset(train_sequences)
    val_dataset = SequenceDataset(val_sequences)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    edge_index_course, edge_index_concept = build_static_graphs(
        vc_data, cv_data, course_map, video_map, concept_map, config
    )
    edge_index_course = edge_index_course.to(device)
    edge_index_concept = edge_index_concept.to(device)
    static_graphs = (edge_index_course, edge_index_concept)

    # 使用新的、严格遵循论文的模型
    model = TOME_Paper_LSTM_Model(
        num_videos, 
        config.EMBED_DIM, 
        config.GNN_HIDDEN_DIM, 
        config.LSTM_HIDDEN_DIM
    ).to(device)
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print("\n--- 开始训练与验证 ---")
    for epoch in range(config.EPOCHS):
        model.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [训练]")
        
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            predictions = model(inputs, edge_index_course, edge_index_concept)
            
            loss = loss_fn(predictions.view(-1, num_videos + 1), targets.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            swanlab.log({"loss":loss})

        avg_train_loss = total_train_loss / len(train_loader)
        
        avg_val_loss, hr_at_k, ndcg_at_k = validate(model, val_loader, static_graphs, loss_fn, device, k=config.EVAL_K)
        
        print(f"Epoch {epoch+1} 完成 | 训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f} | HR@{config.EVAL_K}: {hr_at_k:.4f} | NDCG@{config.EVAL_K}: {ndcg_at_k:.4f}")
        
        swanlab.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            f"HR@{config.EVAL_K}": hr_at_k,
            f"NDCG@{config.EVAL_K}": ndcg_at_k,
            "epoch": epoch + 1
        })

    print("\n--- 训练完成 ---")
    print(f"正在保存模型权重到 {config.MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print("模型权重保存成功。")

if __name__ == '__main__':
    main()