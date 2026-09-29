import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn import knn_graph

class MoEHead(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=6, dropout=0.0):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim, output_dim)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_experts)
        )

    def forward(self, x):
        gate_logits = self.gate(x)
        gate_weights = F.softmax(gate_logits, dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        final_output = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)
        return final_output, gate_logits


class GR2ST(nn.Module):
    def __init__(self, temperature, image_dim, spot_dim, projection_dim,
                 heads_num, dropout=0.0,
                 fusion_type='concat',
                 alpha_mse=50.0,
                 alpha_gate=1.0,
                 alpha_entropy=0.01,
                 spatial_radius=3.0,
                 conf_threshold=0.6):
        super().__init__()
        self.temperature = temperature
        self.alpha_mse = alpha_mse
        self.alpha_gate = alpha_gate
        self.alpha_entropy = alpha_entropy
        self.spatial_radius = spatial_radius
        self.conf_threshold = conf_threshold
        self.heads_num = heads_num
        self.fusion_type = fusion_type

        self.x_embed = nn.Embedding(65536, spot_dim)
        self.y_embed = nn.Embedding(65536, spot_dim)
        
        self.cell_type_embed = nn.Sequential(
            nn.Embedding(6, spot_dim//2),
            nn.Linear(spot_dim//2, spot_dim),
            nn.ReLU(),
            nn.LayerNorm(spot_dim),
            nn.Dropout(dropout)
        )
       
        self.image_projection = ProjectionHead(embedding_dim=image_dim, projection_dim=projection_dim, dropout=dropout)
        
        self.dynamic_head_proj = nn.Linear(spot_dim, projection_dim)
        self.dynamic_tail_proj = nn.Linear(spot_dim, projection_dim)
        self.spatial_proj = nn.Linear(spot_dim, projection_dim)
        
        self.dynamic_gat = GATConv(projection_dim, projection_dim, heads=heads_num, dropout=dropout)
        self.spatial_gat = GATConv(projection_dim, projection_dim, heads=heads_num, dropout=dropout)
        
        if fusion_type == 'concat':
            self.fusion_dim = heads_num * projection_dim * 2  
        else:
            self.fusion_dim = heads_num * projection_dim * 1  
            
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.fusion_dim, projection_dim),  
            nn.ReLU(),
            nn.LayerNorm(projection_dim),
            nn.Dropout(dropout)
        )
        
        self.spot_projection = ProjectionHead(embedding_dim=projection_dim, projection_dim=projection_dim, dropout=dropout)
        
        self.moe_head = MoEHead(input_dim=projection_dim, output_dim=spot_dim, num_experts=6, dropout=dropout)

    def build_threshold_functional_graph(self, spot_features, cell_types):
        head_features = self.dynamic_head_proj(spot_features)
        tail_features = self.dynamic_tail_proj(spot_features)
        
        raw_sim = torch.mm(head_features, tail_features.t()) / self.temperature
        feature_sim = torch.sigmoid(raw_sim) 
        
        cell_type_sim = (cell_types.unsqueeze(1) == cell_types.unsqueeze(0)).float()
        combined_sim = 0.7 * feature_sim + 0.3 * cell_type_sim
        
        mask = combined_sim > self.conf_threshold
        
        diag_mask = torch.eye(mask.size(0), device=mask.device).bool()
        mask = mask | diag_mask
        
        edge_index = mask.nonzero().t()
        return edge_index

    def build_radius_spatial_graph(self, positions):
        dist_matrix = torch.cdist(positions, positions)
        mask = dist_matrix < self.spatial_radius
        edge_index = mask.nonzero().t()
        return edge_index

    def forward(self, batch):
        image_features = batch["image_features"]
        image_embeddings = self.image_projection(image_features)
        
        spot_feature = batch["expression"]
        x = batch["position"][:, 0].long()
        y = batch["position"][:, 1].long()
        centers_x = self.x_embed(x)
        centers_y = self.y_embed(y)
        cell_type = batch["cell_type"].long()
        cell_type_embeddings = self.cell_type_embed(cell_type)
        
        spot_features = spot_feature + centers_x + centers_y + cell_type_embeddings
        
        positions_float = batch["position"].float()
        dynamic_edge_index = self.build_threshold_functional_graph(spot_features, cell_type)
        spatial_edge_index = self.build_radius_spatial_graph(positions_float)

        dynamic_features = self.dynamic_head_proj(spot_features)
        spatial_features = self.spatial_proj(spot_features)
        
        dynamic_features, (_, alpha_dyn) = self.dynamic_gat(dynamic_features, dynamic_edge_index, return_attention_weights=True)
        spatial_features, (_, alpha_spa) = self.spatial_gat(spatial_features, spatial_edge_index, return_attention_weights=True)
        
        def calculate_entropy(alpha):
            return -torch.mean(torch.sum(alpha * torch.log(alpha + 1e-9), dim=1))
            
        entropy_loss = calculate_entropy(alpha_dyn) + calculate_entropy(alpha_spa)
        
        if self.fusion_type == 'concat':
            fused_features = torch.cat([dynamic_features, spatial_features], dim=-1)
        elif self.fusion_type == 'sum':
            fused_features = dynamic_features + spatial_features
        elif self.fusion_type == 'max':
            fused_features = torch.max(dynamic_features, spatial_features)
        else:
            fused_features = (dynamic_features + spatial_features) / 2
        fused_features = self.fusion_layer(fused_features)
        spot_embeddings = self.spot_projection(fused_features)
        
        cos_sim = (spot_embeddings @ image_embeddings.T) / self.temperature
        labels = torch.eye(cos_sim.shape[0], device=cos_sim.device)
        con_loss = (F.cross_entropy(cos_sim, labels) + F.cross_entropy(cos_sim.T, labels.T)) / 2.0
        
        pred_expression, gate_logits = self.moe_head(image_embeddings)
        mse_loss = F.mse_loss(pred_expression, spot_feature)
        gate_loss = F.cross_entropy(gate_logits, cell_type)
        
        total_loss = con_loss + \
                     self.alpha_mse * mse_loss + \
                     self.alpha_gate * gate_loss + \
                     self.alpha_entropy * entropy_loss
        
        return total_loss

    def predict_expression(self, image_features):
        image_embeddings = self.image_projection(image_features)
        pred_expression, _ = self.moe_head(image_embeddings)
        return pred_expression

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout=0.0):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x