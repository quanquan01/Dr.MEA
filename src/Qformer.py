import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, visual_embed, text_embed):
        # Normalize embeddings
        visual_embed = F.normalize(visual_embed, p=2, dim=1)
        text_embed = F.normalize(text_embed, p=2, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(visual_embed, text_embed.T) / self.temperature

        # Create labels
        batch_size = visual_embed.shape[0]
        labels = torch.arange(batch_size).to(sim_matrix.device)

        # Compute loss for both directions
        loss_visual_to_text = self.criterion(sim_matrix, labels)
        loss_text_to_visual = self.criterion(sim_matrix.T, labels)

        # Total loss
        loss = (loss_visual_to_text + loss_text_to_visual) / 2
        return loss

class QFormer(nn.Module):
    def __init__(self, num_queries, embed_dim, visual_dim=4096, text_dim=768, output_dim=512):
        super(QFormer, self).__init__()
        self.query_embed = nn.Parameter(torch.randn(num_queries, embed_dim))

        # Project both visual and text features to a common embedding space
        self.visual_fc = nn.Linear(visual_dim, embed_dim)
        self.text_fc = nn.Linear(text_dim, embed_dim)

        # Output layer for contrastive learning
        self.output_fc = nn.Linear(embed_dim, output_dim)

        # Attention layers
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads=8, dropout=0.1)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8, dropout=0.1)

    def forward(self, visual_features, text_features):
        # Project visual and text features to the common embedding dimension
        visual_features = self.visual_fc(visual_features)  # Shape: (batch_size, embed_dim)
        text_features = self.text_fc(text_features)  # Shape: (batch_size, embed_dim)

        # Prepare query embeddings
        query_embed = self.query_embed.unsqueeze(1).repeat(1, visual_features.size(0),
                                                           1)  # (num_queries, batch_size, embed_dim)

        # Self-attention on query embeddings
        query_embed, _ = self.self_attn(query_embed, query_embed, query_embed)

        # Cross-attention with visual features
        query_embed, _ = self.cross_attn(query_embed, visual_features.unsqueeze(0).repeat(query_embed.size(0), 1, 1),
                                         visual_features.unsqueeze(0).repeat(query_embed.size(0), 1, 1))

        # Cross-attention with text features
        query_embed, _ = self.cross_attn(query_embed, text_features.unsqueeze(0).repeat(query_embed.size(0), 1, 1),
                                         text_features.unsqueeze(0).repeat(query_embed.size(0), 1, 1))

        # Aggregate the query embeddings
        fused_features = query_embed.mean(dim=0)  # Shape: (batch_size, embed_dim)

        # Generate output embeddings for contrastive loss
        visual_embed = self.output_fc(visual_features)  # Shape: (batch_size, output_dim)
        text_embed = self.output_fc(text_features)  # Shape: (batch_size, output_dim)
        fused_embed = self.output_fc(fused_features)

        return fused_embed, visual_embed, text_embed


