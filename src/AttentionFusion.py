from torch import nn

class AttentionMLPResidualFusion(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionMLPResidualFusion, self).__init__()
        # 定义多头注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)

        # 定义线性映射层用于对齐属性嵌入的维度
        self.attr_projection = nn.Linear(768, embed_dim)  # 假设属性嵌入维度是 768

        # 定义MLP增强特征
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # 残差连接层
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, desc_embed, attr_embed):
        # 对齐属性嵌入的维度
        projected_attr_embed = self.attr_projection(attr_embed)  # shape: (batch_size, embed_dim)

        # 扩展维度以适配注意力机制
        # projected_desc_embed=self.attr
        query = desc_embed.unsqueeze(1)  # shape: (batch_size, 1, embed_dim)
        key_value = projected_attr_embed.unsqueeze(1)  # shape: (batch_size, 1, embed_dim)

        # 注意力交互
        attn_output, _ = self.attention(query=query, key=key_value, value=key_value)

        # 使用MLP增强特征
        attn_output = self.mlp(attn_output.squeeze(1))  # shape: (batch_size, embed_dim)

        # 最后通过全连接层得到融合后的表示，并加上残差连接
        fused_embed = self.fc(attn_output) + desc_embed  # 残差连接

        return fused_embed
