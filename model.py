

import torch
from torch import nn
import math
from torch.nn.utils import weight_norm
from typing import Tuple, Optional, Dict
from dgl.nn.pytorch import GATv2Conv, GlobalAttentionPooling


# ============================================
# 基础模块（TCN相关）
# ============================================

class Chomp1d(nn.Module):
    """裁剪模块，用于实现因果卷积"""

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalCnn(nn.Module):
    """TCN基础块"""

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalCnn, self).__init__()
        self.conv = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
        self.chomp = Chomp1d(padding)
        self.leakyrelu = nn.LeakyReLU(True)
        self.dropout = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv, self.chomp, self.leakyrelu, self.dropout)
        self.init_weights()

    def init_weights(self):
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        return self.net(x)


class Tcn_Local(nn.Module):
    """局部TCN（因果卷积）"""

    def __init__(self, num_outputs, kernel_size=3, dropout=0.2):
        super(Tcn_Local, self).__init__()
        layers = []
        num_levels = 3
        out_channels = num_outputs
        for i in range(num_levels):
            layers += [TemporalCnn(out_channels, out_channels, kernel_size, stride=1, dilation=1,
                                   padding=(kernel_size - 1), dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Tcn_Global(nn.Module):
    """全局TCN（扩张卷积）"""

    def __init__(self, num_inputs, num_outputs, kernel_size=3, dropout=0.2):
        super(Tcn_Global, self).__init__()
        layers = []
        num_levels = math.ceil(math.log2((num_inputs - 1) * (2 - 1) / (kernel_size - 1) + 1))
        out_channels = num_outputs
        for i in range(num_levels):
            dilation_size = 2 ** i
            layers += [TemporalCnn(out_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                   padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ============================================
# 双TCN特征提取模块
# ============================================

class DualTCNFeatureExtractor(nn.Module):
    """
    双TCN特征提取模块
    并行提取局部和全局时间模式
    """

    def __init__(
            self,
            metric_num: int,
            seq_length: int,
            tcn_output_dim: int = 64,
            kernel_size: int = 3,
            dropout: float = 0.2,
            use_batch_norm: bool = True,
            residual_connection: bool = False
    ):
        """
        Args:
            metric_num: 输入指标数量
            seq_length: 时间序列长度
            tcn_output_dim: TCN输出维度
            kernel_size: 卷积核大小
            dropout: Dropout率
            use_batch_norm: 是否使用批归一化
            residual_connection: 是否使用残差连接
        """
        super(DualTCNFeatureExtractor, self).__init__()

        self.metric_num = metric_num
        self.seq_length = seq_length
        self.tcn_output_dim = tcn_output_dim
        self.kernel_size = kernel_size
        self.use_batch_norm = use_batch_norm
        self.residual_connection = residual_connection

        # 输入投影层：将metric_num维度投影到tcn_output_dim
        self.input_projection = nn.Linear(metric_num, tcn_output_dim)

        # 局部TCN：捕捉短期依赖
        self.local_tcn = Tcn_Local(
            num_outputs=tcn_output_dim,
            kernel_size=kernel_size,
            dropout=dropout
        )

        # 全局TCN：捕捉长期依赖
        self.global_tcn = Tcn_Global(
            num_inputs=seq_length,
            num_outputs=tcn_output_dim,
            kernel_size=kernel_size,
            dropout=dropout
        )

        # 批归一化层
        if use_batch_norm:
            self.local_bn = nn.BatchNorm1d(tcn_output_dim)
            self.global_bn = nn.BatchNorm1d(tcn_output_dim)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.input_projection.weight)
        if self.input_projection.bias is not None:
            nn.init.constant_(self.input_projection.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入张量 (Batch, SeqLen, MetricNum)

        Returns:
            local_features: (Batch, MetricNum, TCN_OutputDim)
            global_features: (Batch, MetricNum, TCN_OutputDim)
        """
        # x: (B, T, M) -> (B, T, D)
        x_proj = self.input_projection(x)  # (B, T, D)

        # 转换为TCN期望的格式: (B, D, T)
        x_tcn = x_proj.permute(0, 2, 1)  # (B, D, T)

        # 双TCN并行特征提取
        local_features = self.local_tcn(x_tcn)  # (B, D, T)
        global_features = self.global_tcn(x_tcn)  # (B, D, T)

        # 批归一化
        if self.use_batch_norm:
            local_features = self.local_bn(local_features)
            global_features = self.global_bn(global_features)

        # 残差连接
        if self.residual_connection:
            local_features = local_features + x_tcn
            global_features = global_features + x_tcn

        # 转换为 (B, M, D) 格式以便后续处理
        # 这里取最后一个时间步的特征作为每个指标的表示
        local_features = local_features[:, :, -1]  # (B, D)
        global_features = global_features[:, :, -1]  # (B, D)

        # 扩展为 (B, M, D) - 每个metric共享相同的特征
        # 注意：这里需要将特征广播到所有metric
        local_features = local_features.unsqueeze(1).expand(-1, self.metric_num, -1)  # (B, M, D)
        global_features = global_features.unsqueeze(1).expand(-1, self.metric_num, -1)  # (B, M, D)

        return local_features, global_features


# ============================================
# CCM模块（频域相关性）
# ============================================

class CCMModule(nn.Module):
    """
    Channel Correlation Matrix模块
    基于频域方法提取metrics之间的关系
    """

    def __init__(
            self,
            metric_num: int,
            seq_length: int,
            correlation_threshold: float = 0.4,
            use_learnable_threshold: bool = True
    ):
        """
        Args:
            metric_num: 指标数量
            seq_length: 序列长度
            correlation_threshold: 相关性阈值
            use_learnable_threshold: 是否使用可学习的阈值
        """
        super(CCMModule, self).__init__()

        self.metric_num = metric_num
        self.seq_length = seq_length

        # 可学习的阈值参数
        if use_learnable_threshold:
            self.threshold = nn.Parameter(torch.tensor(correlation_threshold))
        else:
            self.register_buffer('threshold', torch.tensor(correlation_threshold))

        # 频域特征提取权重
        self.freq_weight = nn.Parameter(torch.ones(seq_length // 2 + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 (Batch, SeqLen, MetricNum)

        Returns:
            channel_mask: 通道相关性矩阵 (Batch, 1, MetricNum, MetricNum)
        """
        batch_size = x.size(0)

        # 转换为 (B, M, T)
        x_freq = x.permute(0, 2, 1)  # (B, M, T)

        # FFT变换到频域
        x_fft = torch.fft.rfft(x_freq, dim=2)  # (B, M, T//2+1)
        x_fft_abs = torch.abs(x_fft)  # 取幅值

        # 加权频域特征
        weighted_fft = x_fft_abs * self.freq_weight.unsqueeze(0).unsqueeze(0)  # (B, M, T//2+1)

        # 计算频域相似度矩阵
        # 归一化
        norm = torch.norm(weighted_fft, p=2, dim=2, keepdim=True)  # (B, M, 1)
        x_normalized = weighted_fft / (norm + 1e-8)  # (B, M, T//2+1)

        # 计算余弦相似度
        correlation_matrix = torch.bmm(x_normalized, x_normalized.transpose(1, 2))  # (B, M, M)

        # 应用阈值生成二值mask
        channel_mask = (correlation_matrix > self.threshold).float()  # (B, M, M)

        # 添加维度以匹配注意力机制的输入格式
        channel_mask = channel_mask.unsqueeze(1)  # (B, 1, M, M)

        return channel_mask


# ============================================
# Fusion模块（多头注意力融合）
# ============================================

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (B, N, D)
            key: (B, N, D)
            value: (B, N, D)
            mask: (B, 1, N, N) or (B, N, N)

        Returns:
            output: (B, N, D)
            attention: (B, H, N, N)
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, N, d_k)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, N, N)

        # Apply mask
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, 1, N, N)
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = torch.softmax(scores, dim=-1)  # (B, H, N, N)
        attention = self.dropout(attention)

        # Apply attention to values
        output = torch.matmul(attention, V)  # (B, H, N, d_k)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # (B, N, D)

        # Final linear projection
        output = self.w_o(output)

        return output, attention


class FeedForward(nn.Module):
    """前馈神经网络"""

    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1, activation: str = 'relu'):
        super(FeedForward, self).__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_ff: int = None,
            dropout: float = 0.1,
            activation: str = 'relu'
    ):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, N, D)
            mask: (B, 1, N, N) or (B, N, N)

        Returns:
            x: (B, N, D)
            attention: (B, H, N, N)
        """
        # Multi-head attention with residual
        attn_output, attention = self.attention(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x, attention


class FusionModule(nn.Module):
    """
    融合模块：融合Local、Global特征和CCM的通道关系
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int = 4,
            d_ff: int = None,
            n_layers: int = 2,
            dropout: float = 0.1,
            activation: str = 'relu',
            fusion_strategy: str = 'concat'  # 'concat', 'add', 'gated'
    ):
        """
        Args:
            d_model: 特征维度
            n_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            n_layers: 编码器层数
            dropout: Dropout率
            activation: 激活函数
            fusion_strategy: 融合策略
        """
        super(FusionModule, self).__init__()

        self.d_model = d_model
        self.fusion_strategy = fusion_strategy

        # 根据融合策略调整输入维度
        if fusion_strategy == 'concat':
            input_dim = 2 * d_model  # local + global
            self.fusion_layer = nn.Linear(input_dim, d_model)
        elif fusion_strategy == 'gated':
            self.gate_global = nn.Linear(d_model, d_model)
            self.gate_local = nn.Linear(d_model, d_model)
            self.fusion_layer = nn.Linear(d_model, d_model)
        # 'add' 策略不需要额外的层

        # Transformer编码器层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, activation)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

    def forward(
            self,
            global_features: torch.Tensor,
            local_features: torch.Tensor,
            channel_mask: torch.Tensor
    ):
        """
        Args:
            global_features: (B, M, D)
            local_features: (B, M, D)
            channel_mask: (B, 1, M, M)

        Returns:
            fused_features: (B, M, D)
            attention_weights: List of attention weights
        """
        # 融合策略
        if self.fusion_strategy == 'concat':
            x = torch.cat([global_features, local_features], dim=-1)  # (B, M, 2D)
            x = self.fusion_layer(x)  # (B, M, D)
        elif self.fusion_strategy == 'add':
            x = global_features + local_features  # (B, M, D)
        elif self.fusion_strategy == 'gated':
            gate = torch.sigmoid(self.gate_global(global_features) + self.gate_local(local_features))
            x = gate * global_features + (1 - gate) * local_features
            x = self.fusion_layer(x)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        # Transformer编码器
        attention_weights = []
        for layer in self.encoder_layers:
            x, attn = layer(x, channel_mask)
            attention_weights.append(attn)

        x = self.final_norm(x)

        return x, attention_weights


# ============================================
# 图神经网络模块（与EADRO保持一致）
# ============================================

class GraphModel(nn.Module):
    """图注意力网络"""

    def __init__(self, in_dim, graph_hiddens=[64], device='cpu', attn_head=4, activation=0.2, **kwargs):
        super(GraphModel, self).__init__()

        layers = []
        for i, hidden in enumerate(graph_hiddens):
            in_feats = graph_hiddens[i - 1] if i > 0 else in_dim
            dropout = kwargs.get("attn_drop", 0)
            layers.append(
                GATv2Conv(
                    in_feats,
                    out_feats=hidden,
                    num_heads=attn_head,
                    attn_drop=dropout,
                    negative_slope=activation,
                    allow_zero_in_degree=True
                )
            )
            self.maxpool = nn.MaxPool1d(attn_head)

        self.net = nn.Sequential(*layers).to(device)
        self.out_dim = graph_hiddens[-1]
        self.pooling = GlobalAttentionPooling(nn.Linear(self.out_dim, 1))

    def forward(self, graph, x):
        """
        Args:
            graph: DGL图对象
            x: 节点特征 (batch_size*node_num, feature_dim)

        Returns:
            graph_embedding: (batch_size, out_dim)
        """
        out = None
        for layer in self.net:
            if out is None:
                out = x
            out = layer(graph, out)
            out = self.maxpool(out.permute(0, 2, 1)).permute(0, 2, 1).squeeze()

        return self.pooling(graph, out)


# ============================================
# 多源编码器（改进版）
# ============================================

class ImprovedMultiSourceEncoder(nn.Module):
    """
    改进的多源编码器
    只使用metrics数据，通过双TCN + CCM + Fusion提取特征
    """

    def __init__(
            self,
            metric_num: int,
            node_num: int,
            seq_length: int,
            device: str,
            tcn_output_dim: int = 64,
            fusion_dim: int = 64,
            **kwargs
    ):
        super(ImprovedMultiSourceEncoder, self).__init__()

        self.node_num = node_num
        self.metric_num = metric_num
        self.seq_length = seq_length

        # 双TCN特征提取
        self.dual_tcn = DualTCNFeatureExtractor(
            metric_num=metric_num,
            seq_length=seq_length,
            tcn_output_dim=tcn_output_dim,
            kernel_size=kwargs.get('tcn_kernel_size', 3),
            dropout=kwargs.get('tcn_dropout', 0.2),
            use_batch_norm=kwargs.get('use_batch_norm', True),
            residual_connection=kwargs.get('residual_connection', False)
        )

        # CCM模块
        self.ccm = CCMModule(
            metric_num=metric_num,
            seq_length=seq_length,
            correlation_threshold=kwargs.get('correlation_threshold', 0.4),
            use_learnable_threshold=kwargs.get('use_learnable_threshold', True)
        )

        # Fusion模块
        self.fusion = FusionModule(
            d_model=tcn_output_dim,
            n_heads=kwargs.get('fusion_heads', 4),
            d_ff=kwargs.get('fusion_ff_dim', None),
            n_layers=kwargs.get('fusion_layers', 2),
            dropout=kwargs.get('fusion_dropout', 0.1),
            activation=kwargs.get('fusion_activation', 'relu'),
            fusion_strategy=kwargs.get('fusion_strategy', 'concat')
        )

        # 节点级别的特征聚合
        # 将(B*N, M, D)的特征聚合为(B*N, D)
        self.node_aggregation = nn.Sequential(
            nn.Linear(tcn_output_dim * metric_num, fusion_dim),
            nn.ReLU(),
            nn.Dropout(kwargs.get('aggregation_dropout', 0.1))
        )

        self.feat_in_dim = fusion_dim

        # 图神经网络
        self.status_model = GraphModel(
            in_dim=self.feat_in_dim,
            device=device,
            **kwargs
        )

        self.feat_out_dim = self.status_model.out_dim

    def forward(self, graph):
        """
        前向传播（并行架构）

        Args:
            graph: DGL图对象，包含节点数据
                   graph.ndata["metrics"]: (batch_size*node_num, seq_length, metric_num)

        Returns:
            embeddings: (batch_size, feat_out_dim)
        """
        # 获取metrics数据
        metrics = graph.ndata["metrics"]  # (B*N, T, M)
        batch_node_num = metrics.size(0)

        # ========== 并行处理阶段 ==========
        # Step 1: 双TCN特征提取（并行，都处理原始metrics）
        local_features, global_features = self.dual_tcn(metrics)  # (B*N, M, D), (B*N, M, D)

        # Step 2: CCM生成通道关系矩阵（并行，也处理原始metrics）
        channel_mask = self.ccm(metrics)  # (B*N, 1, M, M)

        # ========== 融合阶段 ==========
        # Step 3: Fusion融合（Q, K, V形式的不变多头注意力）
        # 根据架构图，这里应该是Invariant Multi-Head Attention
        fused_features, _ = self.fusion(global_features, local_features, channel_mask)  # (B*N, M, D)

        # Step 4: 节点级别特征聚合
        # 将(B*N, M, D)展平为(B*N, M*D)然后聚合为(B*N, fusion_dim)
        fused_features_flat = fused_features.reshape(batch_node_num, -1)  # (B*N, M*D)
        node_features = self.node_aggregation(fused_features_flat)  # (B*N, fusion_dim)

        # ========== 图处理阶段 ==========
        # Step 5: Dynamic GAT图神经网络处理
        embeddings = self.status_model(graph, node_features)  # (B, graph_dim)

        return embeddings


# ============================================
# 全连接分类器（与EADRO保持一致）
# ============================================

class FullyConnected(nn.Module):
    """全连接网络"""

    def __init__(self, in_dim, out_dim, linear_sizes):
        super(FullyConnected, self).__init__()
        layers = []
        for i, hidden in enumerate(linear_sizes):
            input_size = in_dim if i == 0 else linear_sizes[i - 1]
            layers += [nn.Linear(input_size, hidden), nn.ReLU()]
        layers += [nn.Linear(linear_sizes[-1], out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.net(x)


# ============================================
# 主模型（改进版）
# ============================================

class ImprovedMainModel(nn.Module):
    """
    改进的主模型
    基于EADRO但使用新的特征提取方式
    """

    def __init__(
            self,
            metric_num: int,
            node_num: int,
            seq_length: int,
            device: str,
            alpha: float = 0.5,
            **kwargs
    ):
        super(ImprovedMainModel, self).__init__()

        self.device = device
        self.node_num = node_num
        self.alpha = alpha

        # 改进的编码器
        self.encoder = ImprovedMultiSourceEncoder(
            metric_num=metric_num,
            node_num=node_num,
            seq_length=seq_length,
            device=device,
            **kwargs
        )

        # 异常检测器
        self.detecter = FullyConnected(
            self.encoder.feat_out_dim,
            2,
            kwargs.get('detect_hiddens', [64])
        ).to(device)
        self.detecter_criterion = nn.CrossEntropyLoss()

        # 故障定位器
        self.localizer = FullyConnected(
            self.encoder.feat_out_dim,
            node_num,
            kwargs.get('locate_hiddens', [64])
        ).to(device)
        self.localizer_criterion = nn.CrossEntropyLoss(ignore_index=0)

        # Softmax用于生成概率
        self.get_prob = nn.Softmax(dim=-1)

    def forward(self, graph, fault_indexs):
        """
        前向传播

        Args:
            graph: DGL图对象
            fault_indexs: 故障节点索引 (batch_size,)
                         0表示正常，>0表示故障节点ID

        Returns:
            结果字典，包含loss、y_pred、y_prob、pred_prob
        """
        batch_size = graph.batch_size
        embeddings = self.encoder(graph)  # (B, feat_out_dim)

        # 构造标签
        # y_prob: 真实故障节点的one-hot表示
        y_prob = torch.zeros((batch_size, self.node_num)).to(self.device)
        for i in range(batch_size):
            if fault_indexs[i] > 0:  # 只有在存在故障时才标记
                y_prob[i, fault_indexs[i]] = 1

        # y_anomaly: 是否存在异常的二分类标签
        y_anomaly = torch.zeros(batch_size).long().to(self.device)
        for i in range(len(fault_indexs)):
            if fault_indexs[i] > 0:
                y_anomaly[i] = 1
            else:
                y_anomaly[i] = 0

        # 故障定位
        locate_logits = self.localizer(embeddings)
        locate_loss = self.localizer_criterion(locate_logits, fault_indexs.to(self.device))

        # 异常检测
        detect_logits = self.detecter(embeddings)
        detect_loss = self.detecter_criterion(detect_logits, y_anomaly)

        # 联合损失
        loss = self.alpha * detect_loss + (1 - self.alpha) * locate_loss

        # 生成预测结果
        node_probs = self.get_prob(locate_logits.detach()).cpu().numpy()
        y_pred = self.inference(batch_size, node_probs, detect_logits)

        return {
            'loss': loss,
            'y_pred': y_pred,
            'y_prob': y_prob.detach().cpu().numpy(),
            'pred_prob': node_probs
        }

    def inference(self, batch_size, node_probs, detect_logits=None):
        """
        推理阶段

        Args:
            batch_size: 批次大小
            node_probs: 节点概率 (batch_size, node_num)
            detect_logits: 检测器输出 (batch_size, 2)

        Returns:
            y_pred: 预测结果列表
        """
        # 按概率排序节点
        node_list = np.flip(node_probs.argsort(axis=1), axis=1)

        y_pred = []
        for i in range(batch_size):
            detect_pred = detect_logits.detach().cpu().numpy().argmax(axis=1).squeeze()
            if detect_pred[i] < 1:
                y_pred.append([0])  # 预测为正常
            else:
                y_pred.append(node_list[i])  # 返回排序的节点列表

        return y_pred


import numpy as np

# 为了让代码完整，添加numpy导入