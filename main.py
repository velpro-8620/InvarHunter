

from torch.utils.data import Dataset, DataLoader
import torch
import dgl
import os
import logging
from improved_model import ImprovedMainModel
from base import BaseModel


class ImprovedChunkDataset(Dataset):
    """
    改进的数据集类
    只加载metrics数据，不加载logs和traces
    """

    def __init__(self, chunks, node_num, edges):
        self.data = []
        self.idx2id = {}

        for idx, chunk_id in enumerate(chunks.keys()):
            self.idx2id[idx] = chunk_id
            chunk = chunks[chunk_id]

            # 创建DGL图
            graph = dgl.graph(edges, num_nodes=node_num)

            # 只加载metrics数据
            graph.ndata["metrics"] = torch.FloatTensor(chunk["metrics"])

            self.data.append((graph, chunk["culprit"]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __get_chunk_id__(self, idx):
        return self.idx2id[idx]


class ImprovedBaseModel(BaseModel):
    """
    改进的基础模型类
    重写model初始化部分以使用新模型
    """

    def __init__(self, metric_num, node_num, seq_length, device, lr=1e-3,
                 epoches=50, patience=5, result_dir='./', hash_id=None, **kwargs):
        # 不调用父类的__init__，而是重新实现
        import torch.nn as nn
        nn.Module.__init__(self)

        self.epoches = epoches
        self.lr = lr
        self.patience = patience
        self.device = device

        self.model_save_dir = os.path.join(result_dir, hash_id)

        # 使用改进的模型
        self.model = ImprovedMainModel(
            metric_num=metric_num,
            node_num=node_num,
            seq_length=seq_length,
            device=device,
            **kwargs
        )
        self.model.to(device)


def collate(data):
    """批处理函数"""
    graphs, labels = map(list, zip(*data))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


def run_improved_model(evaluation_epoch=5):
    """
    运行改进的模型
    """
    from utils import read_json, load_chunks, dump_params, seed_everything, dump_scores
    import argparse

    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", default=42, type=int)

    ### 训练参数
    parser.add_argument("--cpu", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--epoches", default=100, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--patience", default=10, type=int)

    ### 融合参数
    parser.add_argument("--alpha", default=0.5, type=float)
    parser.add_argument("--locate_hiddens", default=[64], type=int, nargs='+')
    parser.add_argument("--detect_hiddens", default=[64], type=int, nargs='+')

    ### 双TCN参数
    parser.add_argument("--tcn_output_dim", default=64, type=int, help="TCN输出维度")
    parser.add_argument("--tcn_kernel_size", default=3, type=int, help="TCN卷积核大小")
    parser.add_argument("--tcn_dropout", default=0.2, type=float, help="TCN dropout率")
    parser.add_argument("--use_batch_norm", default=True, type=lambda x: x.lower() == "true")
    parser.add_argument("--residual_connection", default=False, type=lambda x: x.lower() == "true")

    ### CCM参数
    parser.add_argument("--correlation_threshold", default=0.4, type=float, help="CCM相关性阈值")
    parser.add_argument("--use_learnable_threshold", default=True, type=lambda x: x.lower() == "true")

    ### Fusion参数
    parser.add_argument("--fusion_dim", default=64, type=int, help="融合后的特征维度")
    parser.add_argument("--fusion_heads", default=4, type=int, help="Fusion注意力头数")
    parser.add_argument("--fusion_layers", default=2, type=int, help="Fusion编码器层数")
    parser.add_argument("--fusion_dropout", default=0.1, type=float, help="Fusion dropout率")
    parser.add_argument("--fusion_activation", default='relu', type=str, choices=['relu', 'gelu'])
    parser.add_argument("--fusion_strategy", default='concat', type=str,
                        choices=['concat', 'add', 'gated'], help="特征融合策略")
    parser.add_argument("--aggregation_dropout", default=0.1, type=float)

    ### 图神经网络参数
    parser.add_argument("--graph_hiddens", default=[64], type=int, nargs='+')
    parser.add_argument("--attn_head", default=4, type=int, help="GAT注意力头数")
    parser.add_argument("--activation", default=0.2, type=float, help="LeakyReLU负斜率")

    ### 数据参数
    parser.add_argument("--data", type=str, default="./SN_chunk")
    parser.add_argument("--result_dir", default="../result/")

    params = vars(parser.parse_args())

    # 设置设备
    def get_device(gpu):
        if gpu and torch.cuda.is_available():
            logging.info("Using GPU...")
            return torch.device("cuda")
        logging.info("Using CPU...")
        return torch.device("cpu")

    # 读取数据
    data_dir = params["data"]
    metadata = read_json(os.path.join(data_dir, "metadata.json"))
    node_num = metadata["node_num"]
    metric_num = metadata["metric_num"]
    edges = metadata["edges"]
    seq_length = metadata["chunk_lenth"]

    edges_tup = tuple(edges)
    params["chunk_lenth"] = seq_length

    # 设置随机种子和设备
    hash_id = dump_params(params)
    params["hash_id"] = hash_id
    seed_everything(params["random_seed"])
    device = get_device(params["cpu"])

    # 加载数据
    train_chunks, test_chunks = load_chunks(data_dir)

    train_data = ImprovedChunkDataset(train_chunks, node_num, edges_tup)
    test_data = ImprovedChunkDataset(test_chunks, node_num, edges_tup)

    train_dl = DataLoader(
        train_data,
        batch_size=params["batch_size"],
        shuffle=True,
        collate_fn=collate,
        pin_memory=True
    )

    test_dl = DataLoader(
        test_data,
        batch_size=params["batch_size"],
        shuffle=False,
        collate_fn=collate,
        pin_memory=True
    )

    logging.info("=" * 80)
    logging.info("Improved Model Training")
    logging.info("=" * 80)
    logging.info(f"Dataset: {data_dir}")
    logging.info(f"Node num: {node_num}, Metric num: {metric_num}, Seq length: {seq_length}")
    logging.info(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    logging.info(f"TCN output dim: {params['tcn_output_dim']}")
    logging.info(f"Fusion strategy: {params['fusion_strategy']}")
    logging.info(f"Fusion heads: {params['fusion_heads']}, layers: {params['fusion_layers']}")
    logging.info("=" * 80)

    # 创建模型
    model = ImprovedBaseModel(
        metric_num=metric_num,
        node_num=node_num,
        seq_length=seq_length,
        device=device,
        **params
    )

    # 训练
    scores, converge = model.fit(train_dl, test_dl, evaluation_epoch=evaluation_epoch)

    # 保存结果
    dump_scores(params["result_dir"], hash_id, scores, converge)
    logging.info(f"Training completed! Hash ID: {hash_id}")

    return scores, converge


if __name__ == "__main__":
    run_improved_model()