"""
BGE-M3 Embedding模型加载和使用模块
用于法律文本的向量化
"""
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Union
import numpy as np
import os

# 配置 HuggingFace 镜像源（解决网络问题）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


class BGEEmbedding:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = None):
        """
        初始化BGE-M3 Embedding模型

        Args:
            model_name: 模型名称，默认为 BAAI/bge-m3
            device: 设备选择，None则自动选择GPU/CPU
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"正在加载BGE-M3模型到 {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dimension = 1024  # BGE-M3的向量维度
        print(f"模型加载完成，向量维度: {self.dimension}")

    def encode(self, texts: Union[str, List[str]],
               batch_size: int = 32,
               show_progress: bool = True,
               normalize: bool = True) -> np.ndarray:
        """
        将文本编码为向量

        Args:
            texts: 单个文本或文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度条
            normalize: 是否归一化向量

        Returns:
            numpy数组，形状为 (n_texts, dimension)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )

        return embeddings

    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        为查询文本编码（针对检索优化）

        Args:
            query: 查询文本
            normalize: 是否归一化

        Returns:
            向量数组
        """
        # BGE模型推荐在查询时添加特殊前缀以提升检索效果
        # 但对于中文法律文本，可以不加前缀
        return self.encode(query, show_progress=False, normalize=normalize)[0]

    def get_dimension(self) -> int:
        """返回向量维度"""
        return self.dimension


# 单例模式，避免重复加载模型
_embedding_model = None

def get_embedding_model(model_name: str = "BAAI/bge-m3", device: str = None) -> BGEEmbedding:
    """
    获取全局Embedding模型实例（单例模式）

    Args:
        model_name: 模型名称
        device: 设备选择

    Returns:
        BGEEmbedding实例
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = BGEEmbedding(model_name=model_name, device=device)
    return _embedding_model


if __name__ == "__main__":
    # 测试代码
    print("=" * 50)
    print("测试BGE-M3 Embedding模型")
    print("=" * 50)

    # 初始化模型
    embedder = get_embedding_model()

    # 测试法律文本
    test_texts = [
        "根据《中华人民共和国民法典》第一千一百七十九条规定，侵害他人造成人身损害的，应当赔偿医疗费、护理费、交通费等。",
        "被告未按约定履行合同义务，构成违约，应承担违约责任。",
        "原告提供的证据充分，足以证明其主张。"
    ]

    # 生成向量
    print(f"\n正在对{len(test_texts)}段法律文本进行向量化...")
    embeddings = embedder.encode(test_texts)
    print(f"生成的向量形状: {embeddings.shape}")
    print(f"第一个向量的前10个值: {embeddings[0][:10]}")

    # 测试查询
    query = "合同违约的赔偿责任"
    print(f"\n查询文本: {query}")
    query_vector = embedder.encode_query(query)
    print(f"查询向量形状: {query_vector.shape}")

    # 计算相似度
    from numpy.linalg import norm
    similarities = np.dot(embeddings, query_vector) / (norm(embeddings, axis=1) * norm(query_vector))
    print(f"\n与各文本的相似度:")
    for i, (text, sim) in enumerate(zip(test_texts, similarities)):
        print(f"{i+1}. 相似度: {sim:.4f} - {text[:30]}...")

    print("\n测试完成!")
