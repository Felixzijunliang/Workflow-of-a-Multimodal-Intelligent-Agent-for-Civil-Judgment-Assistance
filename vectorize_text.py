"""
文本向量化脚本
用于将法律法规txt文件向量化并存储到Qdrant
"""
import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple
import uuid
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from embedding_model import get_embedding_model


class TextVectorizer:
    def __init__(self, qdrant_path: str = None, qdrant_host: str = None,
                 qdrant_port: int = 6333, collection_name: str = "law_knowledge"):
        """
        初始化文本向量化器

        Args:
            qdrant_path: 本地Qdrant存储路径，如果指定则使用本地存储
            qdrant_host: Qdrant服务器地址（当qdrant_path为None时使用）
            qdrant_port: Qdrant端口
            collection_name: 集合名称
        """
        self.collection_name = collection_name

        # 优先使用本地存储
        if qdrant_path:
            print(f"使用本地Qdrant存储: {qdrant_path}")
            self.client = QdrantClient(path=qdrant_path)
        else:
            print(f"连接到Qdrant: {qdrant_host}:{qdrant_port}")
            self.client = QdrantClient(host=qdrant_host or "localhost", port=qdrant_port)

        # 加载embedding模型
        self.embedder = get_embedding_model()

        # 初始化collection
        self._init_collection()

    def _init_collection(self):
        """初始化或创建Qdrant集合"""
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]

        if self.collection_name not in collection_names:
            print(f"创建新集合: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedder.get_dimension(),
                    distance=Distance.COSINE
                )
            )
        else:
            print(f"集合已存在: {self.collection_name}")

    def read_txt_file(self, file_path: str, encoding: str = "utf-8") -> str:
        """读取txt文件内容"""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            return content
        except UnicodeDecodeError:
            # 尝试其他编码
            for enc in ['gbk', 'gb2312', 'utf-8-sig']:
                try:
                    with open(file_path, 'r', encoding=enc) as f:
                        content = f.read()
                    print(f"使用编码 {enc} 读取文件")
                    return content
                except:
                    continue
            raise ValueError(f"无法读取文件 {file_path}，请检查编码")

    def split_text(self, text: str, chunk_size: int = 500,
                   overlap: int = 50) -> List[str]:
        """
        将长文本分割成小块

        Args:
            text: 输入文本
            chunk_size: 每块的字符数
            overlap: 块之间的重叠字符数

        Returns:
            文本块列表
        """
        # 按段落分割
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            # 如果当前段落本身就很长，需要进一步分割
            if len(para) > chunk_size:
                # 先保存当前chunk
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""

                # 分割长段落
                for i in range(0, len(para), chunk_size - overlap):
                    chunk = para[i:i + chunk_size]
                    if chunk:
                        chunks.append(chunk)
            else:
                # 如果加上这个段落会超过chunk_size
                if len(current_chunk) + len(para) > chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = para
                else:
                    current_chunk = current_chunk + "\n" + para if current_chunk else para

        # 添加最后一个chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def vectorize_and_upload(self, file_path: str, chunk_size: int = 500,
                            overlap: int = 50, metadata: dict = None) -> int:
        """
        向量化文本文件并上传到Qdrant

        Args:
            file_path: 文件路径
            chunk_size: 文本块大小
            overlap: 重叠大小
            metadata: 额外的元数据

        Returns:
            上传的向量数量
        """
        print(f"\n处理文件: {file_path}")

        # 读取文件
        content = self.read_txt_file(file_path)
        print(f"文件大小: {len(content)} 字符")

        # 分割文本
        chunks = self.split_text(content, chunk_size, overlap)
        print(f"分割成 {len(chunks)} 个文本块")

        # 生成向量
        print("正在生成向量...")
        embeddings = self.embedder.encode(chunks, show_progress=True)

        # 准备上传数据
        points = []
        file_name = Path(file_path).name

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())
            payload = {
                "text": chunk,
                "source_file": file_name,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }

            # 添加额外元数据
            if metadata:
                payload.update(metadata)

            points.append(PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=payload
            ))

        # 批量上传
        print(f"上传 {len(points)} 个向量到Qdrant...")
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )

        print(f"✓ 成功上传 {len(points)} 个向量")
        return len(points)

    def vectorize_directory(self, dir_path: str, chunk_size: int = 500,
                           overlap: int = 50, metadata: dict = None) -> Tuple[int, int]:
        """
        向量化目录下所有txt文件

        Args:
            dir_path: 目录路径
            chunk_size: 文本块大小
            overlap: 重叠大小
            metadata: 额外元数据

        Returns:
            (处理的文件数, 上传的向量总数)
        """
        dir_path = Path(dir_path)
        txt_files = list(dir_path.glob("**/*.txt"))

        if not txt_files:
            print(f"警告: 在 {dir_path} 中未找到txt文件")
            return 0, 0

        print(f"\n找到 {len(txt_files)} 个txt文件")
        total_vectors = 0

        for file_path in tqdm(txt_files, desc="处理文件"):
            try:
                vectors_count = self.vectorize_and_upload(
                    str(file_path),
                    chunk_size=chunk_size,
                    overlap=overlap,
                    metadata=metadata
                )
                total_vectors += vectors_count
            except Exception as e:
                print(f"✗ 处理文件失败 {file_path}: {e}")
                continue

        return len(txt_files), total_vectors


def main():
    parser = argparse.ArgumentParser(description="法律文本向量化工具")
    parser.add_argument("path", help="txt文件路径或包含txt文件的目录")
    parser.add_argument("--local-path", default="./qdrant_storage", help="本地Qdrant存储路径（默认使用本地）")
    parser.add_argument("--host", help="Qdrant服务器地址（如果指定则使用远程服务器）")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant端口")
    parser.add_argument("--collection", default="law_knowledge", help="集合名称")
    parser.add_argument("--chunk-size", type=int, default=500, help="文本块大小")
    parser.add_argument("--overlap", type=int, default=50, help="块重叠大小")
    parser.add_argument("--category", help="法律类别（可选元数据）")

    args = parser.parse_args()

    # 准备元数据
    metadata = {}
    if args.category:
        metadata["category"] = args.category

    # 初始化向量化器（优先使用本地存储，除非指定了host）
    vectorizer = TextVectorizer(
        qdrant_path=args.local_path if not args.host else None,
        qdrant_host=args.host,
        qdrant_port=args.port,
        collection_name=args.collection
    )

    # 处理路径
    path = Path(args.path)
    if path.is_file():
        vectorizer.vectorize_and_upload(
            str(path),
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            metadata=metadata
        )
    elif path.is_dir():
        file_count, vector_count = vectorizer.vectorize_directory(
            str(path),
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            metadata=metadata
        )
        print(f"\n总计: 处理 {file_count} 个文件, 生成 {vector_count} 个向量")
    else:
        print(f"错误: 路径不存在: {path}")
        sys.exit(1)

    print("\n向量化完成!")


if __name__ == "__main__":
    main()
