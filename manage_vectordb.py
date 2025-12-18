"""
向量数据库管理脚本
提供对Qdrant向量数据库的增删查改操作
"""
import argparse
import sys
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range
)
from embedding_model import get_embedding_model
import json


class VectorDBManager:
    def __init__(self, qdrant_path: str = None, qdrant_host: str = None,
                 qdrant_port: int = 6333, collection_name: str = "law_knowledge"):
        """
        初始化向量数据库管理器

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

        self.embedder = get_embedding_model()

    def list_collections(self):
        """列出所有集合"""
        collections = self.client.get_collections().collections
        print(f"\n共有 {len(collections)} 个集合:")
        for col in collections:
            print(f"  - {col.name}")
        return collections

    def collection_info(self):
        """显示当前集合信息"""
        try:
            info = self.client.get_collection(self.collection_name)
            print(f"\n集合名称: {self.collection_name}")
            print(f"向量数量: {info.points_count}")
            print(f"向量维度: {info.config.params.vectors.size}")
            print(f"距离度量: {info.config.params.vectors.distance}")
            return info
        except Exception as e:
            print(f"错误: 集合不存在或无法访问 - {e}")
            return None

    def create_collection(self, vector_size: int = 1024, force: bool = False):
        """
        创建新集合

        Args:
            vector_size: 向量维度
            force: 是否强制重新创建（会删除已存在的集合）
        """
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]

        if self.collection_name in collection_names:
            if force:
                print(f"删除已存在的集合: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
            else:
                print(f"集合已存在: {self.collection_name}")
                return

        print(f"创建集合: {self.collection_name}")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        print("集合创建成功")

    def delete_collection(self):
        """删除集合"""
        confirm = input(f"确认删除集合 '{self.collection_name}' ? (yes/no): ")
        if confirm.lower() == 'yes':
            self.client.delete_collection(self.collection_name)
            print(f"集合 '{self.collection_name}' 已删除")
        else:
            print("操作已取消")

    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.0,
               filter_dict: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        语义搜索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            score_threshold: 相似度阈值
            filter_dict: 过滤条件，例如 {"source_file": "民法典.txt"}

        Returns:
            搜索结果列表
        """
        print(f"\n查询: {query}")
        print(f"参数: top_k={top_k}, threshold={score_threshold}")

        # 生成查询向量
        query_vector = self.embedder.encode_query(query)

        # 构建过滤条件
        search_filter = None
        if filter_dict:
            conditions = []
            for key, value in filter_dict.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            if conditions:
                search_filter = Filter(must=conditions)
            print(f"过滤条件: {filter_dict}")

        # 执行搜索
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=search_filter
        )

        # 格式化结果
        formatted_results = []
        print(f"\n找到 {len(results)} 个结果:\n")
        for i, result in enumerate(results):
            formatted_result = {
                "id": result.id,
                "score": result.score,
                "text": result.payload.get("text", ""),
                "source_file": result.payload.get("source_file", ""),
                "metadata": result.payload
            }
            formatted_results.append(formatted_result)

            print(f"[{i+1}] 相似度: {result.score:.4f}")
            print(f"    来源: {result.payload.get('source_file', 'unknown')}")
            print(f"    内容: {result.payload.get('text', '')[:100]}...")
            print()

        return formatted_results

    def delete_by_filter(self, filter_dict: Dict):
        """
        根据条件删除向量

        Args:
            filter_dict: 删除条件，例如 {"source_file": "test.txt"}
        """
        print(f"删除条件: {filter_dict}")
        confirm = input("确认删除? (yes/no): ")

        if confirm.lower() != 'yes':
            print("操作已取消")
            return

        conditions = []
        for key, value in filter_dict.items():
            conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
            )

        delete_filter = Filter(must=conditions)

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=delete_filter
        )
        print("删除完成")

    def delete_by_ids(self, point_ids: List[str]):
        """
        根据ID删除向量

        Args:
            point_ids: 向量ID列表
        """
        print(f"删除 {len(point_ids)} 个向量")
        confirm = input("确认删除? (yes/no): ")

        if confirm.lower() != 'yes':
            print("操作已取消")
            return

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=point_ids
        )
        print("删除完成")

    def scroll_points(self, limit: int = 10, offset: Optional[str] = None):
        """
        浏览向量数据

        Args:
            limit: 每页数量
            offset: 偏移ID
        """
        results = self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )

        points, next_offset = results

        print(f"\n共 {len(points)} 个结果:\n")
        for point in points:
            print(f"ID: {point.id}")
            print(f"Payload: {json.dumps(point.payload, ensure_ascii=False, indent=2)}")
            print("-" * 50)

        if next_offset:
            print(f"\n还有更多数据，下一个offset: {next_offset}")

        return points, next_offset


def main():
    parser = argparse.ArgumentParser(description="向量数据库管理工具")
    parser.add_argument("--local-path", default="./qdrant_storage", help="本地Qdrant存储路径（默认使用本地）")
    parser.add_argument("--host", help="Qdrant服务器地址（如果指定则使用远程服务器）")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant端口")
    parser.add_argument("--collection", default="law_knowledge", help="集合名称")

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # 列出集合
    subparsers.add_parser("list", help="列出所有集合")

    # 集合信息
    subparsers.add_parser("info", help="显示集合信息")

    # 创建集合
    create_parser = subparsers.add_parser("create", help="创建集合")
    create_parser.add_argument("--size", type=int, default=1024, help="向量维度")
    create_parser.add_argument("--force", action="store_true", help="强制重新创建")

    # 删除集合
    subparsers.add_parser("delete-collection", help="删除集合")

    # 搜索
    search_parser = subparsers.add_parser("search", help="语义搜索")
    search_parser.add_argument("query", help="查询文本")
    search_parser.add_argument("--top-k", type=int, default=5, help="返回结果数")
    search_parser.add_argument("--threshold", type=float, default=0.0, help="相似度阈值")
    search_parser.add_argument("--filter", help="过滤条件(JSON格式)")

    # 删除向量
    delete_parser = subparsers.add_parser("delete", help="删除向量")
    delete_parser.add_argument("--filter", help="删除条件(JSON格式)")
    delete_parser.add_argument("--ids", nargs="+", help="向量ID列表")

    # 浏览数据
    scroll_parser = subparsers.add_parser("scroll", help="浏览数据")
    scroll_parser.add_argument("--limit", type=int, default=10, help="每页数量")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # 初始化管理器（优先使用本地存储，除非指定了host）
    manager = VectorDBManager(
        qdrant_path=args.local_path if not args.host else None,
        qdrant_host=args.host,
        qdrant_port=args.port,
        collection_name=args.collection
    )

    # 执行命令
    if args.command == "list":
        manager.list_collections()

    elif args.command == "info":
        manager.collection_info()

    elif args.command == "create":
        manager.create_collection(vector_size=args.size, force=args.force)

    elif args.command == "delete-collection":
        manager.delete_collection()

    elif args.command == "search":
        filter_dict = None
        if args.filter:
            filter_dict = json.loads(args.filter)

        results = manager.search(
            query=args.query,
            top_k=args.top_k,
            score_threshold=args.threshold,
            filter_dict=filter_dict
        )

    elif args.command == "delete":
        if args.filter:
            filter_dict = json.loads(args.filter)
            manager.delete_by_filter(filter_dict)
        elif args.ids:
            manager.delete_by_ids(args.ids)
        else:
            print("错误: 必须提供 --filter 或 --ids")
            sys.exit(1)

    elif args.command == "scroll":
        manager.scroll_points(limit=args.limit)


if __name__ == "__main__":
    main()
