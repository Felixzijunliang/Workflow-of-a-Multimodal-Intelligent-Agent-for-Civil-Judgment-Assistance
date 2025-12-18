"""
RAG系统查询API接口
提供FastAPI接口供外部LLM调用，查询相关法律法规
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from embedding_model import get_embedding_model


# ============= 数据模型 =============

class QueryRequest(BaseModel):
    """查询请求模型"""
    query: str = Field(..., description="查询文本，例如：'合同违约的法律责任'")
    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数量(1-20)")
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="相似度阈值(0-1)")
    filter: Optional[Dict[str, Any]] = Field(default=None, description="过滤条件，例如 {'category': '民法'}")


class SearchResult(BaseModel):
    """单个搜索结果"""
    id: str = Field(..., description="向量ID")
    score: float = Field(..., description="相似度分数")
    text: str = Field(..., description="法律条文内容")
    source_file: str = Field(..., description="来源文件")
    metadata: Dict[str, Any] = Field(..., description="其他元数据")


class QueryResponse(BaseModel):
    """查询响应模型"""
    success: bool = Field(..., description="是否成功")
    query: str = Field(..., description="查询文本")
    results: List[SearchResult] = Field(..., description="搜索结果列表")
    count: int = Field(..., description="返回结果数量")
    timestamp: str = Field(..., description="查询时间")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    qdrant_connected: bool
    model_loaded: bool
    collection_name: str
    vector_count: Optional[int] = None


class RAGContextRequest(BaseModel):
    """获取RAG上下文请求（为LLM生成判决书提供上下文）"""
    case_facts: str = Field(..., description="案件事实描述")
    evidence_chain: Optional[str] = Field(default=None, description="证据链摘要")
    top_k: int = Field(default=5, ge=1, le=10, description="检索的法律条文数量")
    min_score: float = Field(default=0.3, ge=0.0, le=1.0, description="最低相似度")


class RAGContextResponse(BaseModel):
    """RAG上下文响应"""
    success: bool
    context: str = Field(..., description="组织好的法律法规上下文，可直接用于LLM prompt")
    relevant_laws: List[SearchResult] = Field(..., description="相关法律条文列表")
    count: int


# ============= RAG服务类 =============

class RAGService:
    def __init__(self, qdrant_path: str = None, qdrant_host: str = None,
                 qdrant_port: int = 6333, collection_name: str = "law_knowledge"):
        """
        初始化RAG服务

        Args:
            qdrant_path: 本地Qdrant存储路径，如果指定则使用本地存储
            qdrant_host: Qdrant服务器地址（当qdrant_path为None时使用）
            qdrant_port: Qdrant端口
            collection_name: 集合名称
        """
        self.collection_name = collection_name

        # 优先使用本地存储
        if qdrant_path:
            self.qdrant_client = QdrantClient(path=qdrant_path)
            print(f"✓ RAG服务初始化完成")
            print(f"  - Qdrant: 本地存储 ({qdrant_path})")
        else:
            self.qdrant_client = QdrantClient(host=qdrant_host or "localhost", port=qdrant_port)
            print(f"✓ RAG服务初始化完成")
            print(f"  - Qdrant: {qdrant_host}:{qdrant_port}")

        print(f"  - Collection: {collection_name}")
        print(f"  - Embedding模型: BGE-M3")

        self.embedder = get_embedding_model()

    def search(self, query: str, top_k: int = 5,
               score_threshold: float = 0.0,
               filter_dict: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """语义搜索"""
        # 生成查询向量
        query_vector = self.embedder.encode_query(query)

        # 构建过滤条件
        search_filter = None
        if filter_dict:
            conditions = []
            for key, value in filter_dict.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            if conditions:
                search_filter = Filter(must=conditions)

        # 执行搜索
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=search_filter
        )

        # 格式化结果
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.id,
                "score": result.score,
                "text": result.payload.get("text", ""),
                "source_file": result.payload.get("source_file", ""),
                "metadata": result.payload
            })

        return formatted_results

    def get_rag_context(self, case_facts: str, evidence_chain: Optional[str] = None,
                       top_k: int = 5, min_score: float = 0.3) -> tuple:
        """
        为判决书生成获取相关法律法规上下文

        Args:
            case_facts: 案件事实部分
            evidence_chain: 证据链
            top_k: 检索数量
            min_score: 最低相似度

        Returns:
            (context_text, relevant_laws)
        """
        # 构建查询文本
        query_parts = [case_facts]
        if evidence_chain:
            query_parts.append(evidence_chain)
        query_text = "\n".join(query_parts)

        # 搜索相关法律
        relevant_laws = self.search(
            query=query_text,
            top_k=top_k,
            score_threshold=min_score
        )

        # 组织上下文文本
        if not relevant_laws:
            context = "未找到相关法律法规。"
        else:
            context_parts = ["【相关法律法规】\n"]
            for i, law in enumerate(relevant_laws, 1):
                context_parts.append(f"{i}. {law['text']}")
                context_parts.append(f"   (来源: {law['source_file']}, 相关度: {law['score']:.3f})\n")
            context = "\n".join(context_parts)

        return context, relevant_laws

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 检查Qdrant连接
            collections = self.qdrant_client.get_collections()
            qdrant_connected = True

            # 检查集合是否存在
            collection_exists = any(
                col.name == self.collection_name
                for col in collections.collections
            )

            vector_count = None
            if collection_exists:
                info = self.qdrant_client.get_collection(self.collection_name)
                vector_count = info.points_count

            return {
                "status": "healthy" if qdrant_connected and collection_exists else "degraded",
                "qdrant_connected": qdrant_connected,
                "model_loaded": self.embedder is not None,
                "collection_name": self.collection_name,
                "vector_count": vector_count
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "qdrant_connected": False,
                "model_loaded": self.embedder is not None,
                "collection_name": self.collection_name,
                "error": str(e)
            }


# ============= FastAPI应用 =============

app = FastAPI(
    title="法律RAG系统API",
    description="为判决书生成系统提供法律法规检索服务",
    version="1.0.0"
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局RAG服务实例
rag_service: Optional[RAGService] = None


@app.on_event("startup")
async def startup_event():
    """启动时初始化RAG服务"""
    global rag_service
    import os

    # 优先使用本地存储，如果设置了QDRANT_HOST则使用远程
    qdrant_path = os.getenv("QDRANT_PATH", "./qdrant_storage")
    qdrant_host = os.getenv("QDRANT_HOST")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name = os.getenv("COLLECTION_NAME", "law_knowledge")

    rag_service = RAGService(
        qdrant_path=qdrant_path if not qdrant_host else None,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        collection_name=collection_name
    )
    print("✓ RAG API服务启动成功")


@app.get("/", response_model=Dict[str, str])
async def root():
    """根路径"""
    return {
        "message": "法律RAG系统API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG服务未初始化")

    health_status = rag_service.health_check()
    return health_status


@app.post("/search", response_model=QueryResponse)
async def search_laws(request: QueryRequest):
    """
    语义搜索法律法规

    示例请求:
    ```json
    {
        "query": "合同违约的赔偿责任",
        "top_k": 5,
        "score_threshold": 0.3
    }
    ```
    """
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG服务未初始化")

    try:
        results = rag_service.search(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            filter_dict=request.filter
        )

        return QueryResponse(
            success=True,
            query=request.query,
            results=[SearchResult(**r) for r in results],
            count=len(results),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")


@app.post("/get_context", response_model=RAGContextResponse)
async def get_rag_context(request: RAGContextRequest):
    """
    获取RAG上下文（专为判决书生成设计）

    根据案件事实和证据链，检索相关法律法规，
    返回格式化的上下文文本，可直接用于LLM的prompt

    示例请求:
    ```json
    {
        "case_facts": "被告张三于2023年1月签订房屋买卖合同后，未按约定支付房款...",
        "evidence_chain": "1. 房屋买卖合同 2. 银行转账记录 3. 催款通知",
        "top_k": 5,
        "min_score": 0.3
    }
    ```

    示例响应中的context字段可直接添加到LLM的prompt中
    """
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG服务未初始化")

    try:
        context, relevant_laws = rag_service.get_rag_context(
            case_facts=request.case_facts,
            evidence_chain=request.evidence_chain,
            top_k=request.top_k,
            min_score=request.min_score
        )

        return RAGContextResponse(
            success=True,
            context=context,
            relevant_laws=[SearchResult(**law) for law in relevant_laws],
            count=len(relevant_laws)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取上下文失败: {str(e)}")


@app.get("/stats")
async def get_stats():
    """获取系统统计信息"""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG服务未初始化")

    try:
        info = rag_service.qdrant_client.get_collection(rag_service.collection_name)
        return {
            "collection_name": rag_service.collection_name,
            "total_vectors": info.points_count,
            "vector_dimension": info.config.params.vectors.size,
            "distance_metric": info.config.params.vectors.distance
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计失败: {str(e)}")


def start_server(host: str = "127.0.0.1", port: int = 8000):
    """启动服务器（默认只允许本地访问）"""
    print(f"启动RAG API服务器: http://{host}:{port}")
    print(f"API文档: http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="启动RAG API服务（默认仅本地访问）")
    parser.add_argument("--host", default="127.0.0.1", help="服务器地址（默认127.0.0.1仅本地访问）")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")

    args = parser.parse_args()
    start_server(host=args.host, port=args.port)
