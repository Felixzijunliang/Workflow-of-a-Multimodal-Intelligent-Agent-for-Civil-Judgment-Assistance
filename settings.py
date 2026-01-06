import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # 项目根目录
    BASE_DIR: Path = Path(__file__).resolve().parent
    
    # --- LLM 服务配置 ---
    # 远程 LLM 地址
    LLM_API_URL: str = "http://104.224.158.247:8007/v1"
    LLM_MODEL: str = "glm-4-9b-chat-tool-enabled"
    
    # --- RAG 服务配置 ---
    # RAG 服务监听地址 (0.0.0.0 允许外部访问)
    RAG_HOST: str = "0.0.0.0"
    # RAG 服务监听端口
    RAG_PORT: int = 8000
    # 其他服务调用 RAG 时使用的基础 URL
    RAG_API_BASE_URL: str = "http://localhost:8000"
    
    # --- 向量数据库 (Qdrant) ---
    # Qdrant 存储路径 (本地模式). 如果设置了此项，优先使用本地模式
    QDRANT_PATH: str | None = None
    # Qdrant 服务器地址 (服务器模式)
    QDRANT_HOST: str = "localhost"
    # Qdrant 服务器端口
    QDRANT_PORT: int = 6333
    # 集合名称
    COLLECTION_NAME: str = "law_knowledge"
    
    # --- 路径配置 ---
    # 数据根目录 (用于替换 /home/titanrtx/lzj/layer 等硬编码路径)
    # 默认指向项目根目录，可通过环境变量修改为实际数据盘路径
    DATA_ROOT_DIR: Path = BASE_DIR
    
    # 判决书模板文件路径 (相对于 DATA_ROOT_DIR 或绝对路径)
    JUDGMENT_TEMPLATE_PATH: str = "判决书案件事实部分模板.txt"
    
    # 民法典文件路径
    CIVIL_CODE_PDF_PATH: str = "中华人民共和国民法典.pdf"
    
    # --- 其他 ---
    # HuggingFace 镜像站
    HF_ENDPOINT: str = "https://hf-mirror.com"

    # 自动加载 .env 文件
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def get_template_path(self) -> Path:
        """获取模板的绝对路径"""
        path = Path(self.JUDGMENT_TEMPLATE_PATH)
        if path.is_absolute():
            return path
        return self.DATA_ROOT_DIR / path

    def get_civil_code_path(self) -> Path:
        """获取民法典 PDF 的绝对路径"""
        path = Path(self.CIVIL_CODE_PDF_PATH)
        if path.is_absolute():
            return path
        return self.DATA_ROOT_DIR / path

    def get_qdrant_client_args(self) -> dict:
        """获取 QdrantClient 的初始化参数"""
        if self.QDRANT_PATH:
            return {"path": self.QDRANT_PATH}
        return {"host": self.QDRANT_HOST, "port": self.QDRANT_PORT}

settings = Settings()
