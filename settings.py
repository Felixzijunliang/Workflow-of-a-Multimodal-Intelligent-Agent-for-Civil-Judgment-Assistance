#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一配置管理模块
所有配置项集中管理，支持环境变量和 .env 文件
"""

import os
from pathlib import Path
from typing import Optional

# 尝试加载 .env 文件（如果 python-dotenv 可用）
try:
    from dotenv import load_dotenv
    # 加载项目根目录的 .env 文件
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ 已加载配置文件: {env_path}")
except ImportError:
    # python-dotenv 未安装，仅使用环境变量
    pass


def get_env(key: str, default: str = None) -> Optional[str]:
    """获取环境变量，支持空字符串转换为 None"""
    value = os.getenv(key, default)
    if value == '':
        return None
    return value


def get_env_bool(key: str, default: bool = False) -> bool:
    """获取布尔类型环境变量"""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


def get_env_int(key: str, default: int) -> int:
    """获取整数类型环境变量"""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


# =============================================================================
# LLM 服务配置
# =============================================================================

# GLM-4 / vLLM API 配置
LLM_BASE_URL = get_env('LLM_BASE_URL', 'http://127.0.0.1:8007/v1')
LLM_MODEL = get_env('LLM_MODEL', 'glm-4-9b-chat-tool-enabled')

# Ollama API 配置（用于 qwen3 等本地模型）
OLLAMA_URL = get_env('OLLAMA_URL', 'http://127.0.0.1:11434')
OLLAMA_MODEL = get_env('OLLAMA_MODEL', 'qwen3:14b')


# =============================================================================
# RAG 服务配置
# =============================================================================

# RAG API 服务地址（统一使用 8000 端口）
RAG_API_HOST = get_env('RAG_API_HOST', '127.0.0.1')
RAG_API_PORT = get_env_int('RAG_API_PORT', 8000)
RAG_BASE_URL = get_env('RAG_BASE_URL', f'http://{RAG_API_HOST}:{RAG_API_PORT}')


# =============================================================================
# Qdrant 向量数据库配置
# =============================================================================

# 本地存储路径（优先使用，设为空字符串则使用远程服务）
QDRANT_PATH = get_env('QDRANT_PATH', './qdrant_storage')

# 远程 Qdrant 服务配置（当 QDRANT_PATH 为空时使用）
QDRANT_HOST = get_env('QDRANT_HOST', 'localhost')
QDRANT_PORT = get_env_int('QDRANT_PORT', 6333)

# 向量集合名称
COLLECTION_NAME = get_env('COLLECTION_NAME', 'law_knowledge')


# =============================================================================
# Embedding 模型配置
# =============================================================================

# HuggingFace 镜像源（解决国内网络问题）
HF_ENDPOINT = get_env('HF_ENDPOINT', 'https://hf-mirror.com')

# Embedding 模型名称
EMBEDDING_MODEL = get_env('EMBEDDING_MODEL', 'BAAI/bge-m3')

# HuggingFace 模型缓存目录（使用项目本地缓存）
HF_HOME = get_env('HF_HOME', str(Path(__file__).parent / '.huggingface'))

# 设置 HuggingFace 环境变量
if HF_ENDPOINT:
    os.environ['HF_ENDPOINT'] = HF_ENDPOINT
if HF_HOME:
    os.environ['HF_HOME'] = HF_HOME
    os.environ['HUGGINGFACE_HUB_CACHE'] = str(Path(HF_HOME) / 'hub')
    os.environ['TRANSFORMERS_CACHE'] = str(Path(HF_HOME) / 'hub')

# 设置离线模式（如果本地有缓存则不联网）
HF_OFFLINE = get_env_bool('HF_OFFLINE', False)
if HF_OFFLINE:
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'


# =============================================================================
# OCR 服务配置（支持容器环境）
# =============================================================================

# OCR 服务地址（如果使用远程 OCR 服务）
# 容器内访问宿主机可用: host.docker.internal 或具体 IP
OCR_SERVICE_URL = get_env('OCR_SERVICE_URL', None)

# PaddleOCR 语言设置
OCR_LANG = get_env('OCR_LANG', 'ch')


# =============================================================================
# 通用配置
# =============================================================================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 调试模式
DEBUG = get_env_bool('DEBUG', False)


# =============================================================================
# 辅助函数
# =============================================================================

def get_qdrant_client_args() -> dict:
    """
    获取 Qdrant 客户端初始化参数
    优先使用本地存储，如果 QDRANT_PATH 为空则使用远程服务
    """
    if QDRANT_PATH:
        return {'path': QDRANT_PATH}
    else:
        return {'host': QDRANT_HOST, 'port': QDRANT_PORT}


def print_config():
    """打印当前配置信息（用于调试）"""
    print("=" * 60)
    print("当前配置信息")
    print("=" * 60)
    print(f"LLM_BASE_URL:    {LLM_BASE_URL}")
    print(f"LLM_MODEL:       {LLM_MODEL}")
    print(f"OLLAMA_URL:      {OLLAMA_URL}")
    print(f"OLLAMA_MODEL:    {OLLAMA_MODEL}")
    print(f"RAG_BASE_URL:    {RAG_BASE_URL}")
    print(f"QDRANT_PATH:     {QDRANT_PATH or '(使用远程服务)'}")
    print(f"QDRANT_HOST:     {QDRANT_HOST}")
    print(f"QDRANT_PORT:     {QDRANT_PORT}")
    print(f"COLLECTION_NAME: {COLLECTION_NAME}")
    print(f"HF_ENDPOINT:     {HF_ENDPOINT}")
    print(f"HF_HOME:         {HF_HOME}")
    print(f"HF_OFFLINE:      {HF_OFFLINE}")
    print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
    print(f"OCR_SERVICE_URL: {OCR_SERVICE_URL or '(使用本地 PaddleOCR)'}")
    print(f"DEBUG:           {DEBUG}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
