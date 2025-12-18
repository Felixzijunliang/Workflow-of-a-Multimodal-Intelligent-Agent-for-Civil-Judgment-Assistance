# 法律RAG系统使用说明

## 系统概述

这是一个完整的法律判决书辅助写作RAG系统，包含：
- **BGE-M3 Embedding模型**：用于法律文本向量化
- **Qdrant向量数据库**：本地存储法律法规
- **FastAPI接口**：供宿主机LLM调用

## 目录结构

```
lzj/layer/
├── embedding_model.py      # BGE-M3模型加载模块
├── vectorize_text.py        # 文本向量化脚本
├── manage_vectordb.py       # 向量数据库管理脚本
├── rag_api.py              # RAG查询API接口
├── start_rag.sh            # 启动脚本
├── requirements.txt        # Python依赖
├── qdrant_storage/         # Qdrant本地存储（自动创建）
└── README.md               # 本文档
```

## 快速开始

### 1. 安装依赖

```bash
cd ~/lzj/layer
pip3 install -r requirements.txt
```

### 2. 准备法律法规文本

将法律法规txt文件放在某个目录下，例如：
```
~/legal_docs/
├── 民法典.txt
├── 刑法.txt
└── 合同法.txt
```

### 3. 向量化法律文本

```bash
# 向量化单个文件
python3 vectorize_text.py ~/legal_docs/民法典.txt

# 向量化整个目录
python3 vectorize_text.py ~/legal_docs/

# 添加分类标签
python3 vectorize_text.py ~/legal_docs/ --category "民法"

# 自定义参数
python3 vectorize_text.py ~/legal_docs/ --chunk-size 500 --overlap 50
```

### 4. 启动RAG API服务

```bash
# 方式1: 使用启动脚本（推荐）
./start_rag.sh

# 方式2: 直接运行（跳过依赖安装）
./start_rag.sh --skip-install

# 方式3: 直接用Python
python3 rag_api.py --host 0.0.0.0 --port 8000
```

启动后访问：
- API文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

## 使用场景

### 场景1：为LLM提供法律上下文

在判决书生成的第二步，LLM需要查询相关法律法规：

```python
import requests

# 准备案件事实
case_facts = """
原告张三与被告李四于2023年1月签订房屋买卖合同，
约定总价款100万元，分三期支付。被告仅支付首期款30万元，
后两期款项均未按约定支付...
"""

# 调用RAG接口获取相关法律
response = requests.post(
    "http://localhost:8000/get_context",
    json={
        "case_facts": case_facts,
        "evidence_chain": "1. 房屋买卖合同 2. 银行转账记录 3. 催款通知",
        "top_k": 5,
        "min_score": 0.3
    }
)

result = response.json()

# 获取格式化的法律上下文（可直接添加到LLM prompt）
legal_context = result["context"]

# 构建完整prompt给LLM
prompt = f"""
【案件事实】
{case_facts}

{legal_context}

【任务】
根据以上案件事实和相关法律法规，撰写裁判理由与结果部分。
"""
```

### 场景2：直接搜索法律法规

```python
import requests

response = requests.post(
    "http://localhost:8000/search",
    json={
        "query": "合同违约的赔偿责任",
        "top_k": 5,
        "score_threshold": 0.3
    }
)

results = response.json()
for item in results["results"]:
    print(f"相似度: {item['score']:.3f}")
    print(f"来源: {item['source_file']}")
    print(f"内容: {item['text']}\n")
```

## 向量数据库管理

### 查看数据库信息

```bash
# 查看所有集合
python3 manage_vectordb.py list

# 查看集合详情
python3 manage_vectordb.py info

# 查看统计信息
python3 manage_vectordb.py scroll --limit 10
```

### 搜索测试

```bash
# 基础搜索
python3 manage_vectordb.py search "合同违约的法律责任"

# 设置返回数量和阈值
python3 manage_vectordb.py search "合同违约" --top-k 10 --threshold 0.5

# 添加过滤条件
python3 manage_vectordb.py search "合同违约" --filter '{"category": "民法"}'
```

### 删除数据

```bash
# 按条件删除
python3 manage_vectordb.py delete --filter '{"source_file": "test.txt"}'

# 按ID删除
python3 manage_vectordb.py delete --ids id1 id2 id3

# 删除整个集合
python3 manage_vectordb.py delete-collection
```

### 创建新集合

```bash
python3 manage_vectordb.py create --size 1024

# 强制重新创建（会删除旧数据）
python3 manage_vectordb.py create --force
```

## API接口说明

### 1. 健康检查
```
GET /health
```

### 2. 语义搜索
```
POST /search
Content-Type: application/json

{
    "query": "合同违约的赔偿责任",
    "top_k": 5,
    "score_threshold": 0.3,
    "filter": {"category": "民法"}  // 可选
}
```

### 3. 获取RAG上下文（推荐）
```
POST /get_context
Content-Type: application/json

{
    "case_facts": "案件事实描述...",
    "evidence_chain": "证据链摘要...",  // 可选
    "top_k": 5,
    "min_score": 0.3
}
```

### 4. 统计信息
```
GET /stats
```

## 环境变量配置

可以通过环境变量自定义配置：

```bash
# 修改存储路径
export QDRANT_PATH="./my_qdrant_storage"

# 使用远程Qdrant（如果需要）
export QDRANT_HOST="192.168.1.100"
export QDRANT_PORT="6333"

# 修改集合名称
export COLLECTION_NAME="my_law_knowledge"

# 启动服务
python3 rag_api.py
```

## 集成到判决书生成系统

在你的 `generate_judgment_glm4.py` 中集成RAG：

```python
import requests

def get_legal_context(case_facts, evidence_chain=None):
    """获取相关法律法规上下文"""
    try:
        response = requests.post(
            "http://localhost:8000/get_context",
            json={
                "case_facts": case_facts,
                "evidence_chain": evidence_chain,
                "top_k": 5,
                "min_score": 0.3
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()["context"]
        else:
            return "【未能获取法律法规】"
    except Exception as e:
        print(f"RAG查询失败: {e}")
        return "【未能获取法律法规】"

# 在生成判决书第二步时使用
case_facts = "..."  # 第一步生成的案件事实
legal_context = get_legal_context(case_facts, evidence_chain)

# 构建prompt
prompt = f"""
{case_facts}

{legal_context}

请根据以上案件事实和相关法律法规，撰写裁判理由与结果。
"""
```

## 常见问题

### Q: BGE-M3模型下载慢怎么办？
A: 第一次运行会自动从HuggingFace下载模型，需要一些时间。可以预先下载或使用镜像源。

### Q: 如何更新法律法规？
A: 直接运行 `vectorize_text.py` 向量化新的txt文件即可，会自动添加到数据库。

### Q: 向量数据库存储在哪里？
A: 默认存储在 `./qdrant_storage/` 目录下，是持久化的本地存储。

### Q: 如何清空数据库重新开始？
A: 删除 `./qdrant_storage/` 目录或使用 `manage_vectordb.py delete-collection`

### Q: API接口支持并发吗？
A: 支持，FastAPI是异步框架，可以处理并发请求。

## 性能优化建议

1. **GPU加速**：如果有GPU，embedding模型会自动使用GPU，大幅提升速度
2. **批量向量化**：一次性向量化多个文件比逐个处理更快
3. **调整chunk_size**：根据法律文本特点调整，通常500-1000字符效果较好
4. **相似度阈值**：根据实际效果调整min_score，建议0.3-0.5之间

## 技术栈

- Python 3.8+
- BGE-M3 (BAAI/bge-m3)
- Qdrant (本地存储)
- FastAPI
- Sentence Transformers
- PyTorch

## 更新日志

- v1.0.0: 初始版本，支持本地Qdrant存储，完整RAG功能
