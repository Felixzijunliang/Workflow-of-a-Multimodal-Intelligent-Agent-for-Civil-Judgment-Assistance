<p align="center">
  <h1 align="center">⚖️ 民事判决书智能辅助系统</h1>
  <p align="center">
    <strong>基于 OCR + RAG + LLM 的多模态智能代理工作流</strong>
  </p>
  <p align="center">
    <em>Multimodal Intelligent Agent for Civil Judgment Assistance</em>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/LLM-GLM--4--9B-green.svg" alt="GLM-4">
  <img src="https://img.shields.io/badge/Embedding-BGE--M3-orange.svg" alt="BGE-M3">
  <img src="https://img.shields.io/badge/OCR-PaddleOCR-red.svg" alt="PaddleOCR">
  <img src="https://img.shields.io/badge/VectorDB-Qdrant-purple.svg" alt="Qdrant">
</p>

---

## 📖 项目简介

本项目是一个**端到端的民事判决书智能辅助写作系统**，旨在帮助法官和法律从业者提高工作效率。系统通过以下技术栈实现完整的案件处理流程：

- 🔍 **OCR 识别**：自动识别扫描版起诉状、答辩状、证据材料等 PDF 文件
- 📚 **RAG 检索**：基于民法典等法律法规的语义检索增强生成
- 🤖 **LLM 生成**：智能生成判决书案件事实、裁判理由及辅助判案建议

### ✨ 核心功能

| 功能模块 | 描述 |
|---------|------|
| 📄 批量 OCR | 一键处理整个案件目录下的所有 PDF 文件 |
| 🔗 证据链提取 | 自动识别并组织证据材料，构建证据链条 |
| ⚡ 矛盾点识别 | LLM 智能分析原被告双方的核心争议焦点 |
| 📜 法律检索 | 基于 BGE-M3 向量模型检索相关法律条文 |
| ✍️ 判决书生成 | 自动生成判决书"案件事实"等核心内容 |
| 💡 判案建议 | 综合分析案件，提供专业辅助判案意见 |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              案件材料输入                                      │
│          (起诉状 PDF / 答辩状 PDF / 证据材料 PDF / 判决书模板)                    │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OCR 文本提取模块                                      │
│                                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                    │
│   │  PyMuPDF    │───▶│ PaddleOCR   │───▶│  TXT 文本   │                    │
│   │ (原生PDF)   │    │ (扫描件)     │    │   输出      │                    │
│   └─────────────┘    └─────────────┘    └─────────────┘                    │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│      RAG 法律检索系统          │   │       LLM 智能分析             │
│                               │   │                               │
│  ┌─────────┐  ┌─────────┐    │   │  ┌─────────────────────────┐  │
│  │ BGE-M3  │  │ Qdrant  │    │   │  │    GLM-4-9B (vLLM)     │  │
│  │Embedding│─▶│VectorDB │    │   │  │                         │  │
│  └─────────┘  └─────────┘    │   │  │  • 矛盾点智能识别         │  │
│       │            │         │   │  │  • 案件事实生成          │  │
│       ▼            ▼         │   │  │  • 裁判理由撰写          │  │
│  ┌─────────────────────┐    │   │  │  • 判案建议输出          │  │
│  │   民法典知识库        │    │   │  └─────────────────────────┘  │
│  │   (1024维向量)       │    │   │                               │
│  └─────────────────────┘    │   └───────────────────────────────┘
│                               │                   │
└───────────────┬───────────────┘                   │
                │                                   │
                └─────────────┬─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              输出结果                                        │
│                                                                             │
│   📄 判决书_案件事实部分.txt     📋 辅助判案建议.txt     📑 相关法律法规.json     │
│   📝 案件矛盾点分析.txt         🔍 证据链摘要.txt       📊 处理汇总报告.json      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 安装部署

### 环境要求

- Python 3.8+
- CUDA 11.0+（推荐，用于 GPU 加速）
- 24GB+ 显存（运行 BGE-M3 模型和glm4-9b-chat）

### 1. 克隆仓库

```bash
git clone https://github.com/Felixzijunliang/Workflow-of-a-Multimodal-Intelligent-Agent-for-Civil-Judgment-Assistance.git
cd Workflow-of-a-Multimodal-Intelligent-Agent-for-Civil-Judgment-Assistance
```

### 2. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装 RAG 系统依赖
pip install -r requirements.txt

# 安装 OCR 依赖（PaddleOCR）
pip install paddlepaddle paddleocr PyMuPDF
```

### 3. 准备法律知识库

```bash
# 向量化民法典（首次运行会自动下载 BGE-M3 模型）
python vectorize_text.py 中华人民共和国民法典.txt --category "民法"

# 或向量化整个目录
python vectorize_text.py ./legal_docs/ --chunk-size 500 --overlap 50
```

### 4. 启动 RAG API 服务

```bash
# 启动服务（默认端口 8000）
python rag_api.py --host 0.0.0.0 --port 8000

# 访问 API 文档
# http://localhost:8000/docs
```

---

## 🚀 快速开始

### 使用场景一：批量处理案件 PDF

假设你有一个案件目录结构如下：

```
案件31774/
├── 起诉状.pdf
├── 答辩状.pdf
├── 判决书模板.pdf
└── proof/
    ├── 证据材料1.pdf
    ├── 证据材料2.pdf
    └── ...
```

**步骤 1：OCR 批量识别**

```bash
python batch_ocr.py ./案件号
```


**步骤 2：生成判决书案件事实**

```bash
python generate_judgment_glm4.py ./案件号
```

**步骤 3：生成辅助判案建议**

```bash
python llm_service.py ./案件号
```

---

### 使用场景二：API 调用检索法律法规

```python
import requests

# 根据案件事实检索相关法律
response = requests.post(
    "http://localhost:8000/get_context",
    json={
        "case_facts": """
        原告张三与被告李四于2023年1月签订房屋买卖合同，
        约定总价款100万元。被告仅支付首期款30万元后拒绝继续支付...
        """,
        "evidence_chain": "1. 房屋买卖合同 2. 银行转账记录 3. 催款通知",
        "top_k": 5,
        "min_score": 0.3
    }
)

result = response.json()
print(result["context"])  # 格式化的法律条文上下文
```

**输出示例：**

```
【相关法律法规】

1. 第五百七十七条　当事人一方不履行合同义务或者履行合同义务不符合约定的，
   应当承担继续履行、采取补救措施或者赔偿损失等违约责任。
   (来源: 中华人民共和国民法典.txt, 相关度: 0.847)

2. 第五百八十四条　当事人一方不履行合同义务或者履行合同义务不符合约定，
   造成对方损失的，损失赔偿额应当相当于因违约所造成的损失...
   (来源: 中华人民共和国民法典.txt, 相关度: 0.823)
...
```

---

## 📁 项目结构

```
├── batch_ocr.py           # 批量 PDF OCR 处理模块
├── pdf2txt.py             # 单文件 PDF 转文本工具
├── test_ocr.py            # OCR 功能测试脚本
│
├── embedding_model.py     # BGE-M3 向量模型封装
├── vectorize_text.py      # 文本向量化入库工具
├── manage_vectordb.py     # 向量数据库管理工具
│
├── rag_api.py             # RAG 查询 FastAPI 服务
├── test_rag.py            # RAG 功能测试脚本
│
├── generate_judgment_glm4.py  # 判决书生成（GLM-4）
├── llm_service.py         # LLM 辅助判案服务
├── process_mingfa.py      # 民法典 PDF 处理入库
│
├── 中华人民共和国民法典.txt   # 法律知识库源文件
├── requirements.txt       # Python 依赖
└── README.md              # 项目文档
```

---

## 🔧 API 接口文档

启动服务后访问 `http://localhost:8000/docs` 查看完整 Swagger 文档。

### 核心接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/search` | POST | 语义搜索法律条文 |
| `/get_context` | POST | 获取 RAG 上下文（推荐） |
| `/stats` | GET | 查看知识库统计 |

### 请求示例

**语义搜索**

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "合同违约的赔偿责任", "top_k": 5}'
```

**获取 RAG 上下文**

```bash
curl -X POST "http://localhost:8000/get_context" \
  -H "Content-Type: application/json" \
  -d '{
    "case_facts": "被告未按合同约定支付货款...",
    "top_k": 5,
    "min_score": 0.3
  }'
```

---

## ⚙️ 配置说明

### 环境变量

| 变量名 | 默认值 | 描述 |
|--------|--------|------|
| `QDRANT_PATH` | `./qdrant_storage` | 本地向量数据库路径 |
| `QDRANT_HOST` | - | 远程 Qdrant 地址（设置后使用远程） |
| `QDRANT_PORT` | `6333` | Qdrant 端口 |
| `COLLECTION_NAME` | `law_knowledge` | 向量集合名称 |
| `LLM_API_URL` | `你的llm地址` | GLM-4 API 地址 |
| `LLM_MODEL` | `glm-4-9b-chat-tool-enabled` | 模型名称 |
| `RAG_API_URL` | `http://127.0.0.1:8001` | RAG 服务地址 |

### 向量化参数调优

```bash
python vectorize_text.py ./docs \
  --chunk-size 500 \   # 文本块大小（字符）
  --overlap 50 \       # 块间重叠（字符）
  --category "民法"    # 分类标签
```

**建议**：
- 法律条文：`chunk-size=300-500`，保持条文完整性
- 长篇判决书：`chunk-size=800-1000`，保留更多上下文

---

## 🧪 测试

```bash
# 测试 OCR 功能
python test_ocr.py ./sample.pdf

# 测试 RAG 检索
python test_rag.py

# 测试 Embedding 模型
python embedding_model.py
```

---

## 📊 性能指标

| 指标 | 数值 |
|------|------|
| OCR 识别速度 | ~2-3 秒/页（GPU） |
| 向量检索延迟 | <100ms（本地 Qdrant） |
| 判决书生成时间 | 30-60 秒（GLM-4-9B） |
| 向量维度 | 1024（BGE-M3） |
| 支持文档格式 | PDF、TXT |

---

## 🛠️ 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| OCR 引擎 | PaddleOCR | 中文识别准确率高 |
| PDF 解析 | PyMuPDF | 原生 PDF 文本提取 |
| 向量模型 | BGE-M3 | 中文语义理解能力强 |
| 向量数据库 | Qdrant | 高性能本地/分布式部署 |
| API 框架 | FastAPI | 异步高性能 |
| LLM 推理 | vLLM + GLM-4-9B | 高吞吐量推理 |

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

---

## 📄 许可证

本项目仅供学术研究和学习使用。

---

## 📬 联系方式

如有问题或建议，欢迎通过 GitHub Issues 联系。

---

<p align="center">
  <strong>⚖️ 让 AI 助力法律公正 ⚖️</strong>
</p>
