#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
法律案件辅助判案服务 - LLM + RAG 集成
功能：
1. 读取案件事实和证据链条
2. 使用 LLM 智能识别案件核心矛盾
3. 调用 RAG 系统检索相关民法典法律法规
4. 结合案件和法律法规，使用 GLM4:9b 生成辅助判案建议
"""

import os
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class LegalAssistantService:
    """法律辅助判案服务"""

    def __init__(
        self,
        llm_api_url: str = "http://104.224.158.247:8007/v1",
        llm_model: str = "glm-4-9b-chat-tool-enabled",
        rag_api_url: str = "http://127.0.0.1:8001"
    ):
        """
        初始化服务

        Args:
            llm_api_url: GLM4 API 地址（远程服务器 vLLM）
            llm_model: 模型名称
            rag_api_url: RAG 系统 API 地址
        """
        self.llm_api_url = llm_api_url
        self.llm_model = llm_model
        self.rag_api_url = rag_api_url

        print("=" * 80)
        print("法律辅助判案服务初始化")
        print("=" * 80)
        print(f"LLM API: {llm_api_url}")
        print(f"LLM 模型: {llm_model}")
        print(f"RAG API: {rag_api_url}")
        print("=" * 80)

    def read_case_materials(self, case_dir: str) -> Dict:
        """
        读取案号文件夹中的案件材料

        Args:
            case_dir: 案号文件夹路径

        Returns:
            包含案件所有材料的字典
        """
        case_dir = Path(case_dir)
        case_number = case_dir.name

        print(f"\n正在读取案件材料: {case_number}")
        print("-" * 80)

        materials = {
            "case_number": case_number,
            "plaintiff_complaint": None,
            "defendant_defense": None,
            "evidence_list": [],
            "template": None
        }

        # 读取起诉状
        plaintiff_files = list(case_dir.glob("*起诉状*.txt"))
        if plaintiff_files:
            with open(plaintiff_files[0], 'r', encoding='utf-8') as f:
                materials["plaintiff_complaint"] = f.read()
                print(f"✓ 起诉状: {plaintiff_files[0].name}")

        # 读取答辩状
        defendant_files = list(case_dir.glob("*答辩状*.txt"))
        if defendant_files:
            with open(defendant_files[0], 'r', encoding='utf-8') as f:
                materials["defendant_defense"] = f.read()
                print(f"✓ 答辩状: {defendant_files[0].name}")

        # 读取判决书模板（如果有）
        template_files = list(case_dir.glob("*判决书*.txt")) + list(case_dir.glob("*模板*.txt"))
        if template_files:
            with open(template_files[0], 'r', encoding='utf-8') as f:
                materials["template"] = f.read()
                print(f"✓ 判决书模板: {template_files[0].name}")

        # 读取证据材料
        proof_dir = case_dir / "proof"
        if proof_dir.exists():
            proof_files = sorted(
                proof_dir.glob("*.txt"),
                key=lambda x: x.name
            )
            for proof_file in proof_files:
                with open(proof_file, 'r', encoding='utf-8') as f:
                    materials["evidence_list"].append({
                        "name": proof_file.stem,
                        "content": f.read()
                    })
            print(f"✓ 证据材料: {len(materials['evidence_list'])} 个文件")

        print("-" * 80)
        return materials

    def extract_case_facts(self, materials: Dict) -> str:
        """
        从案件材料中提取案件事实摘要

        Args:
            materials: 案件材料字典

        Returns:
            案件事实摘要文本
        """
        parts = []

        # 如果有生成的案件事实，优先使用
        case_dir = Path(".")
        generated_facts_files = [
            "判决书_案件事实部分_GLM4生成.txt",
            "判决书_案件事实部分_完整版.txt",
            "判决书_案件事实部分_生成结果.txt"
        ]

        for fact_file in generated_facts_files:
            fact_path = case_dir / materials["case_number"] / fact_file
            if fact_path.exists():
                with open(fact_path, 'r', encoding='utf-8') as f:
                    return f.read()

        # 否则从起诉状和答辩状中提取
        if materials["plaintiff_complaint"]:
            parts.append(f"【原告主张】\n{materials['plaintiff_complaint'][:1500]}")

        if materials["defendant_defense"]:
            parts.append(f"【被告抗辩】\n{materials['defendant_defense'][:1500]}")

        return "\n\n".join(parts)

    def build_evidence_chain(self, materials: Dict) -> str:
        """
        构建证据链摘要

        Args:
            materials: 案件材料字典

        Returns:
            证据链摘要文本
        """
        if not materials["evidence_list"]:
            return "无证据材料"

        evidence_summary = []
        for i, evidence in enumerate(materials["evidence_list"][:20], 1):
            # 提取证据名称和简要内容
            content_preview = evidence["content"][:200].strip()
            evidence_summary.append(f"{i}. {evidence['name']}: {content_preview}...")

        if len(materials["evidence_list"]) > 20:
            evidence_summary.append(f"... 以及其他 {len(materials['evidence_list']) - 20} 个证据")

        return "\n".join(evidence_summary)

    def identify_contradictions_with_llm(self, materials: Dict) -> List[str]:
        """
        使用 LLM 智能识别案件核心矛盾点

        Args:
            materials: 案件材料字典

        Returns:
            矛盾点列表
        """
        print("\n使用 LLM 分析案件矛盾点...")
        print("-" * 80)

        plaintiff = materials.get("plaintiff_complaint", "")
        defendant = materials.get("defendant_defense", "")

        # 构建矛盾识别提示词
        prompt = f"""# 任务：识别案件核心矛盾点

请仔细阅读以下原告起诉状和被告答辩状，识别出双方在事实认定、法律适用、责任承担等方面的核心矛盾和争议焦点。

## 原告起诉状
{plaintiff[:3000]}

## 被告答辩状
{defendant[:3000]}

## 要求

请分析并列出本案的核心矛盾点，每个矛盾点应包括：
1. 矛盾的具体内容（原告主张什么，被告如何抗辩）
2. 矛盾涉及的法律问题或事实争议

请按照以下格式输出（不超过5个核心矛盾）：

1. [矛盾点1的简要描述]
2. [矛盾点2的简要描述]
...

注意：
- 只列出真正的核心矛盾，不要列举所有细节差异
- 描述要简洁、准确、专业
- 聚焦于影响判决结果的关键争议

开始分析："""

        try:
            # 调用 LLM 分析矛盾
            result = self.call_llm(prompt, stream=False)

            if result:
                # 解析返回的矛盾点
                contradictions = []
                lines = result.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    # 匹配序号开头的行
                    if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                        # 去除序号和标点
                        clean_line = line.lstrip('0123456789.-•）) ').strip()
                        if clean_line:
                            contradictions.append(clean_line)

                if contradictions:
                    print(f"✓ 识别到 {len(contradictions)} 个核心矛盾点：")
                    for i, c in enumerate(contradictions, 1):
                        print(f"  {i}. {c[:80]}...")
                    print("-" * 80)
                    return contradictions
                else:
                    print("✗ 未能解析出矛盾点，使用默认分析")
                    return ["原告与被告对案件事实和责任认定存在重大分歧"]
            else:
                print("✗ LLM 矛盾分析失败")
                return ["原告与被告对案件事实和责任认定存在重大分歧"]

        except Exception as e:
            print(f"✗ 矛盾识别异常: {e}")
            return ["原告与被告对案件事实和责任认定存在重大分歧"]

    def query_rag_system(
        self,
        case_facts: str,
        evidence_chain: str,
        contradictions: List[str],
        top_k: int = 5,
        min_score: float = 0.3
    ) -> Tuple[str, List[Dict]]:
        """
        调用 RAG 系统检索相关法律法规

        Args:
            case_facts: 案件事实
            evidence_chain: 证据链
            contradictions: 核心矛盾点
            top_k: 检索数量
            min_score: 最低相似度阈值

        Returns:
            (格式化的法律上下文, 相关法律条文列表)
        """
        print("\n正在调用 RAG 系统检索相关法律法规...")
        print("-" * 80)

        # 结合案件事实和矛盾点构建查询
        query_text = f"{case_facts}\n\n核心争议：\n" + "\n".join(contradictions)

        try:
            response = requests.post(
                f"{self.rag_api_url}/get_context",
                json={
                    "case_facts": query_text,
                    "evidence_chain": evidence_chain,
                    "top_k": top_k,
                    "min_score": min_score
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                context = result.get("context", "")
                relevant_laws = result.get("relevant_laws", [])

                print(f"✓ 检索到 {len(relevant_laws)} 条相关法律法规")
                for i, law in enumerate(relevant_laws[:3], 1):
                    print(f"  {i}. 相关度: {law['score']:.3f} - {law['text'][:50]}...")
                print("-" * 80)

                return context, relevant_laws
            else:
                print(f"✗ RAG 查询失败: HTTP {response.status_code}")
                return "【未能获取相关法律法规】", []

        except Exception as e:
            print(f"✗ RAG 查询异常: {e}")
            return "【未能获取相关法律法规】", []

    def build_judgment_prompt(
        self,
        materials: Dict,
        case_facts: str,
        evidence_chain: str,
        contradictions: List[str],
        legal_context: str
    ) -> str:
        """
        构建辅助判案的提示词

        Args:
            materials: 案件材料
            case_facts: 案件事实
            evidence_chain: 证据链
            contradictions: 矛盾点列表
            legal_context: RAG 检索到的法律上下文

        Returns:
            完整的提示词
        """
        contradictions_text = "\n".join([f"{i}. {c}" for i, c in enumerate(contradictions, 1)])

        prompt = f"""# 法律辅助判案任务

你是一位经验丰富的法官助理，需要根据案件材料和相关法律法规，为法官提供辅助判案建议。

## 案件编号
{materials['case_number']}

## 案件事实
{case_facts}

## 证据链条
{evidence_chain}

## 核心矛盾点（已由AI识别）
{contradictions_text}

{legal_context}

## 任务要求

请根据以上案件事实、证据链条和相关法律法规，完成以下分析：

### 1. 案件性质识别
- 明确本案的案由和法律关系
- 识别适用的主要法律领域（如合同法、侵权法等）

### 2. 争议焦点梳理
- 针对上述核心矛盾点，逐一分析双方的主张和理由
- 识别关键事实认定问题
- 明确法律适用争议

### 3. 证据效力分析
- 评估现有证据对各方主张的支持程度
- 指出证据链条的完整性和缺陷
- 分析举证责任分配

### 4. 法律适用分析
- 结合检索到的民法典条文，分析本案应适用的具体法律规定
- 解释法律条文与案件事实的对应关系
- 分析各项法律构成要件是否满足

### 5. 责任认定建议
- 基于事实和法律，针对每个矛盾点给出分析意见
- 提出可能的判决方向
- 说明判决的法律依据和理由

### 6. 裁判要点提示
- 指出本案判决需要特别注意的法律问题
- 提示可能的法律风险或争议点
- 建议判决书中应重点论述的内容

## 输出格式

请按照以上六个方面，逐一进行专业、客观的法律分析。分析应当：
- 严格依据法律事实和证据
- 充分引用相关法律法规（尤其是上面检索到的民法典条文）
- 逻辑严密，说理充分
- 保持中立、客观的司法立场
- 语言专业、规范

开始分析：
"""
        return prompt

    def call_llm(self, prompt: str, stream: bool = True) -> Optional[str]:
        """
        调用 GLM4:9b 模型生成内容（使用 GLM-4 特殊格式）

        Args:
            prompt: 提示词（将被转换为 GLM-4 格式）
            stream: 是否流式输出

        Returns:
            生成的文本
        """
        print("\n正在调用 GLM4:9b 模型生成辅助判案建议...")
        print("=" * 80)

        # 构造 GLM-4 特殊格式的 prompt
        glm4_prompt = f"""[gMASK]<sop><|system|>
你是一个名为 ChatGLM 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。
<|user|>
{prompt}
<|assistant|>
"""

        url = f"{self.llm_api_url}/completions"
        data = {
            "model": self.llm_model,
            "prompt": glm4_prompt,
            "temperature": 0.2,  # 较低温度，保证输出更稳定和专业
            "top_p": 0.9,
            "max_tokens": 4000,
            "stream": stream,
            "stop": ["<|user|>", "<|endoftext|>"]
        }

        try:
            if stream:
                response = requests.post(url, json=data, stream=True, timeout=600)
                response.raise_for_status()

                generated_text = ""
                print("\n生成内容：")
                print("-" * 80)

                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]
                            if data_str.strip() == '[DONE]':
                                break
                            try:
                                chunk = json.loads(data_str)
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    # GLM-4 使用 text 字段
                                    text = chunk['choices'][0].get('text', '')
                                    generated_text += text
                                    print(text, end='', flush=True)
                            except json.JSONDecodeError:
                                continue

                print("\n" + "-" * 80)
                print("✓ 生成完成")
                print("=" * 80)
                return generated_text
            else:
                response = requests.post(url, json=data, timeout=600)
                response.raise_for_status()
                result = response.json()
                return result['choices'][0]['text']

        except requests.exceptions.RequestException as e:
            print(f"✗ LLM 调用失败: {e}")
            return None

    def save_results(
        self,
        case_dir: str,
        contradictions: List[str],
        prompt: str,
        judgment_advice: str,
        relevant_laws: List[Dict]
    ):
        """
        保存分析结果

        Args:
            case_dir: 案号文件夹
            contradictions: 识别的矛盾点
            prompt: 使用的提示词
            judgment_advice: 生成的判案建议
            relevant_laws: 检索到的相关法律条文
        """
        case_dir = Path(case_dir)

        # 保存判案建议
        advice_file = case_dir / "辅助判案建议_LLM生成.txt"
        with open(advice_file, 'w', encoding='utf-8') as f:
            f.write(judgment_advice)
        print(f"\n✓ 判案建议已保存: {advice_file}")

        # 保存矛盾点分析
        contradictions_file = case_dir / "案件矛盾点分析_LLM识别.txt"
        with open(contradictions_file, 'w', encoding='utf-8') as f:
            f.write("# 案件核心矛盾点（LLM智能识别）\n\n")
            for i, c in enumerate(contradictions, 1):
                f.write(f"{i}. {c}\n\n")
        print(f"✓ 矛盾点分析已保存: {contradictions_file}")

        # 保存提示词
        prompt_file = case_dir / "辅助判案_prompt.txt"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt)
        print(f"✓ 提示词已保存: {prompt_file}")

        # 保存检索到的法律条文
        laws_file = case_dir / "检索到的相关法律法规.json"
        with open(laws_file, 'w', encoding='utf-8') as f:
            json.dump(relevant_laws, f, ensure_ascii=False, indent=2)
        print(f"✓ 相关法律法规已保存: {laws_file}")

    def generate_judgment_assistance(self, case_dir: str) -> Optional[str]:
        """
        生成辅助判案建议的完整流程

        Args:
            case_dir: 案号文件夹路径

        Returns:
            生成的判案建议文本
        """
        print("\n" + "=" * 80)
        print("开始生成辅助判案建议")
        print("=" * 80)

        # 1. 读取案件材料
        materials = self.read_case_materials(case_dir)

        # 2. 提取案件事实
        case_facts = self.extract_case_facts(materials)
        print(f"\n✓ 提取案件事实 ({len(case_facts)} 字符)")

        # 3. 构建证据链
        evidence_chain = self.build_evidence_chain(materials)
        print(f"✓ 构建证据链 ({len(materials['evidence_list'])} 个证据)")

        # 4. 使用 LLM 识别矛盾点
        contradictions = self.identify_contradictions_with_llm(materials)

        # 5. 调用 RAG 检索相关法律
        legal_context, relevant_laws = self.query_rag_system(
            case_facts=case_facts,
            evidence_chain=evidence_chain,
            contradictions=contradictions,
            top_k=5,
            min_score=0.3
        )

        # 6. 构建提示词
        prompt = self.build_judgment_prompt(
            materials=materials,
            case_facts=case_facts,
            evidence_chain=evidence_chain,
            contradictions=contradictions,
            legal_context=legal_context
        )

        # 7. 调用 LLM 生成判案建议
        print("\n正在调用 GLM4:9b 模型生成辅助判案建议...")
        print("=" * 80)
        print("\n生成内容：")
        print("-" * 80)

        judgment_advice = self.call_llm(prompt, stream=True)

        print("\n" + "-" * 80)
        print("✓ 生成完成")
        print("=" * 80)

        if judgment_advice:
            # 8. 保存结果
            self.save_results(
                case_dir=case_dir,
                contradictions=contradictions,
                prompt=prompt,
                judgment_advice=judgment_advice,
                relevant_laws=relevant_laws
            )

            print("\n" + "=" * 80)
            print("✅ 辅助判案建议生成完成！")
            print("=" * 80)

            return judgment_advice
        else:
            print("\n" + "=" * 80)
            print("❌ 生成失败")
            print("=" * 80)
            return None


def main():
    """主函数"""
    import sys

    if len(sys.argv) < 2:
        print("使用方法:")
        print(f"  python3 {sys.argv[0]} <案件目录路径>")
        print("\n示例:")
        print(f"  python3 {sys.argv[0]} ./31774")
        print("\n功能说明:")
        print("  1. 使用 LLM 智能识别案件核心矛盾点")
        print("  2. 调用 RAG 系统检索相关民法典条文")
        print("  3. 生成专业的辅助判案建议")
        print("\n可选环境变量:")
        print("  LLM_API_URL - GLM4 API 地址 (默认: http://localhost:8000/v1)")
        print("  LLM_MODEL - 模型名称 (默认: glm4:9b)")
        print("  RAG_API_URL - RAG API 地址 (默认: http://localhost:8000)")
        sys.exit(1)

    case_dir = sys.argv[1]

    if not os.path.exists(case_dir):
        print(f"错误: 目录不存在 - {case_dir}")
        sys.exit(1)

    # 从环境变量读取配置（如果没有设置则使用远程服务器）
    llm_api_url = os.getenv("LLM_API_URL", "http://104.224.158.247:8007/v1")
    llm_model = os.getenv("LLM_MODEL", "glm-4-9b-chat-tool-enabled")
    rag_api_url = os.getenv("RAG_API_URL", "http://127.0.0.1:8001")

    # 创建服务并运行
    service = LegalAssistantService(
        llm_api_url=llm_api_url,
        llm_model=llm_model,
        rag_api_url=rag_api_url
    )

    service.generate_judgment_assistance(case_dir)


if __name__ == "__main__":
    main()
