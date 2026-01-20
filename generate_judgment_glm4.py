#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
判决书生成脚本 - GLM4 vLLM 版本
使用 vLLM GLM4-9B 模型从案件事实生成判决和法律意见
"""

import os
import json
import requests
from pathlib import Path

# 导入统一配置
import settings


class JudgmentGenerator:
    """判决书生成器（vLLM版）- 从案件事实生成判决"""

    def __init__(self, api_url=None, model=None):
        self.api_url = api_url or settings.LLM_BASE_URL
        self.model = model or settings.LLM_MODEL

    def read_case_facts_file(self, facts_file):
        """读取案件事实文件（格式：案号_案件事实部分.txt）"""
        facts_file = Path(facts_file)

        if not facts_file.exists():
            raise FileNotFoundError(f"案件事实文件不存在: {facts_file}")

        print("=" * 80)
        print(f"正在读取案件事实文件: {facts_file.name}")
        print("=" * 80)

        # 从文件名提取案号
        case_number = facts_file.stem.replace("_案件事实部分", "")
        print(f"案号: {case_number}")

        # 读取案件事实内容
        with open(facts_file, 'r', encoding='utf-8') as f:
            facts_content = f.read()

        print(f"✓ 案件事实读取完成 ({len(facts_content)} 字符)")

        return {
            'case_number': case_number,
            'facts': facts_content
        }

    def build_prompt(self, case_data):
        """构建生成判决和法律意见的 prompt"""

        prompt = f"""# 角色
你是一位资深法官，需要根据已整理的案件事实，撰写判决书的"本院认为"部分（法律分析和判决理由）。

# 案件事实
以下是本案已经整理好的案件事实部分，包含了原告诉求、被告答辩、第三方答辩等内容：

{case_data['facts']}

# 任务要求
请根据上述案件事实，撰写判决书的"本院认为"部分，包括：

## 1. 法律关系认定
- 明确本案的法律关系性质（如合同关系、侵权关系、继承关系等）
- 确定适用的法律依据

## 2. 争议焦点分析
- 归纳双方的主要争议焦点
- 逐一分析各方的主张和理由

## 3. 证据采信与事实认定
- 对双方提交的证据进行分析和采信
- 基于证据认定案件事实
- 对有争议的事实进行判断

## 4. 法律适用与裁判理由
- 引用相关法律条文
- 阐述法律适用的理由
- 说明为何支持或驳回各方诉求
- 对于部分支持的，说明计算依据和理由

## 5. 诉讼费用承担
- 根据案件结果确定诉讼费用的承担方式
- 说明费用承担的法律依据

# 写作要求
1. **法律语言规范**：使用专业、严谨的法律文书语言
2. **逻辑严密**：论证过程清晰，说理充分
3. **引用准确**：准确引用相关法律条文（如《中华人民共和国民法典》等）
4. **客观公正**：保持中立立场，依法裁判
5. **结论明确**：对原告的每项诉讼请求都要有明确的支持或驳回意见

# 输出格式
直接输出"本院认为"部分的正文，格式如下：

本院认为，[法律关系认定]...

关于[争议焦点一]，[分析论证]...根据《中华人民共和国民法典》第XXX条规定，[法律适用]...

关于[争议焦点二]，[分析论证]...

综上所述，[总结性意见]。原告的诉讼请求，[支持/部分支持/驳回]。

关于诉讼费用，根据《中华人民共和国民事诉讼法》的相关规定，[费用承担方案]。

# 注意事项
- 不要重复案件事实部分的内容，直接进行法律分析
- 必须引用具体的法律条文
- 对于金额计算要说明依据和计算过程
- 保持客观中立，不偏袒任何一方
- 如果案件事实中提到了第三方，也要在分析中涉及

请开始撰写"本院认为"部分：
"""
        return prompt

    def wrap_glm4_prompt(self, user_prompt):
        """将用户prompt包装成GLM-4格式"""
        glm4_prompt = "[gMASK]<sop><|user|>\n"
        glm4_prompt += user_prompt
        glm4_prompt += "\n<|assistant|>\n"
        return glm4_prompt

    def generate_with_vllm(self, prompt):
        """调用 vLLM Completions API 生成内容（GLM-4格式）"""
        print("\n" + "=" * 80)
        print(f"正在调用 {self.model} 模型生成判决书...")
        print("=" * 80)

        # 将prompt包装成GLM-4格式
        glm4_prompt = self.wrap_glm4_prompt(prompt)

        url = f"{self.api_url}/completions"
        data = {
            "model": self.model,
            "prompt": glm4_prompt,
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 6000,
            "stream": True,
            "stop": ["<|user|>", "<|endoftext|>"]
        }

        try:
            response = requests.post(url, json=data, stream=True, timeout=600)
            response.raise_for_status()

            generated_text = ""
            print("\n生成进度：\n")

            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # 去掉 'data: ' 前缀
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data_str)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                # completions API 使用 'text' 字段而不是 'delta'
                                choice = chunk['choices'][0]
                                if 'text' in choice:
                                    text = choice['text']
                                    generated_text += text
                                    print(text, end='', flush=True)
                        except json.JSONDecodeError:
                            continue

            print("\n\n" + "=" * 80)
            print("生成完成！")
            print("=" * 80)

            return generated_text

        except requests.exceptions.RequestException as e:
            print(f"错误：调用 vLLM API 失败 - {e}")
            return None

    def save_result(self, content, output_file):
        """保存生成结果（去除思考过程）"""
        import re

        # 去除 <think> 标签及其内容
        cleaned_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

        # 去除可能的 markdown 代码块标记
        cleaned_content = cleaned_content.strip()
        if cleaned_content.startswith('```'):
            lines = cleaned_content.split('\n')
            lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            cleaned_content = '\n'.join(lines)

        cleaned_content = cleaned_content.strip()

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        print(f"\n✓ 结果已保存到: {output_file}")

    def run(self, facts_file, output_file=None):
        """运行生成流程"""
        facts_file = Path(facts_file)

        if output_file is None:
            # 默认输出文件名：案号_判决理由.txt
            case_number = facts_file.stem.replace("_案件事实部分", "")
            output_file = facts_file.parent / f"{case_number}_判决理由.txt"

        # 1. 读取案件事实文件
        case_data = self.read_case_facts_file(facts_file)

        # 2. 构建 prompt
        print("\n" + "=" * 80)
        print("正在构建 Prompt...")
        print("=" * 80)
        prompt = self.build_prompt(case_data)
        print(f"✓ Prompt 构建完成（长度: {len(prompt)} 字符）")

        # 保存 prompt
        prompt_file = facts_file.parent / f"{case_data['case_number']}_判决理由_prompt.txt"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt)
        print(f"✓ Prompt 已保存到: {prompt_file}")

        # 3. 调用模型生成
        result = self.generate_with_vllm(prompt)

        if result:
            # 4. 保存结果
            self.save_result(result, output_file)

            print("\n" + "=" * 80)
            print("✅ 判决理由部分生成成功！")
            print("=" * 80)
            print(f"输出文件: {output_file}")
            print(f"Prompt 文件: {prompt_file}")

            return output_file
        else:
            print("\n❌ 生成失败")
            return None


def main():
    """主函数"""
    import sys

    if len(sys.argv) < 2:
        print("使用方法:")
        print(f"  python3 {sys.argv[0]} <案件事实文件路径> [输出文件路径]")
        print("\n说明:")
        print("  案件事实文件格式: 案号_案件事实部分.txt")
        print("  文件应包含: 案件事实、原告诉求、被告答辩、第三方答辩等内容")
        print("\n示例:")
        print(f"  python3 {sys.argv[0]} 7512/7512_案件事实部分.txt")
        print(f"  python3 {sys.argv[0]} 7512/7512_案件事实部分.txt 7512/判决理由.txt")
        print("\n配置说明:")
        print("  配置文件: .env（可从 .env.example 复制）")
        print(f"  当前 LLM API: {settings.LLM_BASE_URL}")
        print(f"  当前模型: {settings.LLM_MODEL}")
        sys.exit(1)

    facts_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(facts_file):
        print(f"错误: 文件不存在 - {facts_file}")
        sys.exit(1)

    # 创建生成器并运行（使用统一配置）
    generator = JudgmentGenerator()
    generator.run(facts_file, output_file)


if __name__ == "__main__":
    main()
