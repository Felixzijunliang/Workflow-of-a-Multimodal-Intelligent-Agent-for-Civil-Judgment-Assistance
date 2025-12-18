#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
判决书生成脚本 - GLM4 vLLM 版本
使用 vLLM GLM4-9B 模型生成判决书"案件事实"部分
"""

import os
import json
import requests
from pathlib import Path


class JudgmentGenerator:
    """判决书生成器（vLLM版）"""

    def __init__(self, api_url="http://104.224.158.247:8007/v1", model="glm-4-9b-chat-tool-enabled"):
        self.api_url = api_url
        self.model = model

    def read_case_files(self, case_dir):
        """读取案件所有文件"""
        case_dir = Path(case_dir)

        print("=" * 80)
        print("正在读取案件材料...")
        print("=" * 80)

        # 获取案件编号
        case_number = case_dir.name

        # 查找判决书模板
        template_file = None
        matching_templates = list(case_dir.glob(f"*{case_number}*判决书*.txt")) + \
                           list(case_dir.glob(f"*{case_number}*模板*.txt"))

        if matching_templates:
            template_file = matching_templates[0]
        else:
            template_files = list(case_dir.glob("*判决书*.txt")) + \
                           list(case_dir.glob("*结案文书*.txt")) + \
                           list(case_dir.glob("*模板*.txt"))
            if template_files:
                template_file = template_files[0]

        if not template_file:
            raise FileNotFoundError("未找到判决书模板文件")

        # 查找起诉状和答辩状
        plaintiff_files = list(case_dir.glob("起诉状*.txt")) + list(case_dir.glob("*起诉状*.txt"))
        if not plaintiff_files:
            raise FileNotFoundError("未找到起诉状文件")
        plaintiff_file = plaintiff_files[0]

        defendant_files = list(case_dir.glob("答辩状*.txt")) + list(case_dir.glob("*答辩状*.txt"))
        if not defendant_files:
            raise FileNotFoundError("未找到答辩状文件")
        defendant_file = defendant_files[0]

        # 读取文件
        print(f"✓ 读取起诉状: {plaintiff_file.name}")
        with open(plaintiff_file, 'r', encoding='utf-8') as f:
            plaintiff_content = f.read()

        print(f"✓ 读取答辩状: {defendant_file.name}")
        with open(defendant_file, 'r', encoding='utf-8') as f:
            defendant_content = f.read()

        print(f"✓ 读取判决书模板: {template_file.name}")
        with open(template_file, 'r', encoding='utf-8') as f:
            template_content = f.read()

        # 读取所有证据材料
        proof_dir = case_dir / "proof"
        proofs = []
        if proof_dir.exists():
            proof_files = sorted(proof_dir.glob("证据材料*.txt"),
                               key=lambda x: int(''.join(filter(str.isdigit, x.stem)) or '0'))
            print(f"\n正在读取 {len(proof_files)} 个证据材料...")
            for proof_file in proof_files:
                with open(proof_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    proofs.append({
                        'name': proof_file.stem,
                        'content': content
                    })
            print(f"✓ 已读取 {len(proofs)} 个证据材料")

        return {
            'plaintiff': plaintiff_content,
            'defendant': defendant_content,
            'template': template_content,
            'proofs': proofs
        }

    def build_prompt(self, case_data):
        """构建优化 prompt（方案1：安全配置，适配16K context限制）"""

        # 证据材料 - 每个保留前200字符，取前8个证据（优先最关键的）
        proof_summary = ""
        for i, proof in enumerate(case_data['proofs'][:8], 1):
            content_preview = proof['content'][:200].replace("=====", "").strip()
            proof_summary += f"\n【证据{i}: {proof['name']}】\n{content_preview}\n"
            if len(proof['content']) > 200:
                proof_summary += "...\n"

        if len(case_data['proofs']) > 8:
            proof_summary += f"\n... 以及其他{len(case_data['proofs'])-8}个证据（略）\n"

        prompt = f"""# 任务
你是一位资深法官，需要根据案件材料撰写判决书的"案件事实"部分。请仔细阅读所有材料，包括起诉状、答辩状和关键证据链条，准确还原案件事实。

# 撰写要求
1. **合同背景**：标的物、签订时间、合同主体、期限、租金等核心要素
2. **合同履行情况**：详细记录实际支付情况、履行时间线
3. **违约事实**：明确何时、如何违约，违约的具体表现
4. **原告诉讼请求**：完整陈述原告的诉讼请求及具体金额
5. **被告答辩意见**：全面反映被告的抗辩理由和事实主张
6. **证据支撑**：基于证据材料还原事实，注意证据之间的逻辑关系

# 材料

## 判决书模板（参考文风）
{case_data['template'][:800]}

## 起诉状（核心内容）
{case_data['plaintiff'][:2200]}

## 答辩状（核心内容）
{case_data['defendant'][:1600]}

## 关键证据材料
{proof_summary}

# 注意事项
- 必须基于实际材料撰写，不得凭空捏造
- 注意证据之间的逻辑关系和时间顺序
- 准确引用具体金额、日期、地点等关键信息
- 客观中立地陈述双方主张
- 突出证据对案件事实的支撑作用

# 输出格式
直接输出正文，按以下结构组织：

一、案涉商铺基本情况及合同签订
二、合同履行情况
三、合同违约及纠纷产生
四、原告诉讼请求
五、被告答辩意见
六、其他相关事实

开始撰写：
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

    def run(self, case_dir, output_file=None):
        """运行生成流程"""
        case_dir = Path(case_dir)

        if output_file is None:
            output_file = case_dir / "判决书_案件事实部分_GLM4生成.txt"

        # 1. 读取案件材料
        case_data = self.read_case_files(case_dir)

        # 2. 构建 prompt
        print("\n" + "=" * 80)
        print("正在构建 Prompt...")
        print("=" * 80)
        prompt = self.build_prompt(case_data)
        print(f"✓ Prompt 构建完成（长度: {len(prompt)} 字符）")

        # 保存 prompt
        prompt_file = case_dir / "生成用_prompt_GLM4.txt"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt)
        print(f"✓ Prompt 已保存到: {prompt_file}")

        # 3. 调用模型生成
        result = self.generate_with_vllm(prompt)

        if result:
            # 4. 保存结果
            self.save_result(result, output_file)

            print("\n" + "=" * 80)
            print("✅ 判决书案件事实部分生成成功！")
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
        print(f"  python3 {sys.argv[0]} <案件目录路径> [输出文件路径]")
        print("\n示例:")
        print(f"  python3 {sys.argv[0]} /home/titanrtx/lzj/layer/31774")
        sys.exit(1)

    case_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(case_dir):
        print(f"错误: 目录不存在 - {case_dir}")
        sys.exit(1)

    # 创建生成器并运行
    generator = JudgmentGenerator(
        api_url="http://104.224.158.247:8007/v1",  # 远程服务器
        model="glm-4-9b-chat-tool-enabled"
    )

    generator.run(case_dir, output_file)


if __name__ == "__main__":
    main()
