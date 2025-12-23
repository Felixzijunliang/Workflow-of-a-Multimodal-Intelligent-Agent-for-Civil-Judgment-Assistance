#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据开庭笔录生成判决书案件事实部分
使用 GLM-4-9B 模型
"""

import os
import json
import requests
import re
from pathlib import Path


class HearingFactsGenerator:
    """基于开庭笔录的案件事实生成器"""

    def __init__(self, api_url="http://104.224.158.247:8007/v1", model="glm-4-9b-chat-tool-enabled"):
        self.api_url = api_url
        self.model = model
        # 参考模板路径（通用民事判决书案件事实撰写模板）
        self.reference_template_path = Path("/home/titanrtx/lzj/lawyer/判决书案件事实部分模板.txt")

    def read_hearing_transcript(self, hearing_file):
        """读取开庭笔录"""
        hearing_file = Path(hearing_file)

        if not hearing_file.exists():
            raise FileNotFoundError(f"开庭笔录文件不存在: {hearing_file}")

        print("=" * 80)
        print(f"正在读取开庭笔录: {hearing_file.name}")
        print("=" * 80)

        with open(hearing_file, 'r', encoding='utf-8') as f:
            content = f.read()

        print(f"✓ 开庭笔录读取完成 ({len(content)} 字符)")
        return content

    def extract_case_facts_from_template(self):
        """读取通用案件事实撰写模板"""
        if not self.reference_template_path.exists():
            raise FileNotFoundError(f"参考模板不存在: {self.reference_template_path}")

        print(f"\n正在读取参考模板: {self.reference_template_path.name}")

        with open(self.reference_template_path, 'r', encoding='utf-8') as f:
            content = f.read()

        print(f"✓ 读取案件事实撰写模板 ({len(content)} 字符)")
        # 直接返回整个模板内容（因为这是一个专门的写作模板）
        return content

    def build_prompt(self, hearing_content, reference_template):
        """构建优化的prompt"""

        # 开庭笔录预览（取前10000字符，确保包含完整的庭审内容）
        hearing_preview = hearing_content[:10000]
        if len(hearing_content) > 10000:
            hearing_preview += "\n\n... [开庭笔录内容过长，已截取前10000字符]"

        prompt = f"""# 任务
你是一位资深法官，需要根据开庭笔录撰写判决书的"案件事实"部分。

# 写作模板

以下是民事判决书"案件事实"部分的标准撰写模板，请严格按照此模板的结构和要求撰写：

{reference_template}

# 开庭笔录

以下是本案的开庭笔录，请仔细阅读并从中提取关键信息：

{hearing_preview}

# 撰写要求

1. **严格遵循模板结构**：按照上述模板的八个部分组织内容
2. **准确提取信息**：从开庭笔录中准确提取各方陈述、证据、争议焦点
3. **客观中立**：使用规范的法律文书语言，客观陈述事实
4. **数据准确**：所有金额、日期、人名、地点必须与笔录一致
5. **逻辑清晰**：按时间顺序和因果关系组织内容
6. **完整覆盖**：
   - 基础法律关系/标的物情况
   - 合同签订及主要约定
   - 履行情况
   - 违约或欠款事实
   - 解除情况（如有）
   - 原告诉讼请求
   - 被告答辩意见
   - 其他相关事实

# 注意事项

- 不要照抄模板中的示例内容，要根据实际开庭笔录的内容填写
- 模板中的[方括号]内容是提示，需要替换为实际内容
- "适用说明"部分不要写入最终输出
- 如果某个部分在笔录中没有对应内容，可以简化或省略

# 输出格式

直接输出案件事实部分的正文，不要包含"案件事实"这个标题，从"一、"开始即可。

开始撰写：
"""
        return prompt

    def wrap_glm4_prompt(self, user_prompt):
        """将用户prompt包装成GLM-4格式"""
        glm4_prompt = "[gMASK]<sop><|user|>\n"
        glm4_prompt += user_prompt
        glm4_prompt += "\n<|assistant|>\n"
        return glm4_prompt

    def generate_with_glm4(self, prompt):
        """调用 GLM-4 API 生成内容"""
        print("\n" + "=" * 80)
        print(f"正在调用 {self.model} 模型生成案件事实部分...")
        print("=" * 80)

        # 包装成GLM-4格式
        glm4_prompt = self.wrap_glm4_prompt(prompt)

        url = f"{self.api_url}/completions"
        data = {
            "model": self.model,
            "prompt": glm4_prompt,
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 6000,
            "stream": True,
            "stop": ["<|user|>", "<|endoftext|>", "九、判决"]  # 添加停止标记防止生成判决理由
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
                        data_str = line_str[6:]
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data_str)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
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
            print(f"错误：调用 GLM-4 API 失败 - {e}")
            return None

    def clean_output(self, content):
        """清理输出内容"""
        # 去除可能的markdown代码块标记
        cleaned_content = content.strip()
        if cleaned_content.startswith('```'):
            lines = cleaned_content.split('\n')
            lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            cleaned_content = '\n'.join(lines)

        return cleaned_content.strip()

    def save_result(self, content, output_file):
        """保存生成结果"""
        cleaned_content = self.clean_output(content)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)

        print(f"\n✓ 结果已保存到: {output_file}")

    def run(self, hearing_file, output_file=None):
        """运行生成流程"""
        hearing_file = Path(hearing_file)

        if output_file is None:
            # 默认输出到开庭笔录同目录
            output_file = hearing_file.parent / f"{hearing_file.stem}_案件事实部分.txt"

        # 1. 读取开庭笔录
        hearing_content = self.read_hearing_transcript(hearing_file)

        # 2. 提取参考模板的案件事实部分
        reference_facts = self.extract_case_facts_from_template()

        # 3. 构建prompt
        print("\n" + "=" * 80)
        print("正在构建 Prompt...")
        print("=" * 80)
        prompt = self.build_prompt(hearing_content, reference_facts)
        print(f"✓ Prompt 构建完成（长度: {len(prompt)} 字符）")

        # 保存prompt（可选）
        prompt_file = hearing_file.parent / f"{hearing_file.stem}_生成用_prompt.txt"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt)
        print(f"✓ Prompt 已保存到: {prompt_file}")

        # 4. 调用GLM-4生成
        result = self.generate_with_glm4(prompt)

        if result:
            # 5. 保存结果
            self.save_result(result, output_file)

            print("\n" + "=" * 80)
            print("✅ 案件事实部分生成成功！")
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
        print(f"  python3 {sys.argv[0]} <开庭笔录文件路径> [输出文件路径]")
        print("\n示例:")
        print(f"  python3 {sys.argv[0]} /path/to/hearing_transcript.txt")
        print(f"  python3 {sys.argv[0]} /path/to/hearing_transcript.txt output.txt")
        print("\n说明:")
        print("  - 开庭笔录文件：必须是txt格式")
        print("  - 输出文件：可选，默认保存在开庭笔录同目录")
        print("  - 参考模板：固定使用 判决书案件事实部分模板.txt（通用民事判决书写作模板）")
        sys.exit(1)

    hearing_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(hearing_file):
        print(f"错误: 开庭笔录文件不存在 - {hearing_file}")
        sys.exit(1)

    # 创建生成器并运行
    generator = HearingFactsGenerator(
        api_url="http://104.224.158.247:8007/v1",
        model="glm-4-9b-chat-tool-enabled"
    )

    generator.run(hearing_file, output_file)


if __name__ == "__main__":
    main()
