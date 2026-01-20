#!/usr/bin/env python3
"""
单个PDF文件OCR转换并加入知识库
"""
import sys
import os
sys.path.insert(0, '/home/titanrtx/lzj/layer')

from batch_ocr import PDFProcessor
from vectorize_text import TextVectorizer

def process_pdf_to_knowledge_base(pdf_path, output_txt_path=None):
    """
    将PDF转换为txt并加入知识库

    Args:
        pdf_path: PDF文件路径
        output_txt_path: 输出txt路径（可选，默认与PDF同目录同名）
    """
    print("=" * 80)
    print("步骤1: PDF OCR识别")
    print("=" * 80)

    # 1. OCR转换
    processor = PDFProcessor()
    result = processor.extract_text_from_pdf(pdf_path, use_ocr=True)

    # 确定输出路径
    if output_txt_path is None:
        output_txt_path = pdf_path.replace('.pdf', '.txt')

    # 保存txt
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(result['full_text'])

    print(f"\n✓ OCR完成！")
    print(f"  输入: {pdf_path}")
    print(f"  输出: {output_txt_path}")
    print(f"  页数: {result['total_pages']}")
    print(f"  方法: {result['extraction_method']}")
    print(f"  字符数: {len(result['full_text'])}")

    # 2. 向量化加入知识库
    print("\n" + "=" * 80)
    print("步骤2: 向量化并加入知识库")
    print("=" * 80)

    vectorizer = TextVectorizer(
        qdrant_path="./qdrant_storage",
        collection_name="law_knowledge"
    )

    vector_count = vectorizer.vectorize_and_upload(
        output_txt_path,
        chunk_size=500,
        overlap=50,
        metadata={"category": "法律法规", "source": "民法典"}
    )

    print("\n" + "=" * 80)
    print("✓ 全部完成！")
    print("=" * 80)
    print(f"PDF文件: {os.path.basename(pdf_path)}")
    print(f"TXT文件: {os.path.basename(output_txt_path)}")
    print(f"生成向量: {vector_count} 个")
    print(f"知识库: law_knowledge")
    print("=" * 80)

    return output_txt_path, vector_count

if __name__ == "__main__":
    pdf_path = "/home/titanrtx/lzj/layer/中华人民共和国民法典.pdf"

    if not os.path.exists(pdf_path):
        print(f"错误: PDF文件不存在 - {pdf_path}")
        sys.exit(1)

    process_pdf_to_knowledge_base(pdf_path)
