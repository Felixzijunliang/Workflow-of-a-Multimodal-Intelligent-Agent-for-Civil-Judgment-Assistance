#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF OCR 测试脚本
使用 PaddleOCR + PyMuPDF 提取 PDF 文本
"""

import os
from paddleocr import PaddleOCR
import fitz  # PyMuPDF
from PIL import Image
import io

def extract_text_from_pdf(pdf_path, use_ocr=True):
    """
    从 PDF 提取文本

    Args:
        pdf_path: PDF 文件路径
        use_ocr: 是否使用 OCR（针对扫描件）

    Returns:
        提取的文本内容
    """
    print(f"正在处理: {pdf_path}")

    # 打开 PDF
    doc = fitz.open(pdf_path)
    all_text = []

    # 先尝试直接提取文本
    print(f"PDF 共有 {len(doc)} 页")

    for page_num in range(len(doc)):
        page = doc[page_num]

        # 尝试直接提取文本
        text = page.get_text()

        if text.strip():
            # 如果能直接提取文本，说明是原生 PDF
            print(f"第 {page_num + 1} 页: 直接提取文本 ({len(text)} 字符)")
            all_text.append(f"\n===== 第 {page_num + 1} 页 =====\n{text}")
        elif use_ocr:
            # 如果提取不到文本，说明是扫描件，使用 OCR
            print(f"第 {page_num + 1} 页: 使用 OCR 识别")

            # 将页面转为图片
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2倍缩放提高清晰度
            img_data = pix.tobytes("png")

            # OCR 识别
            ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
            result = ocr.ocr(img_data, cls=True)

            # 提取文本
            page_text = []
            if result and result[0]:
                for line in result[0]:
                    if line[1]:
                        page_text.append(line[1][0])

            ocr_text = "\n".join(page_text)
            print(f"第 {page_num + 1} 页: OCR 识别了 {len(ocr_text)} 字符")
            all_text.append(f"\n===== 第 {page_num + 1} 页（OCR）=====\n{ocr_text}")

    doc.close()
    return "\n".join(all_text)


if __name__ == "__main__":
    # 测试文件路径
    test_pdf = "/home/titanrtx/lzj/layer/31774/起诉状（脱敏）.pdf"

    if os.path.exists(test_pdf):
        print("=" * 60)
        print("开始测试 PDF OCR")
        print("=" * 60)

        # 提取文本
        extracted_text = extract_text_from_pdf(test_pdf)

        # 保存结果
        output_file = test_pdf.replace(".pdf", "_extracted.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(extracted_text)

        print("\n" + "=" * 60)
        print(f"提取完成！结果已保存到: {output_file}")
        print("=" * 60)

        # 显示前 500 个字符
        print("\n提取内容预览（前500字符）:")
        print("-" * 60)
        print(extracted_text[:500])
        print("-" * 60)
    else:
        print(f"错误: 找不到文件 {test_pdf}")
