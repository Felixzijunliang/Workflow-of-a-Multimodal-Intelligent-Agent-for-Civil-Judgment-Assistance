#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""简单的批量PDF转TXT脚本"""
import os
import fitz
from pathlib import Path


def pdf_to_txt(pdf_path):
    """将PDF转换为TXT"""
    print(f"转换: {pdf_path.name}")

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    all_text = []
    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text()
        page_header = f"\n===== 第 {page_num + 1} 页 =====\n"
        all_text.append(page_header)
        all_text.append(text)

    doc.close()
    full_text = ''.join(all_text)

    # 保存到同目录，同名txt文件
    txt_path = pdf_path.parent / f"{pdf_path.stem}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(full_text)

    print(f"  ✓ 已保存: {txt_path.name} ({len(full_text)} 字符)")
    return txt_path


def batch_convert(case_dir):
    """批量转换案件目录中的所有PDF"""
    case_dir = Path(case_dir)

    print("=" * 80)
    print(f"批量转换案件目录: {case_dir}")
    print("=" * 80)

    count = 0

    # 转换主目录PDF
    print("\n处理主目录PDF...")
    for pdf_file in case_dir.glob("*.pdf"):
        try:
            pdf_to_txt(pdf_file)
            count += 1
        except Exception as e:
            print(f"  ✗ 失败: {pdf_file.name} - {e}")

    # 转换proof目录
    proof_dir = case_dir / "proof"
    if proof_dir.exists():
        print("\n处理证据材料...")
        for pdf_file in proof_dir.glob("*.pdf"):
            try:
                pdf_to_txt(pdf_file)
                count += 1
            except Exception as e:
                print(f"  ✗ 失败: {pdf_file.name} - {e}")

    # 转换起诉书目录
    qisushu_dir = case_dir / "起诉书"
    if qisushu_dir.exists():
        print("\n处理起诉书...")
        for pdf_file in qisushu_dir.glob("*.pdf"):
            try:
                pdf_to_txt(pdf_file)
                count += 1
            except Exception as e:
                print(f"  ✗ 失败: {pdf_file.name} - {e}")

    print("\n" + "=" * 80)
    print(f"转换完成！共处理 {count} 个文件")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("使用方法: python3 simple_batch_convert.py <案件目录>")
        sys.exit(1)

    batch_convert(sys.argv[1])
