#!/usr/bin/env python3
"""
PDF转TXT脚本（适用于原生PDF，无需OCR）
使用方法: python3 pdf2txt.py <pdf文件路径> [输出txt路径]
"""
import sys
import os

try:
    import fitz  # PyMuPDF
except ImportError:
    print("错误: 缺少PyMuPDF库")
    print("请运行: pip3 install PyMuPDF")
    sys.exit(1)


def pdf_to_txt(pdf_path, output_path=None):
    """
    将PDF转换为TXT

    Args:
        pdf_path: PDF文件路径
        output_path: 输出TXT路径（可选，默认与PDF同名）

    Returns:
        输出文件路径
    """
    # 检查文件是否存在
    if not os.path.exists(pdf_path):
        print(f"错误: 文件不存在 - {pdf_path}")
        sys.exit(1)

    # 确定输出路径
    if output_path is None:
        output_path = os.path.splitext(pdf_path)[0] + '.txt'

    print(f"正在转换: {pdf_path}")
    print(f"输出到: {output_path}")

    # 打开PDF
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"总页数: {total_pages}")

    # 提取所有页面文本
    all_text = []
    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text()

        # 添加页面分隔符
        page_header = f"\n{'='*60}\n第 {page_num + 1} 页 / 共 {total_pages} 页\n{'='*60}\n"
        all_text.append(page_header)
        all_text.append(text)

        # 显示进度
        if (page_num + 1) % 10 == 0 or page_num == 0:
            print(f"  已处理: {page_num + 1}/{total_pages} 页")

    doc.close()

    # 合并文本
    full_text = ''.join(all_text)

    # 保存到文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)

    # 统计信息
    char_count = len(full_text)
    line_count = full_text.count('\n')

    print(f"\n转换完成!")
    print(f"  总字符数: {char_count:,}")
    print(f"  总行数: {line_count:,}")
    print(f"  输出文件: {output_path}")

    return output_path


def main():
    if len(sys.argv) < 2:
        print("使用方法:")
        print(f"  python3 {sys.argv[0]} <PDF文件路径> [输出TXT路径]")
        print("\n示例:")
        print(f"  python3 {sys.argv[0]} 中华人民共和国民法典.pdf")
        print(f"  python3 {sys.argv[0]} input.pdf output.txt")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    pdf_to_txt(pdf_path, output_path)


if __name__ == "__main__":
    main()
