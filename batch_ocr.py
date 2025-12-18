#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量 PDF OCR 处理脚本
用于法律文书的 OCR 识别和文本提取

适用于：起诉状、答辩状、判决书模板、证据材料等
"""

import os
import json
from paddleocr import PaddleOCR
import fitz  # PyMuPDF
from pathlib import Path


class PDFProcessor:
    """PDF 文本提取处理器"""

    def __init__(self):
        # 初始化 PaddleOCR（懒加载，只在需要时创建）
        self.ocr = None

    def _init_ocr(self):
        """延迟初始化 OCR"""
        if self.ocr is None:
            print("正在初始化 PaddleOCR...")
            self.ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)

    def extract_text_from_pdf(self, pdf_path, use_ocr=True):
        """
        从 PDF 提取文本

        Args:
            pdf_path: PDF 文件路径
            use_ocr: 是否使用 OCR（针对扫描件）

        Returns:
            dict: {
                'file_name': 文件名,
                'total_pages': 总页数,
                'extraction_method': 'direct' or 'ocr',
                'pages': [页面文本列表],
                'full_text': 完整文本
            }
        """
        print(f"\n处理: {os.path.basename(pdf_path)}")

        # 打开 PDF
        doc = fitz.open(pdf_path)
        pages_data = []
        extraction_method = 'direct'
        total_pages = len(doc)  # 保存总页数

        print(f"  共有 {total_pages} 页")

        for page_num in range(total_pages):
            page = doc[page_num]

            # 尝试直接提取文本
            text = page.get_text()

            if text.strip() and len(text.strip()) > 50:
                # 如果能直接提取到足够的文本，说明是原生 PDF
                pages_data.append({
                    'page_num': page_num + 1,
                    'method': 'direct',
                    'text': text.strip()
                })
                print(f"  第 {page_num + 1} 页: 直接提取 ({len(text.strip())} 字符)")
            elif use_ocr:
                # 如果提取不到文本，说明是扫描件，使用 OCR
                extraction_method = 'ocr'
                self._init_ocr()

                print(f"  第 {page_num + 1} 页: OCR 识别中...")

                # 将页面转为图片（提高分辨率）
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("png")

                # OCR 识别
                result = self.ocr.ocr(img_data, cls=True)

                # 提取文本
                page_text = []
                if result and result[0]:
                    for line in result[0]:
                        if line[1]:
                            page_text.append(line[1][0])

                ocr_text = "\n".join(page_text)
                pages_data.append({
                    'page_num': page_num + 1,
                    'method': 'ocr',
                    'text': ocr_text
                })
                print(f"  第 {page_num + 1} 页: OCR 完成 ({len(ocr_text)} 字符)")

        doc.close()

        # 合并所有页面文本
        full_text = "\n\n".join([
            f"===== 第 {p['page_num']} 页 =====\n{p['text']}"
            for p in pages_data
        ])

        return {
            'file_name': os.path.basename(pdf_path),
            'total_pages': total_pages,
            'extraction_method': extraction_method,
            'pages': pages_data,
            'full_text': full_text
        }

    def process_directory(self, case_dir):
        """
        批量处理案件目录下的所有 PDF 文件
        每个 PDF 的 txt 文件保存在相同目录下

        Args:
            case_dir: 案件目录路径

        Returns:
            dict: 处理结果汇总
        """
        case_dir = Path(case_dir)

        print("=" * 80)
        print(f"开始批量处理案件目录: {case_dir}")
        print("=" * 80)

        results = {
            'case_dir': str(case_dir),
            'documents': {},
            'proofs': {},
            'statistics': {
                'total_files': 0,
                'success': 0,
                'failed': 0
            }
        }

        # 处理主文档（起诉状、答辩状、判决书）
        main_docs = list(case_dir.glob("*.pdf"))
        print(f"\n找到 {len(main_docs)} 个主文档")

        for pdf_file in main_docs:
            try:
                result = self.extract_text_from_pdf(str(pdf_file))

                # 保存文本文件到 PDF 相同目录
                txt_file = pdf_file.parent / f"{pdf_file.stem}.txt"
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(result['full_text'])

                results['documents'][pdf_file.name] = {
                    'status': 'success',
                    'output_file': str(txt_file),
                    'pages': result['total_pages'],
                    'method': result['extraction_method']
                }

                results['statistics']['success'] += 1
                print(f"  ✓ 已保存到: {txt_file}")

            except Exception as e:
                print(f"  ✗ 处理失败: {e}")
                results['documents'][pdf_file.name] = {
                    'status': 'failed',
                    'error': str(e)
                }
                results['statistics']['failed'] += 1

            results['statistics']['total_files'] += 1

        # 处理证据材料
        proof_dir = case_dir / "proof"
        if proof_dir.exists():
            proof_docs = list(proof_dir.glob("*.pdf"))
            print(f"\n找到 {len(proof_docs)} 个证据材料")

            for pdf_file in proof_docs:
                try:
                    result = self.extract_text_from_pdf(str(pdf_file))

                    # 保存文本文件到 PDF 相同目录（proof 目录下）
                    txt_file = pdf_file.parent / f"{pdf_file.stem}.txt"
                    with open(txt_file, 'w', encoding='utf-8') as f:
                        f.write(result['full_text'])

                    results['proofs'][pdf_file.name] = {
                        'status': 'success',
                        'output_file': str(txt_file),
                        'pages': result['total_pages'],
                        'method': result['extraction_method']
                    }

                    results['statistics']['success'] += 1
                    print(f"  ✓ 已保存到: {txt_file}")

                except Exception as e:
                    print(f"  ✗ 处理失败: {e}")
                    results['proofs'][pdf_file.name] = {
                        'status': 'failed',
                        'error': str(e)
                    }
                    results['statistics']['failed'] += 1

                results['statistics']['total_files'] += 1

        # 保存处理结果汇总到案件根目录
        summary_file = case_dir / "processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 80)
        print("批量处理完成！")
        print(f"  总文件数: {results['statistics']['total_files']}")
        print(f"  成功: {results['statistics']['success']}")
        print(f"  失败: {results['statistics']['failed']}")
        print(f"  处理汇总: {summary_file}")
        print("=" * 80)

        return results


def main():
    """主函数"""
    import sys

    if len(sys.argv) < 2:
        print("使用方法:")
        print(f"  python3 {sys.argv[0]} <案件目录路径>")
        print("\n示例:")
        print(f"  python3 {sys.argv[0]} /home/titanrtx/lzj/layer/31774")
        sys.exit(1)

    case_dir = sys.argv[1]

    if not os.path.exists(case_dir):
        print(f"错误: 目录不存在 - {case_dir}")
        sys.exit(1)

    # 创建处理器并执行
    processor = PDFProcessor()
    processor.process_directory(case_dir)


if __name__ == "__main__":
    main()
