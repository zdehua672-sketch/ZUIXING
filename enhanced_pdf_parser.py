"""
增强版 PDF 解析器
==================
整合 LitMind 的 PDF 清洗和章节识别能力，
替代原有的基础 pdf_parser.py。

功能:
  1. 多引擎 PDF 读取（pymupdf / pdfplumber / PyPDF2）
  2. 三层清洗（页码/页眉页脚/噪声行）
  3. 30+ 中英文章节标题正则匹配
  4. 输出结构化 PaperContent

借鉴自 LitMind (https://github.com/meishiwhy/Literature-Mind) 的 litmind-parser 模块。

用法:
    from enhanced_pdf_parser import EnhancedPDFParser

    parser = EnhancedPDFParser()
    content = parser.parse("paper.pdf")

    print(content.sections['abstract'])
    print(content.sections['methods'])
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


# ── 数据模型 ──────────────────────────────────────────────

@dataclass
class PDFSections:
    """论文各章节文本"""
    abstract: str = ""
    introduction: str = ""
    methods: str = ""
    results: str = ""
    discussion: str = ""
    conclusion: str = ""
    references: str = ""
    other: str = ""

    def to_dict(self):
        return asdict(self)

    def __getitem__(self, key):
        return getattr(self, key, "")


@dataclass
class PDFContent:
    """结构化 PDF 内容"""
    source_path: str = ""
    full_text: str = ""           # 清洗后的完整文本
    raw_text: str = ""            # 清洗前的原始文本
    sections: PDFSections = field(default_factory=PDFSections)
    page_count: int = 0
    char_count: int = 0
    parse_success: bool = False
    parse_errors: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        return d


# ── PDF 读取引擎 ──────────────────────────────────────────

def _read_pdf_pymupdf(pdf_path: str) -> tuple:
    """使用 pymupdf 读取 PDF"""
    import fitz
    doc = fitz.open(pdf_path)
    pages = []
    full_text_parts = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text("text")
        pages.append(page_text)
        full_text_parts.append(page_text)
    doc.close()
    return "\n\n".join(full_text_parts), pages, len(pages)


def _read_pdf_fallback(pdf_path: str) -> tuple:
    """备用: pdfplumber → PyPDF2"""
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                text = p.extract_text() or ""
                pages.append(text)
        return "\n\n".join(pages), pages, len(pages)
    except ImportError:
        pass

    try:
        from PyPDF2 import PdfReader
        pages = []
        reader = PdfReader(pdf_path)
        for p in reader.pages:
            text = p.extract_text() or ""
            pages.append(text)
        return "\n\n".join(pages), pages, len(pages)
    except ImportError:
        pass

    raise ImportError(
        "需要安装 PDF 解析库:\n"
        "  pip install pymupdf  (推荐)\n"
        "  或 pip install pdfplumber\n"
        "  或 pip install PyPDF2"
    )


def read_pdf(pdf_path: str) -> tuple:
    """读取 PDF，自动选择可用引擎"""
    try:
        return _read_pdf_pymupdf(pdf_path)
    except ImportError:
        return _read_pdf_fallback(pdf_path)


# ── 文本清洗 ──────────────────────────────────────────────

# 页码模式
_PAGE_NUM_PATTERNS = [
    re.compile(r"^\s*\d+\s*$"),
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),
    re.compile(r"^\s*\d+\s*of\s*\d+\s*$", re.I),
    re.compile(r"^\s*Page\s*\d+\s*$", re.I),
    re.compile(r"^\s*第\s*\d+\s*页\s*$"),
]

# 噪声行模式
_NOISE_PATTERNS = [
    re.compile(r"^\s*doi:\s*10\.\S+", re.I),
    re.compile(r"^\s*(received|accepted|published)\s*:", re.I),
    re.compile(r"^\s*correspondence\s*(to|author)", re.I),
    re.compile(r"^\s*e[- ]?mail\s*:"),
    re.compile(r"^\s*copyright\s.*", re.I),
    re.compile(r"^\s*©\s"),
    re.compile(r"^\s*This\s+article\s+is\s+(an?\s+)?(open\s+access|protected|distributed)", re.I),
    re.compile(r"^\s*Figure\s+\d+[\.\-\s]"),
    re.compile(r"^\s*Table\s+\d+[\.\-\s]"),
    re.compile(r"^\s*Author\s+(contributions|note)", re.I),
    re.compile(r"^\s*Conflict", re.I),
    re.compile(r"^\s*Funding", re.I),
    re.compile(r"^\s*Data\s+availability", re.I),
    re.compile(r"^\s*Supplementary", re.I),
    re.compile(r"^\s*Supporting\s+information", re.I),
    re.compile(r"^\s*Ethics", re.I),
]


def _is_page_number(line: str) -> bool:
    return any(pat.match(line) for pat in _PAGE_NUM_PATTERNS)


def _find_repeated_lines(pages: list, threshold: float = 0.6) -> set:
    """找出在超过 threshold 比例的页面中重复出现的行（页眉/页脚）"""
    if len(pages) < 3:
        return set()
    candidates = []
    for page in pages:
        lines = page.split("\n")
        candidates.extend(lines[:3])
        if len(lines) > 3:
            candidates.extend(lines[-3:])
    counter = Counter(candidates)
    min_count = max(2, int(len(pages) * threshold))
    return {line.strip() for line, count in counter.items()
            if count >= min_count and line.strip()}


def _remove_noise_lines(text: str) -> str:
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        if not line.strip():
            cleaned.append(line)
            continue
        if any(pat.match(line) for pat in _NOISE_PATTERNS):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _deduplicate_paragraphs(text: str) -> str:
    paragraphs = text.split("\n\n")
    deduped = []
    prev = ""
    for para in paragraphs:
        stripped = para.strip()
        if stripped and stripped != prev:
            deduped.append(para)
            prev = stripped
        elif not stripped:
            deduped.append(para)
    return "\n\n".join(deduped)


def clean_text(text: str, pages: list = None) -> str:
    """
    三层清洗 PDF 提取文本

    1. 移除页码
    2. 移除页眉/页脚（跨页重复行）
    3. 移除噪声行（doi/版权/资助声明等）
    4. 去除重复段落
    5. 压缩多余空白
    """
    # 1. 页码
    lines = text.split("\n")
    text = "\n".join(l for l in lines if not _is_page_number(l))

    # 2. 页眉/页脚
    if pages and len(pages) > 2:
        repeated = _find_repeated_lines(pages)
        if repeated:
            blocks = text.split("\n\n")
            cleaned_blocks = []
            for block in blocks:
                block_lines = block.split("\n")
                if len(block_lines) <= 3 and all(
                    l.strip() in repeated for l in block_lines if l.strip()
                ):
                    continue
                cleaned_blocks.append(block)
            text = "\n\n".join(cleaned_blocks)

    # 3. 噪声行
    text = _remove_noise_lines(text)

    # 4. 重复段落
    text = _deduplicate_paragraphs(text)

    # 5. 多余空白
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── 章节识别 ──────────────────────────────────────────────

_NUM = r"(?:\d+\.?\s*|(?:I{1,3}V?|IV|VI{0,3})\.?\s*)?"

# 英文章节标题模式
_SECTION_PATTERNS_EN = {
    "abstract": [
        re.compile(rf"^{_NUM}[Aa]bstract\s*\.?\s*$"),
        re.compile(rf"^{_NUM}[Aa]bstract\.\s+\w"),
    ],
    "introduction": [
        re.compile(rf"^{_NUM}[Ii]ntroduction\s*$"),
        re.compile(rf"^{_NUM}[Ii]ntroduction\s*\.?\s*$"),
    ],
    "methods": [
        re.compile(rf"^{_NUM}[Mm]ethods?\s*$"),
        re.compile(rf"^{_NUM}[Mm]aterials?\s+(?:and\s+)?[Mm]ethods?\s*$"),
        re.compile(rf"^{_NUM}[Mm]ethodology\s*$"),
        re.compile(rf"^{_NUM}[Ee]xperimental\s+(?:setup|procedures?|design|protocol)\s*$"),
        re.compile(rf"^{_NUM}[Pp]articipants?\s+(?:and\s+)?(?:procedures?|methods?)\s*$"),
        re.compile(rf"^{_NUM}[Ss]ubjects?\s+and\s+[Mm]ethods?\s*$"),
    ],
    "results": [
        re.compile(rf"^{_NUM}[Rr]esults?\s*$"),
        re.compile(rf"^{_NUM}[Rr]esults?\s+.*$"),
    ],
    "discussion": [
        re.compile(rf"^{_NUM}[Dd]iscussion\s*$"),
    ],
    "conclusion": [
        re.compile(rf"^{_NUM}[Cc]onclusion[s]?\s*$"),
        re.compile(rf"^{_NUM}[Ss]ummary\s*$"),
        re.compile(rf"^{_NUM}[Cc]oncluding\s+[Rr]emarks?\s*$"),
    ],
    "references": [
        re.compile(r"^[Rr]eferences?\s*$"),
        re.compile(r"^[Bb]ibliography\s*$"),
        re.compile(r"^[Ww]orks\s+[Cc]ited\s*$"),
    ],
}

# 中文章节标题模式
_SECTION_PATTERNS_CN = {
    "abstract": [re.compile(r"^摘\s*要\s*\.?\s*$")],
    "introduction": [re.compile(r"^(?:引\s*言|前\s*言|绪\s*论|概\s*述)\s*\.?\s*$")],
    "methods": [re.compile(r"^(?:研究\s*方法|实验\s*方法|材料\s*与\s*方[法式]|方[法式]|实验\s*设计)\s*\.?\s*$")],
    "results": [re.compile(r"^(?:研究\s*结果|实验\s*结果|结\s*果|分析\s*结果)\s*\.?\s*$")],
    "discussion": [re.compile(r"^(?:讨\s*论|分析与\s*讨论)\s*\.?\s*$")],
    "conclusion": [re.compile(r"^(?:结\s*论|结\s*语|总\s*结)\s*\.?\s*$")],
    "references": [re.compile(r"^(?:参\s*考\s*文\s*献|致\s*谢)\s*\.?\s*$")],
}


def _merge_patterns() -> dict:
    merged = {}
    for key in _SECTION_PATTERNS_EN:
        merged.setdefault(key, []).extend(_SECTION_PATTERNS_EN[key])
    for key in _SECTION_PATTERNS_CN:
        merged.setdefault(key, []).extend(_SECTION_PATTERNS_CN[key])
    return merged


def sectionize(text: str) -> PDFSections:
    """
    将清洗后的全文分割为标准章节

    Returns: PDFSections 对象
    """
    sections = PDFSections()
    lines = text.split("\n")

    if not lines or not text.strip():
        sections.other = text
        return sections

    patterns = _merge_patterns()
    boundaries = {}  # {section_key: (line_idx, is_inline)}

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        for section_key, pats in patterns.items():
            if section_key in boundaries:
                continue
            for pat in pats:
                m = pat.match(stripped)
                if m:
                    is_inline = m.end() < len(stripped)
                    boundaries[section_key] = (i, is_inline)
                    break

    # 按出现顺序排序
    sorted_sections = sorted(boundaries.items(), key=lambda x: x[1][0])

    # 构建区间
    taken_lines = set()
    items = list(sorted_sections)
    for idx, (key, (line_idx, inline)) in enumerate(items):
        # 内容起始行：标题行的下一行
        content_start = line_idx + 1
        # 内容结束行：下一个标题行
        if idx + 1 < len(items):
            content_end = items[idx + 1][1][0]
        else:
            content_end = len(lines)
        content = "\n".join(lines[content_start:content_end]).strip()
        setattr(sections, key, content)
        taken_lines.update(range(line_idx, content_end))

    # 剩余内容归入 other
    remaining = sorted(set(range(len(lines))) - taken_lines)
    if remaining:
        sections.other = "\n".join(lines[i] for i in remaining).strip()

    return sections


# ── 主解析器类 ──────────────────────────────────────────────

class EnhancedPDFParser:
    """
    增强版 PDF 解析器

    用法:
        parser = EnhancedPDFParser()
        content = parser.parse("paper.pdf")
        print(content.sections.abstract)
    """

    def parse(self, pdf_path: str) -> PDFContent:
        """
        解析单篇 PDF

        Parameters
        ----------
        pdf_path : str, PDF 文件路径

        Returns
        -------
        PDFContent, 结构化内容
        """
        content = PDFContent(source_path=pdf_path)

        if not pdf_path or not os.path.exists(pdf_path):
            content.parse_errors.append(f"PDF 文件不存在: {pdf_path}")
            return content

        try:
            # 1. 读取原始文本
            full_text, pages, page_count = read_pdf(pdf_path)
            content.raw_text = full_text
            content.page_count = page_count

            if not full_text.strip():
                content.parse_errors.append("PDF 中未提取到文本")
                return content

            # 2. 清洗
            cleaned = clean_text(full_text, pages=pages)
            content.full_text = cleaned

            # 3. 章节识别
            content.sections = sectionize(cleaned)
            content.char_count = len(cleaned)
            content.parse_success = True

            logger.info(
                f"Parsed PDF: {page_count} pages, {len(cleaned)} chars, "
                f"sections: {[k for k in ['abstract','introduction','methods','results','discussion','conclusion'] if getattr(content.sections, k)]}"
            )

        except ImportError as e:
            content.parse_errors.append(str(e))
        except Exception as e:
            content.parse_errors.append(f"解析错误: {e}")
            logger.warning(f"PDF parse error for {pdf_path}: {e}")

        return content

    def parse_batch(self, pdf_paths: list) -> list:
        """
        批量解析多篇 PDF

        Parameters
        ----------
        pdf_paths : list of str, PDF 文件路径列表

        Returns
        -------
        list of PDFContent
        """
        results = []
        for path in pdf_paths:
            result = self.parse(path)
            results.append(result)
        return results


# ── CLI 入口 ──────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("用法: python enhanced_pdf_parser.py <pdf_path> [section]")
        print("  python enhanced_pdf_parser.py paper.pdf          # 解析并显示摘要")
        print("  python enhanced_pdf_parser.py paper.pdf abstract # 显示指定章节")
        sys.exit(1)

    pdf_path = sys.argv[1]
    section = sys.argv[2] if len(sys.argv) > 2 else None

    parser = EnhancedPDFParser()
    content = parser.parse(pdf_path)

    if not content.parse_success:
        print(f"解析失败: {content.parse_errors}")
        sys.exit(1)

    print(f"解析成功: {content.page_count}页, {content.char_count}字")
    print(f"检测到章节: {[k for k in ['abstract','introduction','methods','results','discussion','conclusion','references'] if getattr(content.sections, k)]}")

    if section:
        text = getattr(content.sections, section, "")
        if text:
            print(f"\n--- {section} ---")
            print(text[:2000])
        else:
            print(f"未找到章节: {section}")
    else:
        # 默认显示摘要
        if content.sections.abstract:
            print(f"\n--- abstract ---")
            print(content.sections.abstract[:1000])
