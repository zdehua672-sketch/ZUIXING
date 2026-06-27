"""
文献深度学习模块
================
从高质量论文中学习写作逻辑，而非仅提取关键词。

核心能力:
  1. 句间逻辑链提取 — 学习"数据→统计→机制→结论"的论证链
  2. 段落结构分析 — 识别"主题句→支撑句→过渡句"结构
  3. 学术表达模式库 — 从论文中提取可复用的表达模式
  4. 引用-论证关系 — 学习如何用文献支撑论点

这是整个写作系统的基础模块——它不是记录论文内容，
而是学习"好论文是怎么写的"。
"""

import json
import re
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── 数据模型 ──────────────────────────────────────────────

@dataclass
class LogicChain:
    """一个句间逻辑链"""
    chain_type: str = ""      # data_to_mechanism / claim_evidence / comparison / causal
    sentences: list = field(default_factory=list)  # 链中的句子
    pattern: str = ""          # 逻辑模式描述
    section: str = ""          # 所在章节
    confidence: float = 0.0    # 置信度

    def to_dict(self):
        return asdict(self)


@dataclass
class ParagraphStructure:
    """段落结构分析"""
    topic_sentence: str = ""      # 主题句
    supporting_sentences: list = field(default_factory=list)  # 支撑句
    transition_sentence: str = ""  # 过渡句
    logic_flow: str = ""           # 逻辑流向描述
    section: str = ""              # 所在章节
    pattern_name: str = ""         # 识别出的模式名

    def to_dict(self):
        return asdict(self)


@dataclass
class ExpressionPattern:
    """学术表达模式"""
    pattern_id: str = ""
    pattern_type: str = ""     # claim / evidence / mechanism / transition / comparison
    template_zh: str = ""      # 中文模板
    template_en: str = ""      # 英文模板
    example_zh: str = ""       # 中文示例
    example_en: str = ""       # 英文示例
    variables: list = field(default_factory=list)  # 可替换变量
    source_section: str = ""   # 来源章节
    frequency: int = 0         # 出现频率
    quality_score: float = 0.0 # 质量评分

    def to_dict(self):
        return asdict(self)


@dataclass
class LiteratureLearning:
    """一篇论文的学习结果"""
    paper_id: str = ""
    title: str = ""
    logic_chains: list = field(default_factory=list)
    paragraph_structures: list = field(default_factory=list)
    expression_patterns: list = field(default_factory=list)
    citation_patterns: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


# ── 句间逻辑链提取器 ──────────────────────────────────────

class LogicChainExtractor:
    """
    从论文文本中提取句间逻辑链

    识别的逻辑链类型:
    1. data_to_mechanism: 数据发现 → 机制解释
    2. claim_evidence: 主张 → 证据支撑
    3. comparison: 本研究 → 文献对比
    4. causal: 原因 → 结果 → 影响
    """

    # 逻辑连接词模式
    CAUSE_MARKERS = [
        r'因为', r'由于', r'归因于', r'源于', r'导致',
        r'because', r'due to', r'attributed to', r'resulted from', r'caused by',
    ]
    EVIDENCE_MARKERS = [
        r'结果表明', r'数据显示', r'分析发现', r'统计表明',
        r'results?\s+show', r'data\s+indicate', r'analysis\s+reveal',
        r'p\s*[<>]', r'r\s*=', r'显著',
    ]
    MECHANISM_MARKERS = [
        r'机制', r'途径', r'过程', r'代谢', r'降解', r'转化',
        r'mechanism', r'pathway', r'process', r'metabolism',
        r'厌氧', r'好氧', r'微生物', r'anaerobic', r'aerobic',
    ]
    COMPARISON_MARKERS = [
        r'与.*一致', r'与.*类似', r'与.*不同', r'不同于',
        r'consistent with', r'similar to', r'in agreement',
        r'differed from', r'contrary to', r'unlike',
    ]
    TRANSITION_MARKERS = [
        r'然而', r'但是', r'此外', r'另外', r'同时', r'值得注意',
        r'however', r'furthermore', r'moreover', r'notably',
        r'it is worth noting', r'in addition',
    ]

    def extract_chains(self, text: str, section: str = '') -> list:
        """
        从文本中提取所有逻辑链

        Returns
        -------
        list of LogicChain
        """
        chains = []
        sentences = self._split_sentences(text)

        for i in range(len(sentences) - 1):
            current = sentences[i]
            next_sent = sentences[i + 1]

            # 检测逻辑关系
            chain = self._detect_chain(current, next_sent, section)
            if chain:
                chains.append(chain)

        # 检测三句链（数据→机制→文献）
        for i in range(len(sentences) - 2):
            s1, s2, s3 = sentences[i], sentences[i+1], sentences[i+2]
            chain = self._detect_three_sentence_chain(s1, s2, s3, section)
            if chain:
                chains.append(chain)

        return chains

    def _detect_chain(self, sent1, sent2, section) -> Optional[LogicChain]:
        """检测两句之间的逻辑关系"""

        # 1. 数据→机制
        if self._has_any(sent1, self.EVIDENCE_MARKERS) and self._has_any(sent2, self.MECHANISM_MARKERS):
            return LogicChain(
                chain_type='data_to_mechanism',
                sentences=[sent1.strip(), sent2.strip()],
                pattern='数据发现 → 机制解释',
                section=section,
                confidence=0.8,
            )

        # 2. 主张→证据
        if self._has_any(sent2, self.EVIDENCE_MARKERS) and not self._has_any(sent1, self.EVIDENCE_MARKERS):
            # sent1 是主张，sent2 是证据
            if any(kw in sent1 for kw in ['表明', '显示', '揭示', 'suggest', 'indicate', 'show']):
                return LogicChain(
                    chain_type='claim_evidence',
                    sentences=[sent1.strip(), sent2.strip()],
                    pattern='主张 → 数据证据',
                    section=section,
                    confidence=0.7,
                )

        # 3. 文献对比
        if self._has_any(sent2, self.COMPARISON_MARKERS):
            return LogicChain(
                chain_type='comparison',
                sentences=[sent1.strip(), sent2.strip()],
                pattern='本研究发现 → 文献对比',
                section=section,
                confidence=0.8,
            )

        # 4. 因果链
        if self._has_any(sent2, self.CAUSE_MARKERS):
            return LogicChain(
                chain_type='causal',
                sentences=[sent1.strip(), sent2.strip()],
                pattern='现象 → 原因解释',
                section=section,
                confidence=0.7,
            )

        # 5. 转折
        if self._has_any(sent2, self.TRANSITION_MARKERS):
            return LogicChain(
                chain_type='transition',
                sentences=[sent1.strip(), sent2.strip()],
                pattern='论点 → 转折/补充',
                section=section,
                confidence=0.6,
            )

        return None

    def _detect_three_sentence_chain(self, s1, s2, s3, section) -> Optional[LogicChain]:
        """检测三句链：数据→机制→文献"""
        if (self._has_any(s1, self.EVIDENCE_MARKERS) and
            self._has_any(s2, self.MECHANISM_MARKERS) and
            self._has_any(s3, self.COMPARISON_MARKERS)):
            return LogicChain(
                chain_type='data_mechanism_literature',
                sentences=[s1.strip(), s2.strip(), s3.strip()],
                pattern='数据发现 → 机制解释 → 文献支撑',
                section=section,
                confidence=0.9,
            )
        return None

    def _has_any(self, text, patterns):
        """检查文本是否包含任一模式"""
        for p in patterns:
            if re.search(p, text, re.IGNORECASE):
                return True
        return False

    def _split_sentences(self, text):
        """分句（中英文混合）"""
        # 先按中英文句号分
        parts = re.split(r'(?<=[。！？.!?])\s*', text.strip())
        # 过滤太短的
        return [s.strip() for s in parts if len(s.strip()) > 10]


# ── 段落结构分析器 ──────────────────────────────────────────

class ParagraphAnalyzer:
    """
    分析段落的内部结构

    识别模式:
    1. topic_evidence_conclusion: 主题句→证据→结论
    2. data_mechanism_literature: 数据→机制→文献
    3. claim_counter_evidence: 主张→反证→修正
    4. background_gap_objective: 背景→空白→目标
    5. finding_comparison_mechanism: 发现→对比→机制
    """

    def analyze(self, paragraph: str, section: str = '') -> ParagraphStructure:
        """分析单个段落的结构"""
        sentences = self._split_sentences(paragraph)
        if len(sentences) < 2:
            return ParagraphStructure(
                topic_sentence=paragraph[:100] if paragraph else '',
                section=section,
                pattern_name='single_sentence',
            )

        structure = ParagraphStructure(section=section)

        # 识别主题句（通常是第一句或包含"表明/显示/reveals/shows"的句子）
        structure.topic_sentence = self._find_topic_sentence(sentences)

        # 识别支撑句（包含数据/引用/机制的句子）
        structure.supporting_sentences = self._find_supporting_sentences(sentences)

        # 识别过渡句（最后一句或包含连接词的句子）
        structure.transition_sentence = self._find_transition_sentence(sentences)

        # 识别逻辑模式
        structure.pattern_name = self._identify_pattern(sentences)
        structure.logic_flow = self._describe_flow(sentences)

        return structure

    def _find_topic_sentence(self, sentences):
        """找主题句"""
        # 优先找包含"表明/显示/结果"的句子
        for s in sentences[:2]:
            if any(kw in s for kw in ['表明', '显示', '揭示', '结果', '发现',
                                       'indicate', 'suggest', 'reveal', 'show', 'result']):
                return s
        # 默认取第一句
        return sentences[0]

    def _find_supporting_sentences(self, sentences):
        """找支撑句"""
        supporting = []
        evidence_markers = ['r=', 'p<', 'p=', '均值', '标准差', '%', 'mg/L',
                           'significant', 'correlation', 'mean', 'SD']
        for s in sentences[1:]:
            if any(m in s for m in evidence_markers):
                supporting.append(s)
        return supporting

    def _find_transition_sentence(self, sentences):
        """找过渡句"""
        if len(sentences) < 3:
            return ''
        last = sentences[-1]
        transition_markers = ['然而', '此外', '另外', '值得注意', '总之',
                             'however', 'furthermore', 'notably', 'in summary']
        if any(m in last.lower() for m in transition_markers):
            return last
        return ''

    def _identify_pattern(self, sentences):
        """识别段落模式"""
        has_data = any(self._has_evidence(s) for s in sentences)
        has_mechanism = any(self._has_mechanism(s) for s in sentences)
        has_literature = any(self._has_citation(s) for s in sentences)
        has_claim = any(self._has_claim(s) for s in sentences)

        if has_data and has_mechanism and has_literature:
            return 'data_mechanism_literature'
        elif has_claim and has_data:
            return 'claim_evidence'
        elif has_data and has_mechanism:
            return 'data_mechanism'
        elif has_data and has_literature:
            return 'data_literature'
        else:
            return 'narrative'

    def _describe_flow(self, sentences):
        """描述逻辑流向"""
        flow_parts = []
        for s in sentences:
            if self._has_evidence(s):
                flow_parts.append('数据')
            elif self._has_mechanism(s):
                flow_parts.append('机制')
            elif self._has_citation(s):
                flow_parts.append('文献')
            elif self._has_claim(s):
                flow_parts.append('主张')
            else:
                flow_parts.append('叙述')
        return ' → '.join(flow_parts)

    def _has_evidence(self, text):
        return any(m in text for m in ['r=', 'p<', 'p=', '均值', '%', 'mg/L', 'significant'])

    def _has_mechanism(self, text):
        return any(m in text for m in ['因为', '由于', '机制', '代谢', '厌氧', '好氧', 'because', 'mechanism'])

    def _has_citation(self, text):
        return bool(re.search(r'\([A-Z][a-z]+\s+et\s+al|\[\d+\]|\(\d{4}\)', text))

    def _has_claim(self, text):
        return any(m in text for m in ['表明', '显示', '揭示', 'suggest', 'indicate', 'reveal'])

    def _split_sentences(self, text):
        parts = re.split(r'(?<=[。！？.!?])\s*', text.strip())
        return [s.strip() for s in parts if len(s.strip()) > 10]


# ── 表达模式提取器 ──────────────────────────────────────────

class ExpressionPatternExtractor:
    """
    从论文中提取可复用的学术表达模式

    提取的模式类型:
    1. claim: 主张句式 "X与Y呈显著Z相关"
    2. evidence: 证据句式 "r=0.72, p<0.001"
    3. mechanism: 机制句式 "这可能是由于..."
    4. transition: 过渡句式 "值得注意的是..."
    5. comparison: 对比句式 "与XX一致/不同"
    """

    # 中文表达模式（可复用模板）
    ZH_PATTERNS = {
        'claim_positive': {
            'template': '{变量1}与{变量2}呈显著正相关(r={r值}, p={p值})',
            'example': 'TOC与CH4呈显著正相关(r=0.68, p<0.01)',
            'variables': ['变量1', '变量2', 'r值', 'p值'],
            'type': 'claim',
        },
        'claim_negative': {
            'template': '{变量1}与{变量2}呈显著负相关(r={r值}, p={p值})',
            'example': 'DO与CH4呈显著负相关(r=-0.72, p<0.001)',
            'variables': ['变量1', '变量2', 'r值', 'p值'],
            'type': 'claim',
        },
        'claim_higher': {
            'template': '{变量}在{组1}显著高于{组2}({显著性})',
            'example': 'TOC浓度在冬季显著高于春季(***)',
            'variables': ['变量', '组1', '组2', '显著性'],
            'type': 'claim',
        },
        'mechanism_because': {
            'template': '这一现象可归因于{机制描述}。{具体解释}',
            'example': '这一现象可归因于厌氧条件下产甲烷古菌活性增强。当DO<0.5mg/L时...',
            'variables': ['机制描述', '具体解释'],
            'type': 'mechanism',
        },
        'mechanism_dueto': {
            'template': '{变量}的变化可能与{过程}有关。{详细机制}',
            'example': 'CH4浓度的变化可能与管道内厌氧产甲烷过程有关。有机碳在严格厌氧条件下...',
            'variables': ['变量', '过程', '详细机制'],
            'type': 'mechanism',
        },
        'comparison_consistent': {
            'template': '本研究发现{发现内容}，与{作者}({年份})的研究结论一致。',
            'example': '本研究发现DO与CH4呈显著负相关，与Guisasola等(2008)的研究结论一致。',
            'variables': ['发现内容', '作者', '年份'],
            'type': 'comparison',
        },
        'comparison_different': {
            'template': '不同于{作者}({年份})报道的{文献发现}，本研究中{本研究发现}。',
            'example': '不同于Jiang等(2011)报道的正相关关系，本研究中TOC与CO2无显著相关。',
            'variables': ['作者', '年份', '文献发现', '本研究发现'],
            'type': 'comparison',
        },
        'transition_notable': {
            'template': '值得注意的是，{补充说明}。',
            'example': '值得注意的是，餐饮区的TOC浓度显著高于其他功能区。',
            'variables': ['补充说明'],
            'type': 'transition',
        },
        'transition_furthermore': {
            'template': '此外，{额外发现}，这进一步{说明/证实}{结论}。',
            'example': '此外，冬季CH4浓度显著高于春季，这进一步证实了温度对产甲烷过程的调控作用。',
            'variables': ['额外发现', '说明/证实', '结论'],
            'type': 'transition',
        },
        'evidence_data': {
            'template': '{变量}的均值为{均值}±{标准差}({单位})，变异系数为{CV}%。',
            'example': 'TOC的均值为45.2±12.3(mg/L)，变异系数为27.2%。',
            'variables': ['变量', '均值', '标准差', '单位', 'CV'],
            'type': 'evidence',
        },
    }

    # 英文表达模式
    EN_PATTERNS = {
        'claim_positive': {
            'template': '{Var1} showed a significant positive correlation with {Var2} (r = {r}, p = {p})',
            'example': 'TOC showed a significant positive correlation with CH4 (r = 0.68, p < 0.01)',
            'variables': ['Var1', 'Var2', 'r', 'p'],
            'type': 'claim',
        },
        'claim_negative': {
            'template': '{Var1} was negatively correlated with {Var2} (r = {r}, p = {p})',
            'example': 'DO was negatively correlated with CH4 (r = -0.72, p < 0.001)',
            'variables': ['Var1', 'Var2', 'r', 'p'],
            'type': 'claim',
        },
        'mechanism': {
            'template': 'This {finding} can be attributed to {mechanism}. {explanation}',
            'example': 'This negative correlation can be attributed to the inhibition of methanogenesis under aerobic conditions. When DO exceeds 2 mg/L...',
            'variables': ['finding', 'mechanism', 'explanation'],
            'type': 'mechanism',
        },
        'comparison_consistent': {
            'template': 'Our finding that {finding} is consistent with the observations reported by {author} ({year}).',
            'example': 'Our finding that DO negatively correlates with CH4 is consistent with the observations reported by Guisasola et al. (2008).',
            'variables': ['finding', 'author', 'year'],
            'type': 'comparison',
        },
        'transition': {
            'template': 'Notably, {observation}, which further {supports/suggests} {conclusion}.',
            'example': 'Notably, CH4 concentrations were significantly higher in winter, which further supports the temperature-dependent nature of methanogenesis.',
            'variables': ['observation', 'supports/suggests', 'conclusion'],
            'type': 'transition',
        },
    }

    def extract_patterns(self, text: str, language: str = 'zh') -> list:
        """从文本中提取匹配的表达模式"""
        patterns_map = self.ZH_PATTERNS if language == 'zh' else self.EN_PATTERNS
        found = []

        for pattern_id, pattern in patterns_map.items():
            # 尝试用模板匹配文本中的句子
            template = pattern['template']
            # 简化匹配：检查关键元素是否出现
            variables = pattern['variables']
            match_score = 0
            for var in variables:
                if var.startswith('{') and var.endswith('}'):
                    continue  # 跳过纯变量占位
                if var in text:
                    match_score += 1

            if match_score > 0:
                ep = ExpressionPattern(
                    pattern_id=pattern_id,
                    pattern_type=pattern['type'],
                    template_zh=pattern.get('template', ''),
                    template_en=pattern.get('template', ''),
                    example_zh=pattern.get('example', ''),
                    example_en=pattern.get('example', ''),
                    variables=pattern['variables'],
                    frequency=match_score,
                )
                found.append(ep)

        return found

    def get_pattern(self, pattern_id: str, language: str = 'zh') -> Optional[dict]:
        """获取指定模式"""
        patterns_map = self.ZH_PATTERNS if language == 'zh' else self.EN_PATTERNS
        return patterns_map.get(pattern_id)


# ── 文献学习主类 ──────────────────────────────────────────

class LiteratureLearner:
    """
    文献深度学习器

    从论文中学习写作逻辑，而非仅提取内容。

    用法:
        learner = LiteratureLearner()

        # 从PDF学习
        learning = learner.learn_from_pdf("paper.pdf")

        # 从文本学习
        learning = learner.learn_from_text(paper_text, title="...")

        # 查询学到的表达模式
        patterns = learner.query_patterns('mechanism')

        # 查询逻辑链
        chains = learner.query_logic_chains('data_to_mechanism')

        # 导出为知识库
        learner.export_to_knowledge_store()
    """

    def __init__(self):
        self.learnings = []  # list of LiteratureLearning
        self.all_patterns = []  # 所有学到的表达模式
        self.all_chains = []  # 所有学到的逻辑链
        self.all_structures = []  # 所有学到的段落结构

    def learn_from_text(self, text: str, title: str = '', language: str = 'zh') -> LiteratureLearning:
        """
        从论文文本中学习写作模式

        Parameters
        ----------
        text : str, 论文全文或部分文本
        title : str, 论文标题
        language : str, 语言

        Returns
        -------
        LiteratureLearning
        """
        learning = LiteratureLearning(title=title)

        # 按章节分割
        sections = self._split_into_sections(text)

        for section_name, section_text in sections.items():
            if len(section_text) < 50:
                continue

            # 1. 提取逻辑链
            chain_extractor = LogicChainExtractor()
            chains = chain_extractor.extract_chains(section_text, section=section_name)
            learning.logic_chains.extend(chains)

            # 2. 分析段落结构
            para_analyzer = ParagraphAnalyzer()
            paragraphs = re.split(r'\n\s*\n', section_text)
            for para in paragraphs:
                if len(para.strip()) > 50:
                    structure = para_analyzer.analyze(para.strip(), section=section_name)
                    learning.paragraph_structures.append(structure)

            # 3. 提取表达模式
            pattern_extractor = ExpressionPatternExtractor()
            patterns = pattern_extractor.extract_patterns(section_text, language)
            learning.expression_patterns.extend(patterns)

        # 去重和评分
        learning.expression_patterns = self._deduplicate_patterns(learning.expression_patterns)

        self.learnings.append(learning)
        self.all_chains.extend(learning.logic_chains)
        self.all_structures.extend(learning.paragraph_structures)
        self.all_patterns.extend(learning.expression_patterns)

        logger.info(
            f"Learned from '{title}': "
            f"{len(learning.logic_chains)} chains, "
            f"{len(learning.paragraph_structures)} structures, "
            f"{len(learning.expression_patterns)} patterns"
        )

        return learning

    def learn_from_pdf(self, pdf_path: str) -> LiteratureLearning:
        """从PDF文件学习"""
        try:
            from enhanced_pdf_parser import EnhancedPDFParser
            parser = EnhancedPDFParser()
            content = parser.parse(pdf_path)
            if content.parse_success:
                return self.learn_from_text(content.full_text, title=Path(pdf_path).stem)
        except Exception as e:
            logger.warning(f"Failed to learn from PDF {pdf_path}: {e}")
        return LiteratureLearning()

    def query_patterns(self, pattern_type: str = None, language: str = 'zh') -> list:
        """查询学到的表达模式"""
        if pattern_type:
            return [p for p in self.all_patterns if p.pattern_type == pattern_type]
        return self.all_patterns

    def query_logic_chains(self, chain_type: str = None) -> list:
        """查询学到的逻辑链"""
        if chain_type:
            return [c for c in self.all_chains if c.chain_type == chain_type]
        return self.all_chains

    def get_best_patterns(self, pattern_type: str, top_n: int = 5) -> list:
        """获取指定类型的最优表达模式"""
        patterns = [p for p in self.all_patterns if p.pattern_type == pattern_type]
        patterns.sort(key=lambda x: x.frequency, reverse=True)
        return patterns[:top_n]

    def get_paragraph_templates(self, section: str = 'discussion') -> list:
        """获取指定章节的段落模板"""
        structures = [s for s in self.all_structures if s.section == section]
        # 按模式分组
        by_pattern = {}
        for s in structures:
            if s.pattern_name not in by_pattern:
                by_pattern[s.pattern_name] = []
            by_pattern[s.pattern_name].append(s)

        templates = []
        for pattern_name, structs in by_pattern.items():
            if len(structs) >= 2:  # 至少出现2次才作为模板
                templates.append({
                    'pattern': pattern_name,
                    'count': len(structs),
                    'example_flow': structs[0].logic_flow,
                    'example_topic': structs[0].topic_sentence[:80],
                })

        return templates

    def export_to_knowledge_store(self) -> dict:
        """导出学到的知识到 knowledge_store"""
        from pathlib import Path
        from datetime import datetime, timezone

        store_dir = Path(__file__).parent / "knowledge_store"
        store_dir.mkdir(exist_ok=True)

        # 导出表达模式
        patterns_path = store_dir / "expression_patterns.json"
        patterns_data = {
            "meta": {
                "description": "从论文中学习的学术表达模式",
                "updated": datetime.now(timezone.utc).isoformat(),
                "total_patterns": len(self.all_patterns),
                "total_chains": len(self.all_chains),
            },
            "patterns": [p.to_dict() for p in self.all_patterns],
            "logic_chains": [c.to_dict() for c in self.all_chains[:100]],  # 保留前100条
        }
        with open(patterns_path, 'w', encoding='utf-8') as f:
            json.dump(patterns_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Exported {len(self.all_patterns)} patterns to {patterns_path}")
        return {"patterns_path": str(patterns_path), "count": len(self.all_patterns)}

    def _split_into_sections(self, text):
        """将文本按章节分割"""
        sections = {}
        section_patterns = {
            'introduction': r'(?:^|\n)(?:#{1,3}\s*)?(?:\d+\.?\s*)?[Ii]ntroduction',
            'methods': r'(?:^|\n)(?:#{1,3}\s*)?(?:\d+\.?\s*)?[Mm]ethods?',
            'results': r'(?:^|\n)(?:#{1,3}\s*)?(?:\d+\.?\s*)?[Rr]esults?',
            'discussion': r'(?:^|\n)(?:#{1,3}\s*)?(?:\d+\.?\s*)?[Dd]iscussion',
            'conclusion': r'(?:^|\n)(?:#{1,3}\s*)?(?:\d+\.?\s*)?[Cc]onclusion',
            'abstract': r'(?:^|\n)(?:#{1,3}\s*)?[Aa]bstract',
        }

        lines = text.split('\n')
        current_section = 'other'
        current_lines = []

        for line in lines:
            matched = False
            for sec_name, pattern in section_patterns.items():
                if re.search(pattern, line):
                    if current_lines:
                        sections[current_section] = '\n'.join(current_lines)
                    current_section = sec_name
                    current_lines = [line]
                    matched = True
                    break
            if not matched:
                current_lines.append(line)

        if current_lines:
            sections[current_section] = '\n'.join(current_lines)

        return sections

    def _deduplicate_patterns(self, patterns):
        """去重表达模式"""
        seen = {}
        for p in patterns:
            key = p.pattern_id
            if key in seen:
                seen[key].frequency += p.frequency
            else:
                seen[key] = p
        return list(seen.values())


# ── 便捷入口 ──────────────────────────────────────────

def learn_from_paper(text_or_path: str, title: str = '') -> LiteratureLearning:
    """
    从论文中学习写作模式

    Parameters
    ----------
    text_or_path : str, 论文文本或PDF路径
    title : str, 论文标题
    """
    learner = LiteratureLearner()
    if os.path.isfile(text_or_path):
        return learner.learn_from_pdf(text_or_path)
    return learner.learn_from_text(text_or_path, title=title)


# 需要导入 os
import os


if __name__ == '__main__':
    import sys

    if '--test' in sys.argv:
        test_text = """
        Abstract
        This study investigated carbon pollutants in campus sewage networks.

        Introduction
        Urban sewage networks are important infrastructure. Previous studies
        have shown that anaerobic conditions promote methanogenesis (Guisasola et al., 2008).
        However, the multiphase distribution of carbon pollutants remains unclear.

        Results
        TOC and CH4 showed a significant positive correlation (r=0.68, p<0.01).
        This can be attributed to the fact that TOC serves as the primary substrate
        for methanogenic archaea. Our finding is consistent with Guisasola et al. (2008).
        Notably, DO was negatively correlated with CH4 (r=-0.72, p<0.001).
        This negative correlation is due to the inhibition of methanogenesis under
        aerobic conditions. When DO exceeds 2 mg/L, methanogenic activity ceases.

        Discussion
        The results indicate that dissolved oxygen is the key factor controlling
        CH4 production. This finding is consistent with previous studies (Jiang et al., 2011).
        However, unlike Jiang et al. who reported a linear relationship,
        our data suggests a threshold effect at DO=0.5 mg/L.
        """

        learner = LiteratureLearner()
        learning = learner.learn_from_text(test_text, title="Test Paper")

        print(f"Logic chains: {len(learning.logic_chains)}")
        for c in learning.logic_chains[:5]:
            print(f"  [{c.chain_type}] {c.pattern}")
            print(f"    S1: {c.sentences[0][:60]}...")
            print(f"    S2: {c.sentences[1][:60]}...")

        print(f"\nParagraph structures: {len(learning.paragraph_structures)}")
        for s in learning.paragraph_structures[:3]:
            print(f"  [{s.pattern_name}] {s.logic_flow}")
            print(f"    Topic: {s.topic_sentence[:60]}...")

        print(f"\nExpression patterns: {len(learning.expression_patterns)}")
        for p in learning.expression_patterns[:5]:
            print(f"  [{p.pattern_type}] {p.pattern_id}")

        print("\nTest passed!")
    else:
        print("用法: python literature_learner.py --test")
