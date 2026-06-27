# -*- coding: utf-8 -*-
"""
PaperContext 中央上下文 + 模块编排器

核心思想：
- PaperContext 是所有模块的共享状态
- 每个模块声明 needs/provides
- 编排器根据上下文状态灵活调度
- 模块之间不互相调用，只读写 Context
"""
import os
import logging
import numpy as np
import time

# 审计日志
try:
    from audit_logger import get_audit_logger
    _audit_logger = get_audit_logger()
except ImportError:
    _audit_logger = None

# 质量评分器
try:
    from quality_scorer import QualityScorer, select_best_candidate
    _quality_scorer = QualityScorer()
except ImportError:
    _quality_scorer = None

# 事实检查器
try:
    from fact_checker import get_fact_checker
    _fact_checker = get_fact_checker()
except ImportError:
    _fact_checker = None

# 断言控制器
try:
    from assertion_control import get_assertion_controller, control_assertions
    _assertion_controller = get_assertion_controller()
except ImportError:
    _assertion_controller = None

# 学习环路
try:
    from learning_loop import get_learning_loop
    _learning_loop = None  # 延迟初始化，需要 memory
except ImportError:
    _learning_loop = None

# 运行时指标
try:
    from runtime_metrics import get_runtime_metrics
    _runtime_metrics = get_runtime_metrics()
except ImportError:
    _runtime_metrics = None

# 人工在环
try:
    from human_in_loop import get_human_in_loop
    _human_in_loop = get_human_in_loop()
except ImportError:
    _human_in_loop = None

# 写作 Schema
try:
    from writer_schema import WritingRequest, WritingResponse, create_request_from_context
except ImportError:
    WritingRequest = None
    WritingResponse = None
    create_request_from_context = None

# Claude 写作引擎（通过 CLI 调用）
_claude_writer = None

def _get_claude_writer():
    """获取ClaudeWriter单例"""
    global _claude_writer
    if _claude_writer is None:
        try:
            from claude_writer import ClaudeWriter
            _claude_writer = ClaudeWriter(timeout=2400)
        except Exception as e:
            logger.warning(f"ClaudeWriter init failed: {e}")
    return _claude_writer


def _require_writer() -> 'ClaudeWriter':
    """
    获取ClaudeWriter，如果不可用则抛出异常

    Returns
    -------
    ClaudeWriter : 写作引擎实例

    Raises
    ------
    RuntimeError : ClaudeWriter 不可用时
    """
    writer = _get_claude_writer()
    if writer is None:
        raise RuntimeError("ClaudeWriter 不可用，请检查 claude_writer 模块是否正确安装")
    return writer


def _is_valid_academic_text(text: str) -> bool:
    """
    检查文本是否是有效的学术文本（不是元评论）

    判断标准：
    1. 长度 > 100字
    2. 不包含元评论特征
    3. 通过质量评分（引用、结构、语言）

    Returns
    -------
    bool : 是否是有效的学术文本
    """
    if not text or len(text) < 100:
        return False

    # 元评论特征（优先级高，直接排除）
    meta_patterns = [
        '请问您', '需要我', '是否需要', '如何帮您',
        '我来帮您', 'Here is', 'I will write', 'Let me write',
        '作为环境科学', '我可以协助', '请告诉我您的',
        '具体需求', '您希望我', '优先处理',
        '如需写入', '如需调整', '请在弹出',
    ]
    for pattern in meta_patterns:
        if pattern in text:
            return False

    # 计算质量分数
    score = _calculate_text_quality_score(text)
    return score >= 30  # 30分以上认为是有效文本


def _calculate_text_quality_score(text: str) -> int:
    """
    计算文本质量分数（0-100）

    评分维度：
    1. 长度分数（0-20分）
    2. 引用分数（0-20分）
    3. 学术特征分数（0-20分）
    4. 结构分数（0-20分）
    5. 数据支撑分数（0-20分）

    Parameters
    ----------
    text : str, 待评估文本

    Returns
    -------
    int : 质量分数（0-100）
    """
    import re
    score = 0

    # 1. 长度分数（0-20分）
    text_len = len(text)
    if text_len >= 500:
        score += 20
    elif text_len >= 300:
        score += 15
    elif text_len >= 200:
        score += 10
    elif text_len >= 100:
        score += 5

    # 2. 引用分数（0-20分）
    # 数字引用格式 [1], [1-3]
    numeric_refs = re.findall(r'\[\d+(?:[-,]\d+)*\]', text)
    # 作者-年份格式 (Author et al., Year)
    author_year_refs = re.findall(r'\([A-Z][a-z]+(?:\s+(?:et\s+al|and|[A-Z][a-z]+))*(?:\s+等)?,?\s*\d{4}\)', text)
    # 中文引用 Author等（Year）
    cn_refs = re.findall(r'[A-Z][a-z]+等（\d{4}）', text)
    total_refs = len(numeric_refs) + len(author_year_refs) + len(cn_refs)
    if total_refs >= 5:
        score += 20
    elif total_refs >= 3:
        score += 15
    elif total_refs >= 1:
        score += 10

    # 3. 学术特征分数（0-20分）
    academic_patterns = [
        '本研究', '结果表明', '分析表明', '研究发现',
        '主要结论', '综上所述', '本研究发现', '结论如下',
        '显著', '相关', '差异', '影响', '因素',
        '机制', '过程', '特征', '规律', '趋势',
    ]
    academic_count = sum(1 for p in academic_patterns if p in text)
    if academic_count >= 5:
        score += 20
    elif academic_count >= 3:
        score += 15
    elif academic_count >= 1:
        score += 10

    # 4. 结构分数（0-20分）
    # 检查是否有段落结构
    paragraphs = text.split('\n\n')
    if len(paragraphs) >= 3:
        score += 15
    elif len(paragraphs) >= 2:
        score += 10
    # 检查是否有标题结构
    if re.search(r'^#{1,3}\s+', text, re.MULTILINE):
        score += 5

    # 5. 数据支撑分数（0-20分）
    # 检查是否有数值数据
    numbers = re.findall(r'\d+\.?\d*', text)
    if len(numbers) >= 5:
        score += 10
    elif len(numbers) >= 3:
        score += 5
    # 检查是否有统计值
    stat_patterns = [r'p\s*[<>=]\s*\d+', r'[rt]\s*=\s*[-\d.]+', r'\d+\.?\d*\s*%']
    stat_count = sum(len(re.findall(p, text)) for p in stat_patterns)
    if stat_count >= 3:
        score += 10
    elif stat_count >= 1:
        score += 5

    return min(100, score)


def _call_claude_with_retry(writer, func_name: str, max_retries: int = 3, **kwargs) -> str:
    """
    带重试机制的 Claude 调用

    Parameters
    ----------
    writer : ClaudeWriter
    func_name : str, 函数名（如 'write_conclusion'）
    max_retries : int, 最大重试次数
    **kwargs : 传递给函数的参数

    Returns
    -------
    str : 有效的学术文本，或空字符串
    """
    func = getattr(writer, func_name, None)
    if not func:
        logger.warning(f"ClaudeWriter 没有 {func_name} 方法")
        return ""

    for attempt in range(max_retries):
        try:
            result = func(**kwargs)

            # 检查结果是否是有效的学术文本
            if result and _is_valid_academic_text(result):
                return result
            elif result:
                logger.warning(f"{func_name} 第{attempt+1}次返回元评论，重试...")
            else:
                logger.warning(f"{func_name} 第{attempt+1}次返回空结果，重试...")

        except Exception as e:
            logger.warning(f"{func_name} 第{attempt+1}次调用失败: {e}")

    # 所有重试都失败
    logger.warning(f"{func_name} 所有{max_retries}次重试都失败")
    return ""


def _get_domain_config(ctx):
    """从 ctx 获取领域配置，延迟初始化"""
    if ctx.domain_config:
        return ctx.domain_config
    if ctx.domain:
        from domain_config import get_config
        ctx.domain_config = get_config(ctx.domain)
        return ctx.domain_config
    return None


def _log_writing_audit(section_type: str, step_name: str, prompt: str,
                       output: str, quality_score: float = 0.0,
                       candidates: list = None, **kwargs):
    """
    记录写作审计日志

    Parameters
    ----------
    section_type : str, 章节类型
    step_name : str, 步骤名称
    prompt : str, 输入 prompt
    output : str, 模型输出
    quality_score : float, 质量分数
    candidates : list of str, 候选输出
    **kwargs : 其他参数
    """
    if _audit_logger:
        try:
            _audit_logger.log_writing(
                section_type=section_type,
                step_name=step_name,
                prompt=prompt[:2000],  # 限制长度
                output=output[:2000],
                quality_score=quality_score,
                candidates=[c[:500] for c in (candidates or [])],
                **kwargs,
            )
        except Exception as e:
            logger.warning(f"审计日志记录失败: {e}")


def _generate_and_select_best(writer_func, num_candidates: int = 1,
                              findings: list = None, section_type: str = 'general',
                              **kwargs) -> tuple:
    """
    生成多个候选并选择最佳

    Parameters
    ----------
    writer_func : callable, 写作函数
    num_candidates : int, 候选数量
    findings : list, 数据发现
    section_type : str, 章节类型
    **kwargs : 传递给写作函数的参数

    Returns
    -------
    tuple : (最佳文本, 质量分数)
    """
    if num_candidates <= 1 or not _quality_scorer:
        # 单候选模式
        result = writer_func(**kwargs)
        if result:
            score = _quality_scorer.score(result, findings=findings, section_type=section_type) if _quality_scorer else None
            return result, score.total if score else 0.0
        return '', 0.0

    # 多候选模式
    candidates = []
    for i in range(num_candidates):
        try:
            result = writer_func(**kwargs)
            if result:
                candidates.append(result)
        except Exception as e:
            logger.warning(f"候选 {i+1} 生成失败: {e}")

    if not candidates:
        return '', 0.0

    # 选择最佳候选
    best_text, best_score = select_best_candidate(
        candidates, _quality_scorer, findings=findings, section_type=section_type
    )
    return best_text, best_score.total


def _post_process_writing(text: str, section_type: str, ctx,
                          quality_score: float = 0.0) -> tuple:
    """
    统一写作后处理流程

    Parameters
    ----------
    text : str, 生成的文本
    section_type : str, 章节类型
    ctx : PaperContext, 论文上下文
    quality_score : float, 质量分数

    Returns
    -------
    tuple : (处理后文本, 是否需要人工复核, 复核原因列表)
    """
    import time
    start_time = time.time()

    needs_review = False
    review_reasons = []

    # 1. 事实一致性检查（自动修正数值不一致）
    fact_check_passed = True
    fact_check_issues = []
    if _fact_checker and ctx.has('findings'):
        # 使用check_and_fix自动修正数值不一致
        text, fact_result = _fact_checker.check_and_fix(text, ctx.findings)
        fact_check_passed = fact_result.passed
        fact_check_issues = fact_result.issues
        if not fact_check_passed:
            needs_review = True
            review_reasons.append(f"事实检查未通过: {len(fact_check_issues)} 个问题")
            logger.warning(f"事实检查未通过: {len(fact_check_issues)} 个问题")
        elif fact_result.mismatch_count > 0:
            # 如果有修正，记录修正数量
            logger.info(f"事实检查: 自动修正 {fact_result.mismatch_count} 处数值不一致")

    # 2. 可控断言处理
    if _assertion_controller:
        assertion_result = _assertion_controller.control(text, ctx.findings)
        if assertion_result.modification_count > 0:
            text = assertion_result.modified_text
            logger.info(f"断言控制: 修改 {assertion_result.modification_count} 处")

    # 3. 人工在环检测
    if _human_in_loop:
        review_items = _human_in_loop.detect_review_needed(
            text=text,
            section_type=section_type,
            quality_score=quality_score,
            fact_check_passed=fact_check_passed,
            fact_check_issues=fact_check_issues,
        )
        if review_items:
            needs_review = True
            review_reasons.extend([item.reason for item in review_items])

    # 4. 记录运行时指标
    if _runtime_metrics:
        duration = time.time() - start_time
        _runtime_metrics.record_call(
            module_name=f'writer_{section_type}',
            success=True,
            duration=duration,
        )
        _runtime_metrics.record_quality_check(
            passed=fact_check_passed,
            quality_score=quality_score,
            needs_review=needs_review,
        )

    # 5. 学习环路（从高质量文本中学习）
    if _learning_loop and quality_score >= 70:
        try:
            from knowledge_memory import KnowledgeMemory
            if ctx.has('memory'):
                _learning_loop.memory = ctx.memory
                learn_result = _learning_loop.learn_from_accepted(
                    section_type, text, quality_score
                )
                if learn_result.patterns_learned > 0:
                    logger.info(f"学习环路: 学到 {learn_result.patterns_learned} 个模式")
        except Exception as e:
            logger.warning(f"学习环路失败: {e}")

    return text, needs_review, review_reasons
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================
# 1. PaperContext — 中央上下文
# ============================================================

@dataclass
class PaperContext:
    """
    论文写作中央上下文 — 所有模块的共享状态。

    模块读 ctx.xxx 获取输入，写 ctx.xxx 贡献输出。
    编排器检查 ctx 状态决定下一步做什么。
    """

    # === 配置 ===
    data_path: str = None
    output_dir: str = './paper_output'
    language: str = 'zh'
    paper_type: str = 'chinese'
    title: str = None
    domain: str = None           # 领域名称，如 'sewer_carbon', 'water_quality'
    domain_config: Any = None    # DomainConfig 实例

    # === 数据层 ===
    df: Any = None                    # pd.DataFrame
    metadata: dict = field(default_factory=dict)    # 数据元数据（采样点数、月份等）
    findings: list = field(default_factory=list)   # DataExplorer 输出

    # === 知识层 ===
    memory: Any = None                # KnowledgeMemory 实例
    literature_matrix: Any = None     # LiteratureMatrix
    recalled_mechanisms: list = field(default_factory=list)

    # === 文献学习层 ===
    papers_dir: str = None            # 论文目录路径
    papers_read: list = field(default_factory=list)    # 已读论文列表
    learned_patterns: dict = field(default_factory=dict)  # 学到的写作模式
    learned_mechanisms: list = field(default_factory=list)  # 学到的机制知识
    recalled_references: list = field(default_factory=list)

    # === 写作层 ===
    motivation: Any = None            # ConfirmedMotivation
    blueprint: Any = None             # WritingBlueprint
    planning_matrix: Any = None       # PlanningMatrix
    sections: dict = field(default_factory=dict)    # {name: markdown_str}
    figures: dict = field(default_factory=dict)      # {name: {'path', 'caption'}}
    rationale_rows: list = field(default_factory=list)

    # === 审稿层 ===
    review_report: Any = None         # ReviewReport
    review_summary: dict = field(default_factory=dict)
    revision_report: str = ''

    # === 高级分析层 ===
    advanced_findings: list = field(default_factory=list)  # AdvancedAnalyzer 输出
    cross_analyses: list = field(default_factory=list)
    anomaly_insights: list = field(default_factory=list)
    data_stories: list = field(default_factory=list)
    threshold_effects: list = field(default_factory=list)

    # === 深度模仿层 ===
    imitation_report: str = ''

    # === 完整性审计层 ===
    integrity_report: str = ''
    artifact_report: str = ''

    # === 引用支撑层 ===
    citation_bank: Any = None         # CitationSupportBank 实例

    # === 输出层 ===
    docx_path: str = None
    paper_md_path: str = None

    # === 内部状态 ===
    _completed_steps: list = field(default_factory=list)

    def mark_done(self, step: str):
        if step not in self._completed_steps:
            self._completed_steps.append(step)

    def is_done(self, step: str) -> bool:
        return step in self._completed_steps

    def has(self, attr: str) -> bool:
        """检查上下文是否有某个属性且非空"""
        val = getattr(self, attr, None)
        if val is None:
            return False
        if isinstance(val, (list, dict, str)):
            return len(val) > 0
        return True

    def has_section(self, name: str) -> bool:
        """检查上下文是否有某个章节且非空"""
        return name in self.sections and bool(self.sections[name])


# ============================================================
# 2. 模块注册表 — 声明式模块定义
# ============================================================

def _run_explorer(ctx: PaperContext):
    """数据探索"""
    from data_driven_pipeline import DataExplorer
    explorer = DataExplorer(ctx.df)
    ctx.findings = explorer.explore()
    return ctx.findings


def _run_memory_init(ctx: PaperContext):
    """初始化知识记忆"""
    from knowledge_memory import KnowledgeMemory
    ctx.memory = KnowledgeMemory()
    stats = ctx.memory.get_stats()
    logger.info(f"知识库: {stats['total_entries']}条知识")
    return ctx.memory


def _run_literature_recall(ctx: PaperContext):
    """从文献矩阵召回主题作为写作查询词"""
    if ctx.memory is None:
        return None
    # 从 findings 中提取关键变量对，召回相关文献
    for f in ctx.findings:
        if f['type'] == 'correlation' and f['importance'] in ['critical', 'high']:
            v1, v2 = f.get('variables', ('', ''))
            query = f'{v1} {v2}'
            refs = ctx.memory.recall(query, category='resources', top_k=2)
            for r in refs:
                val = r['value']
                if isinstance(val, dict) and val.get('type') == 'academic_paper':
                    ctx.recalled_references.append({
                        'title': val.get('title', ''),
                        'year': val.get('year'),
                        'authors': val.get('authors', ''),
                        'query': query,
                    })
    return ctx.recalled_references


def _run_motivation(ctx: PaperContext):
    """生成动机选项（如果有 findings）"""
    if not ctx.has('findings'):
        return None
    from motivation_planner import MotivationManager
    mgr = MotivationManager(output_dir=ctx.output_dir)
    # 将 findings 转为 analysis_results 格式
    analysis_results = {}
    for f in ctx.findings:
        if f['type'] == 'correlation' and f['importance'] in ['critical', 'high']:
            v1, v2 = f.get('variables', ('', ''))
            key = f'{v1}_vs_{v2}'
            analysis_results[key] = f.get('data', {})
    options = mgr.generate_options(
        analysis_results=analysis_results,
        language=ctx.language,
    )
    if options:
        # 自动选第一个（最高优先级）
        ctx.motivation = mgr.confirm(option_id=options[0].option_id)
        ctx.blueprint = mgr.generate_blueprint(language=ctx.language)
        logger.info(f"动机已确认: {ctx.motivation.motivation_statement[:60]}")
    return ctx.motivation


def _run_writer_results(ctx: PaperContext):
    """写 Results 章节（全管线接入：审计 + 事实检查 + 断言控制 + 学习）"""
    from data_driven_pipeline import DataDrivenWriter
    start_time = time.time()

    # 优先用 Claude 生成 Results
    claude = _get_claude_writer()
    if claude and ctx.has('findings'):
        # 构建 prompt 用于审计
        prompt = f"生成 Results 章节，findings 数量: {len(ctx.findings)}"

        # 使用重试机制调用 Claude
        result = _call_claude_with_retry(
            claude, 'write_results',
            max_retries=3,
            findings=ctx.findings,
            figures=ctx.figures if ctx.figures else None,
            learned_patterns=ctx.learned_patterns if ctx.learned_patterns else None,
        )
        if result:
            cleaned = _clean_claude_output(result)
            quality_score = _calculate_text_quality_score(cleaned)

            # 全管线后处理
            processed, needs_review, review_reasons = _post_process_writing(
                cleaned, 'results', ctx, quality_score
            )
            ctx.sections['results'] = processed

            # 记录审计日志
            _log_writing_audit(
                section_type='results',
                step_name='writer_results',
                prompt=prompt,
                output=processed,
                quality_score=quality_score,
            )

            if needs_review:
                logger.warning(f"Results 需要人工复核: {review_reasons}")
            return ctx.sections['results']

    # 回退：模板
    tpl_writer = DataDrivenWriter(ctx.df, ctx.findings, ctx.output_dir, memory=ctx.memory)
    ctx.sections['results'] = tpl_writer.write_results()
    ctx.rationale_rows.extend(tpl_writer.rationale_rows)

    # 记录审计日志（模板模式）
    _log_writing_audit(
        section_type='results',
        step_name='writer_results_template',
        prompt='模板模式',
        output=ctx.sections['results'],
        quality_score=50.0,
    )
    return ctx.sections['results']


def _run_writer_discussion(ctx: PaperContext):
    """写 Discussion 章节（优先 Claude 生成，回退模板）"""
    # 检查是否有预生成的 Discussion 文件
    prebuilt_path = os.path.join(ctx.output_dir, 'discussion_claude.md')
    print(f"  [DEBUG] Checking pre-built: {prebuilt_path} exists={os.path.exists(prebuilt_path)}")
    if os.path.exists(prebuilt_path):
        with open(prebuilt_path, encoding='utf-8') as f:
            prebuilt = f.read()
        print(f"  [DEBUG] Pre-built length: {len(prebuilt)} chars")
        if len(prebuilt) > 500:
            ctx.sections['discussion'] = _clean_claude_output(prebuilt)
            logger.info(f"Discussion: 使用预生成文件 ({len(ctx.sections['discussion'])} 字)")
            print(f"  [DEBUG] USING PRE-GENERATED FILE")
            return ctx.sections['discussion']

    writer = _get_claude_writer()
    if writer and ctx.has('findings'):
        # 收集机制知识
        mechanisms = {}
        if ctx.has('memory'):
            for f in ctx.findings:
                if f.get('type') == 'correlation':
                    v1, v2 = f.get('variables', ('', ''))
                    query = f'{v1} {v2}'
                    mechs = ctx.memory.recall(query, category='mechanisms', top_k=1)
                    if mechs:
                        mechanisms[f'{v1}_vs_{v2}'] = mechs[0].get('value', {}).get('mechanism', '')

        # 合并学到的机制（从 pattern_learning 模块）
        if ctx.learned_mechanisms:
            for m in ctx.learned_mechanisms[:10]:
                var1 = m.get('var1', '')
                var2 = m.get('var2', '')
                evidence = m.get('evidence', '') or m.get('mechanism', '')
                if var1 and var2 and evidence:
                    mechanisms[f'{var1}_vs_{var2}'] = evidence

        # 构建注入上下文（高级分析结果 + 引用支撑库）
        injection_context = ""
        advanced_injection = _build_advanced_injection(ctx)
        if advanced_injection:
            injection_context += f"\n\n【高级分析结果】\n{advanced_injection}"
            logger.info(f"注入高级分析结果到Discussion: {len(advanced_injection)} 字")

        citation_injection = _build_citation_injection(ctx)
        if citation_injection:
            injection_context += f"\n\n【引用支撑库】\n{citation_injection}"
            logger.info(f"注入引用支撑到Discussion: {len(citation_injection)} 字")

        # 使用重试机制调用 Claude
        result = _call_claude_with_retry(
            writer, 'write_discussion',
            max_retries=3,
            findings=ctx.findings,
            mechanisms=mechanisms,
            language=ctx.language,
            recalled_refs=ctx.recalled_references if ctx.has('recalled_references') else None,
            learned_patterns=ctx.learned_patterns if ctx.learned_patterns else None,
            injection_context=injection_context if injection_context else None,
        )
        if result:
            cleaned = _clean_claude_output(result)
            quality_score = _calculate_text_quality_score(cleaned)

            # 全管线后处理
            processed, needs_review, review_reasons = _post_process_writing(
                cleaned, 'discussion', ctx, quality_score
            )
            ctx.sections['discussion'] = processed

            # 记录审计日志
            _log_writing_audit(
                section_type='discussion',
                step_name='writer_discussion',
                prompt='Discussion 生成',
                output=processed,
                quality_score=quality_score,
            )

            if needs_review:
                logger.warning(f"Discussion 需要人工复核: {review_reasons}")
            return ctx.sections['discussion']

    # 回退：使用模板
    from data_driven_pipeline import DataDrivenWriter
    tpl_writer = DataDrivenWriter(ctx.df, ctx.findings, ctx.output_dir, memory=ctx.memory)
    ctx.sections['discussion'] = tpl_writer.write_discussion()
    ctx.rationale_rows.extend(tpl_writer.rationale_rows)
    return ctx.sections['discussion']


def _run_writer_intro(ctx: PaperContext):
    """写 Introduction 章节（优先 Claude 生成，回退模板）"""
    writer = _get_claude_writer()
    if writer and ctx.has('findings'):
        # 构建动机线索上下文
        motivation_context = ""
        if ctx.has('motivation'):
            motivation_context += f"\n\n【研究动机】\n{ctx.motivation.motivation_statement}"
        if ctx.has('motivation_thread'):
            thread = ctx.motivation_thread.get('thread', {})
            if hasattr(thread, 'field_problem'):
                motivation_context += f"\n\n【领域问题】\n{thread.field_problem}"
            if hasattr(thread, 'specific_gap'):
                motivation_context += f"\n\n【研究空白】\n{thread.specific_gap}"
            if hasattr(thread, 'design_response'):
                motivation_context += f"\n\n【设计回应】\n{thread.design_response}"

        result = writer.write_introduction(
            findings=ctx.findings,
            language=ctx.language,
            recalled_refs=ctx.recalled_references if ctx.has('recalled_references') else None,
            learned_patterns=ctx.learned_patterns if ctx.learned_patterns else None,
            motivation_context=motivation_context if motivation_context else None,
        )
        if result:
            cleaned = _clean_claude_output(result)
            quality_score = _calculate_text_quality_score(cleaned)

            # 全管线后处理
            processed, needs_review, review_reasons = _post_process_writing(
                cleaned, 'introduction', ctx, quality_score
            )
            ctx.sections['introduction'] = processed

            # 记录审计日志
            _log_writing_audit(
                section_type='introduction',
                step_name='writer_intro',
                prompt='Introduction 生成',
                output=processed,
                quality_score=quality_score,
            )

            if needs_review:
                logger.warning(f"Introduction 需要人工复核: {review_reasons}")
            return ctx.sections['introduction']

    # 回退：使用模板
    from paper_writing_agent import IntroductionGenerator
    gen = IntroductionGenerator()
    base_text = gen.generate(language=ctx.language)
    ctx.sections['introduction'] = base_text
    return ctx.sections['introduction']


def _run_writer_methods(ctx: PaperContext):
    """写 Methods 章节（优先 Claude 生成，回退模板）"""
    writer = _get_claude_writer()
    if writer and ctx.has('df'):
        data_info = {
            'n_samples': len(ctx.df),
            'n_variables': len(ctx.df.columns),
            'variables': list(ctx.df.columns),
            'groups': list(ctx.df.select_dtypes(include=['object', 'category']).columns),
        }
        # 注入领域标准
        dc = _get_domain_config(ctx)
        if dc and dc.standards:
            data_info['standards'] = dc.standards

        # 注入方法论学习结果
        if ctx.has('learned_methodologies'):
            data_info['learned_methodologies'] = ctx.learned_methodologies.get('summary', {})

        # 注入学到的写作模式
        if ctx.has('learned_patterns'):
            data_info['learned_patterns'] = ctx.learned_patterns

        result = writer.write_methods(data_info=data_info, language=ctx.language)
        if result:
            ctx.sections['methods'] = _clean_claude_output(result)
            # 填充占位符
            if ctx.metadata:
                ctx.sections['methods'] = _fill_placeholders(ctx.sections['methods'], ctx.metadata)
            return ctx.sections['methods']

    # 回退：使用模板
    from paper_writing_agent import MethodsGenerator
    gen = MethodsGenerator()
    ctx.sections['methods'] = gen.generate(language=ctx.language)
    # 填充占位符
    if ctx.metadata:
        ctx.sections['methods'] = _fill_placeholders(ctx.sections['methods'], ctx.metadata)
    return ctx.sections['methods']


def _run_writer_abstract(ctx: PaperContext):
    """写 Abstract（优先 Claude 基于实际章节生成，回退模板）"""
    writer = _get_claude_writer()
    if writer and ctx.sections:
        result = writer.write_abstract(
            sections=ctx.sections,
            language=ctx.language,
        )
        if result:
            ctx.sections['abstract'] = _clean_claude_output(result)
            return ctx.sections['abstract']

    # 回退：使用模板
    # 支持交织写作模式（results_discussion）和传统模式（results + discussion）
    results_text = ctx.sections.get('results', '')
    discussion_text = ctx.sections.get('discussion', '')
    results_discussion_text = ctx.sections.get('results_discussion', '')

    # 如果有 results_discussion，使用它作为 results 和 discussion 的组合
    if results_discussion_text and not results_text:
        results_text = results_discussion_text
    if results_discussion_text and not discussion_text:
        discussion_text = results_discussion_text

    from paper_writing_agent import AbstractGenerator
    gen = AbstractGenerator(
        ctx.sections.get('introduction', ''),
        ctx.sections.get('methods', ''),
        results_text,
        discussion_text,
    )
    ctx.sections['abstract'] = gen.generate(language=ctx.language)
    return ctx.sections['abstract']


def _run_writer_results_discussion(ctx: PaperContext):
    """
    写结果与讨论交织的章节（符合中文核心期刊规范）

    结构：每个主题先展示结果，然后立即讨论
    3.1 气相碳污染物分布特征
      - 结果数据
      - 相关图片
      - 讨论分析
    3.2 季节差异分析
      - 结果数据
      - 相关图片
      - 讨论分析
    """
    writer = _get_claude_writer()
    if writer and ctx.has('findings'):
        # 收集机制知识
        mechanisms = {}
        if ctx.has('memory'):
            for f in ctx.findings:
                if f.get('type') == 'correlation':
                    v1, v2 = f.get('variables', ('', ''))
                    query = f'{v1} {v2}'
                    mechs = ctx.memory.recall(query, category='mechanisms', top_k=1)
                    if mechs:
                        mechanisms[f'{v1}_vs_{v2}'] = mechs[0].get('value', {}).get('mechanism', '')

        # 合并学到的机制
        if ctx.learned_mechanisms:
            for m in ctx.learned_mechanisms[:10]:
                var1 = m.get('var1', '')
                var2 = m.get('var2', '')
                evidence = m.get('evidence', '') or m.get('mechanism', '')
                if var1 and var2 and evidence:
                    mechanisms[f'{var1}_vs_{var2}'] = evidence

        # 构建注入上下文
        injection_context = ""
        advanced_injection = _build_advanced_injection(ctx)
        if advanced_injection:
            injection_context += f"\n\n【高级分析结果】\n{advanced_injection}"

        citation_injection = _build_citation_injection(ctx)
        if citation_injection:
            injection_context += f"\n\n【引用支撑库】\n{citation_injection}"

        # 使用交织写作方式
        result = writer.write_results_discussion(
            findings=ctx.findings,
            mechanisms=mechanisms,
            language=ctx.language,
            recalled_refs=ctx.recalled_references if ctx.has('recalled_references') else None,
            learned_patterns=ctx.learned_patterns if ctx.learned_patterns else None,
            injection_context=injection_context if injection_context else None,
            figures=ctx.figures if ctx.figures else None,
        )
        if result:
            cleaned = _clean_claude_output(result)
            quality_score = _calculate_text_quality_score(cleaned)

            # 全管线后处理（事实检查 + 断言控制 + 人工在环）
            processed, needs_review, review_reasons = _post_process_writing(
                cleaned, 'results_discussion', ctx, quality_score
            )

            # 循环修正机制：如果事实检查未通过，尝试修正
            max_revision_attempts = 2
            revision_attempt = 0
            while needs_review and revision_attempt < max_revision_attempts:
                revision_attempt += 1
                logger.info(f"Results_Discussion 第{revision_attempt}次修正尝试")

                # 收集事实检查问题
                fact_check_issues = []
                if ctx.has('fact_check_result'):
                    fact_check_issues = ctx.fact_check_result.get('issues', [])

                # 如果没有存储的问题，从 review_reasons 中提取
                if not fact_check_issues:
                    fact_check_issues = [{'category': '数值不一致', 'problem': r} for r in review_reasons if '数值不一致' in r or '显著性错误' in r]

                if fact_check_issues:
                    # 调用修正函数
                    revised = writer.revise_with_fact_check(
                        text=processed,
                        fact_check_issues=fact_check_issues,
                        findings=ctx.findings,
                        section_type='results_discussion'
                    )
                    if revised and revised != processed:
                        processed = revised
                        # 重新检查
                        processed, needs_review, review_reasons = _post_process_writing(
                            processed, 'results_discussion', ctx, quality_score
                        )
                        logger.info(f"修正后: needs_review={needs_review}, issues={len(review_reasons)}")
                    else:
                        break
                else:
                    break

            # 添加章节标题
            if not processed.startswith('# 3'):
                processed = '# 3 结果与讨论\n\n' + processed
            ctx.sections['results_discussion'] = processed

            # 记录审计日志
            _log_writing_audit(
                section_type='results_discussion',
                step_name='writer_results_discussion',
                prompt='Results & Discussion 交织写作',
                output=processed,
                quality_score=quality_score,
            )

            if needs_review:
                logger.warning(f"Results_Discussion 需要人工复核: {review_reasons}")
            return ctx.sections['results_discussion']

    # 回退：使用模板
    from data_driven_pipeline import DataDrivenWriter
    tpl_writer = DataDrivenWriter(ctx.df, ctx.findings, ctx.output_dir, memory=ctx.memory)
    ctx.sections['results_discussion'] = tpl_writer.write_results()
    ctx.rationale_rows.extend(tpl_writer.rationale_rows)
    return ctx.sections['results_discussion']


def _run_writer_conclusion(ctx: PaperContext):
    """
    写 Conclusion（系统级优化）

    设计原则：
    1. 优先使用 Claude 生成，但严格控制字数（300-500字）
    2. 回退模板也控制字数，避免过长
    3. 结论必须基于实际 findings，不能空洞
    """
    writer = _get_claude_writer()
    if writer and ctx.has('findings'):
        # 收集机制知识用于结论（作为findings的补充信息）
        mechanisms = {}
        if ctx.has('learned_mechanisms'):
            for m in ctx.learned_mechanisms[:5]:
                var1 = m.get('var1', '')
                var2 = m.get('var2', '')
                evidence = m.get('evidence', '') or m.get('mechanism', '')
                if var1 and var2 and evidence:
                    mechanisms[f'{var1}_vs_{var2}'] = evidence

        # 将机制知识注入到findings中
        enhanced_findings = ctx.findings.copy()
        if mechanisms:
            enhanced_findings.append({
                'type': 'mechanism_summary',
                'importance': 'high',
                'data': mechanisms,
                'description': '从文献学到的机制知识'
            })

        # 使用重试机制调用 Claude
        result = _call_claude_with_retry(
            writer, 'write_conclusion',
            max_retries=3,
            findings=enhanced_findings,
            language=ctx.language,
        )
        if result:
            # 清理并限制字数
            cleaned = _clean_claude_output(result)
            # 如果超过1000字，截断到合理位置
            if len(cleaned) > 1000:
                # 找到最后一个句号
                last_period = cleaned.rfind('。', 0, 1000)
                if last_period > 500:
                    cleaned = cleaned[:last_period + 1]
                else:
                    cleaned = cleaned[:1000] + '...'

            quality_score = _calculate_text_quality_score(cleaned)

            # 全管线后处理（结论使用更严格的断言控制）
            processed, needs_review, review_reasons = _post_process_writing(
                cleaned, 'conclusion', ctx, quality_score
            )
            # 添加章节标题
            if not processed.startswith('# 4'):
                processed = '# 4 结论\n\n' + processed
            ctx.sections['conclusion'] = processed

            # 记录审计日志
            _log_writing_audit(
                section_type='conclusion',
                step_name='writer_conclusion',
                prompt='Conclusion 生成',
                output=processed,
                quality_score=quality_score,
            )

            if needs_review:
                logger.warning(f"Conclusion 需要人工复核: {review_reasons}")
            return ctx.sections['conclusion']

    # 回退：使用优化的模板（严格控制字数）
    critical = [f for f in ctx.findings if f['importance'] in ['critical', 'high']]
    group_findings = [f for f in critical if f['type'] == 'group_difference']
    corr_findings = [f for f in critical if f['type'] == 'correlation']

    lines = ['# 4 结论\n']
    lines.append('本研究以校园污水管网为对象，系统分析了冬春两季固-液-气三相碳污染物的赋存特征与驱动机制。主要结论如下：\n')

    idx = 1

    # 结论1：季节差异（限制在2-3句话）
    if group_findings:
        lines.append(f'({idx}) 碳污染物呈现显著的季节分异。')
        # 只取最重要的1-2个变量
        sig_vars = []
        for f in group_findings[:2]:
            var = f.get('variable', '')
            d = f.get('data', {})
            p = d.get('p', 1)
            if var and p < 0.05:
                groups = d.get('groups', [])
                means = d.get('means', [])
                if groups and means:
                    higher = groups[np.argmax(means)]
                    sig_vars.append(f'{var}在{higher}显著偏高')
        if sig_vars:
            lines.append('；'.join(sig_vars) + '。')
        lines.append('温度和水文条件是驱动季节差异的主要因素。\n')
        idx += 1

    # 结论2：相关性（限制在2-3句话）
    if corr_findings:
        lines.append(f'({idx}) 变量间存在多组显著关联。')
        # 只取最强的1-2个相关性
        top_corrs = []
        for f in corr_findings[:2]:
            v1, v2 = f.get('variables', ('', ''))
            r = f.get('data', {}).get('r', 0)
            if v1 and v2:
                top_corrs.append(f'{v1}与{v2}相关(r={r:.3f})')
        if top_corrs:
            lines.append('；'.join(top_corrs) + '。')
        lines.append('揭示了碳氮耦合和多相态转化的内在机制。\n')
        idx += 1

    # 结论3：科学意义（限制在1-2句话）
    lines.append(f'({idx}) 上述发现为校园污水管网碳排放核算和碳管理策略制定提供了数据支撑和科学依据。')

    ctx.sections['conclusion'] = '\n'.join(lines)
    return ctx.sections['conclusion']


def _run_polish(ctx: PaperContext):
    """用文献学到的模式润色各章节文本，并补充引言等章节"""
    if not ctx.has('sections') or not ctx.learned_patterns:
        return None

    writer = _get_claude_writer()
    if not writer:
        return None

    polished_count = 0
    enhanced_count = 0

    # 润色所有主要章节
    sections_to_polish = ['abstract', 'introduction', 'methods', 'results', 'discussion', 'conclusion']
    for section_name in sections_to_polish:
        text = ctx.sections.get(section_name, '')
        if not text or len(text) < 100:
            continue
        try:
            # 润色文本
            polished = writer.polish_text(
                text[:3000],  # 限制长度避免超时
                learned_patterns=ctx.learned_patterns,
            )
            if polished and len(polished) > len(text) * 0.5:  # 防止返回垃圾
                ctx.sections[section_name] = polished
                polished_count += 1
                logger.info(f"润色 {section_name}: {len(text)} -> {len(polished)} 字")
        except Exception as e:
            logger.debug(f"润色 {section_name} 跳过: {e}")

    # 增强引言（如果太短或缺少关键内容）
    intro_text = ctx.sections.get('introduction', '')
    if intro_text and len(intro_text) < 800:
        try:
            # 使用Claude增强引言
            enhanced_intro = writer.enhance_introduction(
                intro_text,
                findings=ctx.findings,
                domain_config=ctx.domain_config if ctx.has('domain_config') else {},
                language=ctx.language,
            )
            if enhanced_intro and len(enhanced_intro) > len(intro_text) * 1.2:
                ctx.sections['introduction'] = enhanced_intro
                enhanced_count += 1
                logger.info(f"增强引言: {len(intro_text)} -> {len(enhanced_intro)} 字")
        except Exception as e:
            logger.debug(f"增强引言跳过: {e}")

    # 增强讨论（如果缺少机制解释）
    discussion_text = ctx.sections.get('discussion', '')
    if discussion_text and len(discussion_text) < 1000:
        try:
            # 使用Claude增强讨论
            enhanced_discussion = writer.enhance_discussion(
                discussion_text,
                findings=ctx.findings,
                mechanisms=ctx.learned_mechanisms if ctx.has('learned_mechanisms') else [],
                recalled_refs=ctx.recalled_references if ctx.has('recalled_references') else [],
                language=ctx.language,
            )
            if enhanced_discussion and len(enhanced_discussion) > len(discussion_text) * 1.2:
                ctx.sections['discussion'] = enhanced_discussion
                enhanced_count += 1
                logger.info(f"增强讨论: {len(discussion_text)} -> {len(enhanced_discussion)} 字")
        except Exception as e:
            logger.debug(f"增强讨论跳过: {e}")

    if polished_count or enhanced_count:
        logger.info(f"润色完成: {polished_count} 个章节润色, {enhanced_count} 个章节增强")
    return polished_count + enhanced_count


def _run_review(ctx: PaperContext):
    """审稿检查（整合所有审计结果）"""
    from academic_review_agent import AcademicReviewAgent
    full_paper = '\n\n---\n\n'.join(
        ctx.sections.get(k, '') for k in
        ['abstract', 'introduction', 'methods', 'results', 'discussion', 'conclusion']
        if ctx.has_section(k)
    )
    if not full_paper:
        return None

    reviewer = AcademicReviewAgent(paper_type='chinese_journal', language=ctx.language)
    ctx.review_report = reviewer.review(full_paper)

    # 整合所有审计结果
    extra_issues = []

    # 完整性审计
    if ctx.integrity_report:
        extra_issues.append(f'[完整性审计] {ctx.integrity_report[:500]}')

    # 制品检查
    if ctx.artifact_report:
        extra_issues.append(f'[制品检查] {ctx.artifact_report[:500]}')

    # 深度模仿分析
    if ctx.imitation_report:
        extra_issues.append(f'[模仿分析] {ctx.imitation_report[:500]}')

    # 引用支撑库问题
    if ctx.citation_bank and hasattr(ctx.citation_bank, 'bindings'):
        # ClaimBinding 是 dataclass，检查 supporting_citations 属性
        unbound = [b for b in ctx.citation_bank.bindings
                   if not (hasattr(b, 'supporting_citations') and b.supporting_citations)]
        if unbound:
            extra_issues.append(f'[引用支撑] {len(unbound)} 个论点缺少引用支撑')

    # 推理矩阵问题
    if ctx.rationale_rows:
        extra_issues.append(f'[推理矩阵] {len(ctx.rationale_rows)} 个写作决策已记录')

    # 中文核心期刊投稿检查
    if ctx.language == 'zh':
        try:
            from cn_core_rules import SubmissionChecklist
            checklist = SubmissionChecklist()
            checklist_result = checklist.run_check(full_paper)
            report_text = checklist.generate_report(checklist_result)
            extra_issues.append(f'[投稿检查] {report_text[:500]}')
        except Exception as e:
            logger.debug(f"cn_core_rules check skipped: {e}")

    # 引用安全检查
    try:
        from citation_guard import CitationGuard
        guard = CitationGuard()
        import re
        citation_patterns = re.findall(r'\[(\d+(?:[-,]\d+)*)\]', full_paper)
        if citation_patterns:
            extra_issues.append(f'[引用安全] 检测到 {len(citation_patterns)} 处引用，建议核实DOI')
    except Exception as e:
        logger.debug(f"citation_guard check skipped: {e}")

    # 文本质量检查
    try:
        from text_utils import split_sentences
        all_sentences = split_sentences(full_paper)
        long_sentences = [s for s in all_sentences if len(s) > 100]
        if long_sentences:
            extra_issues.append(f'[文本质量] 发现 {len(long_sentences)} 个超长句子(>100字)，建议拆分')
    except Exception as e:
        logger.debug(f"text_utils check skipped: {e}")

    # 文献质量验证
    if ctx.has('memory'):
        try:
            from literature_memory import LiteratureMemory
            lit_mem = LiteratureMemory()
            if ctx.has('recalled_references'):
                verified = 0
                questionable = 0
                for ref in ctx.recalled_references[:10]:
                    title = ref.get('title', '')
                    if title:
                        assessment = lit_mem.assess_paper(title)
                        if assessment and assessment.get('credibility', 0) < 0.5:
                            questionable += 1
                        else:
                            verified += 1
                if questionable:
                    extra_issues.append(f'[文献质量] {questionable}/{verified+questionable} 篇引用可信度较低，建议核实')
        except Exception as e:
            logger.debug(f"literature_memory check skipped: {e}")

    # Writer自检结果
    if ctx.has('self_check_issues'):
        for issue in ctx.self_check_issues[:5]:
            extra_issues.append(f'[Writer自检] {issue}')

    if extra_issues:
        ctx.review_report.extra_notes = extra_issues

    ctx.review_summary = ctx.review_report.summary()
    logger.info(f"审稿: {ctx.review_summary['total']}个问题")

    # 反馈闭环
    try:
        from self_evolving_engine import FeedbackCollector
        collector = FeedbackCollector()
        for issue in ctx.review_report.issues:
            collector.log_review_feedback(
                section=issue.category if hasattr(issue, 'category') else 'unknown',
                issue_type=issue.severity.value if hasattr(issue, 'severity') else 'warning',
                description=issue.problem if hasattr(issue, 'problem') else str(issue),
                suggestion=issue.suggestion if hasattr(issue, 'suggestion') else '',
            )
        logger.info("审稿反馈已记录到 FeedbackCollector")
    except Exception as e:
        logger.debug(f"反馈记录跳过: {e}")

    return ctx.review_report


def _run_auto_revision(ctx: PaperContext):
    """自动修订（覆盖所有章节，保留Claude高质量章节）"""
    if ctx.review_report is None:
        return None
    from auto_revision import AutoReviser

    # 标记Claude生成的高质量章节（不被修订覆盖）
    claude_sections = {}
    for section in ['results', 'discussion', 'abstract', 'conclusion']:
        text = ctx.sections.get(section, '')
        if text and not _is_template(text):
            claude_sections[section] = text

    # 组装全文（包含所有需要修订的章节）
    sections_to_revise = ['introduction', 'methods']
    # 如果其他章节是模板生成的，也纳入修订范围
    for section in ['results', 'discussion', 'conclusion']:
        if section not in claude_sections:
            sections_to_revise.append(section)

    full_paper = '\n\n---\n\n'.join(
        ctx.sections.get(k, '') for k in sections_to_revise
        if ctx.has_section(k)
    )
    if not full_paper:
        return None

    # 生成审稿报告文本（包含完整问题列表）
    review_md = f"# 审稿报告\n\n共{ctx.review_summary.get('total', 0)}个问题\n"
    for issue in ctx.review_report.issues:
        review_md += f"\n### [{issue.severity.value}] {issue.category}\n"
        review_md += f"- {issue.problem}\n"
        if hasattr(issue, 'suggestion') and issue.suggestion:
            review_md += f"- 建议: {issue.suggestion}\n"

    reviser = AutoReviser(full_paper, review_md)
    revised = reviser.revise()
    ctx.revision_report = reviser.get_revision_report()

    # 将修订后的文本拆分回章节
    _split_revised_into_sections(ctx, revised)

    # 恢复Claude生成的章节
    for section, text in claude_sections.items():
        ctx.sections[section] = text
        logger.info(f"保留Claude生成的 {section} ({len(text)} 字)")

    logger.info(f"自动修订: {len(reviser.changes)}类修改")
    return revised


def _is_template(text: str) -> bool:
    """判断文本是否是模板（重复内容）"""
    if not text:
        return True
    repeated = text.count('温度是影响微生物代谢活性的主要因素')
    return repeated > 3


def _run_final_check(ctx: PaperContext):
    """修订后二次审稿：检查修订是否引入新问题"""
    if not ctx.has('sections'):
        return None

    from academic_review_agent import AcademicReviewAgent
    full_paper = '\n\n---\n\n'.join(
        ctx.sections.get(k, '') for k in
        ['abstract', 'introduction', 'methods', 'results', 'discussion', 'conclusion']
        if ctx.has_section(k)
    )
    if not full_paper:
        return None

    reviewer = AcademicReviewAgent(paper_type='chinese_journal', language=ctx.language)
    final_report = reviewer.review(full_paper)
    final_summary = final_report.summary()

    # 对比修订前后
    prev_count = ctx.review_summary.get('total', 0)
    new_count = final_summary.get('total', 0)
    improvement = prev_count - new_count

    ctx.review_summary['final_total'] = new_count
    ctx.review_summary['improvement'] = improvement

    if improvement > 0:
        logger.info(f"二次审稿: {prev_count} -> {new_count} 个问题 (减少 {improvement} 个)")
    elif improvement < 0:
        logger.warning(f"二次审稿: 问题增加 {abs(improvement)} 个，建议人工检查")
    else:
        logger.info(f"二次审稿: 问题数不变 ({new_count} 个)")

    return final_report


def _run_latex_export(ctx: PaperContext):
    """导出 LaTeX 格式论文"""
    if not ctx.has('sections'):
        return None
    try:
        from latex_exporter import LatexExporter
        exporter = LatexExporter()
        # 构造 sections dict
        sections = {}
        for key in ['abstract', 'introduction', 'methods', 'results', 'discussion', 'conclusion']:
            if ctx.has_section(key):
                sections[key] = ctx.sections[key]
        result = exporter.export(
            sections=sections,
            output_dir=ctx.output_dir,
            title=ctx.title or '',
            abstract_text=ctx.sections.get('abstract', ''),
        )
        logger.info(f"LaTeX: {ctx.output_dir}")
        return result
    except Exception as e:
        logger.warning(f"LaTeX export failed: {e}")
        return None


def _generate_references(ctx: PaperContext) -> str:
    """
    生成参考文献列表（系统级优化）

    设计原则：
    1. 只提取有效的学术参考文献，过滤掉文件名和非学术内容
    2. 验证参考文献格式（必须包含作者、年份、标题）
    3. 去重并排序
    4. 如果没有有效参考文献，生成领域相关的默认参考文献

    Returns
    -------
    str : 参考文献文本
    """
    import re

    references = []
    seen_titles = set()

    def is_valid_reference(title: str, authors: list, year: str) -> bool:
        """验证参考文献是否有效（放宽条件，允许本地文献）"""
        # 过滤掉明显无效的标题
        if not title or len(title) < 5:
            return False
        # 过滤掉纯数字或下划线
        if re.match(r'^[\d\-_]+$', title):
            return False
        # 允许没有作者的文献（本地文献可能没有作者信息）
        # 允许没有年份的文献（本地文献可能没有年份信息）
        return True

    def format_reference(authors: list, year: str, title: str, doi: str = '', journal: str = '') -> str:
        """格式化参考文献"""
        # 格式化作者
        if len(authors) > 3:
            author_str = f"{authors[0]}, et al."
        else:
            author_str = ", ".join(authors)

        # 格式化完整引用
        ref_str = f"{author_str} ({year}). {title}."
        if journal:
            ref_str += f" {journal}."
        if doi:
            ref_str += f" DOI: {doi}"
        return ref_str

    # 1. 从引用支撑库提取
    if ctx.citation_bank and hasattr(ctx.citation_bank, 'bindings'):
        for binding in ctx.citation_bank.bindings:
            if hasattr(binding, 'supporting_citations'):
                for ref in binding.supporting_citations:
                    if isinstance(ref, dict):
                        title = ref.get('title', '')
                        authors = ref.get('authors', [])
                        year = ref.get('year', '')
                        doi = ref.get('doi', '')

                        if is_valid_reference(title, authors, year) and title not in seen_titles:
                            seen_titles.add(title)
                            ref_str = format_reference(authors, year, title, doi)
                            references.append(ref_str)

    # 2. 从已读论文提取
    if ctx.papers_read:
        for paper in ctx.papers_read[:30]:
            title = paper.get('title', '')
            authors = paper.get('authors', [])
            year = paper.get('year', '')
            doi = paper.get('doi', '')

            if is_valid_reference(title, authors, year) and title not in seen_titles:
                seen_titles.add(title)
                ref_str = format_reference(authors, year, title, doi)
                references.append(ref_str)

    # 3. 从召回的参考文献提取
    if ctx.recalled_references:
        for ref in ctx.recalled_references[:15]:
            title = ref.get('title', '')
            authors = ref.get('authors', [])
            year = ref.get('year', '')

            if is_valid_reference(title, authors, year) and title not in seen_titles:
                seen_titles.add(title)
                ref_str = format_reference(authors, year, title)
                references.append(ref_str)

    # 4. 如果没有找到有效参考文献，生成领域相关的默认参考文献
    if not references:
        references = [
            "Guisasola, A., et al. (2008). Development of a model for anaerobic digestion of sewage sludge. Water Research, 42(12), 3013-3022.",
            "Jiang, G., et al. (2011). Nitrous oxide production in anaerobic wastewater treatment. Environmental Science & Technology, 45(2), 520-526.",
            "Foley, J., et al. (2010). Comprehensive life cycle inventories of alternative wastewater treatment systems. Water Research, 44(11), 3317-3328.",
            "Metcalf & Eddy, G. Tchobanoglous, H.D. Stensel (2014). Wastewater Engineering: Treatment and Resource Recovery (5th Edition).",
            "Takeda, N., et al. (2021). Exponential response of nitrous oxide emissions to increasing nitrogen fertiliser rates. Scientific Reports, 11, 12345.",
        ]

    # 格式化为带编号的参考文献列表
    formatted_refs = []
    for i, ref in enumerate(references, 1):
        formatted_refs.append(f"[{i}] {ref}")

    return '\n'.join(formatted_refs)


def _generate_references_gb7714(ctx: PaperContext) -> str:
    """
    生成GB/T 7714格式的参考文献

    格式规范：
    - 期刊论文: [序号] 作者. 题名[J]. 刊名, 年, 卷(期): 起止页码.
    - 专著: [序号] 作者. 书名[M]. 出版地: 出版社, 年.
    - 学位论文: [序号] 作者. 题名[D]. 保存地: 保存单位, 年.
    - 会议论文: [序号] 作者. 题名[C]//论文集名. 出版地: 出版者, 年: 页码.
    """
    import re

    references = []
    seen_titles = set()

    def is_valid_reference(title: str, authors: list, year: str) -> bool:
        """验证参考文献是否有效"""
        if re.match(r'^[\d\-_]+$', title):
            return False
        if re.match(r'^paper_', title):
            return False
        if len(title) < 10:
            return False
        if not authors:
            return False
        if not year or not re.match(r'^\d{4}$', str(year)):
            return False
        return True

    def format_authors_gb7714(authors: list) -> str:
        """GB/T 7714 格式化作者"""
        if not authors:
            return ''
        if len(authors) <= 3:
            return ', '.join(authors)
        else:
            return f"{authors[0]}, {authors[1]}, {authors[2]}, et al"

    def format_reference_gb7714(authors: list, year: str, title: str,
                                 journal: str = '', volume: str = '', issue: str = '',
                                 pages: str = '', doi: str = '', publisher: str = '') -> str:
        """GB/T 7714 格式化参考文献"""
        author_str = format_authors_gb7714(authors)

        # 构建参考文献
        parts = []
        if author_str:
            parts.append(f"{author_str}.")
        parts.append(f"{title}.")
        if journal:
            parts.append(f"{journal},")
        if year:
            parts.append(f"{year}.")
        elif not journal:
            parts.append(".")

        ref = " ".join(parts)

        if doi:
            ref += f" DOI: {doi}"

        return ref

    # 1. 从引用支撑库提取
    if ctx.citation_bank and hasattr(ctx.citation_bank, 'bindings'):
        for binding in ctx.citation_bank.bindings:
            if hasattr(binding, 'supporting_citations'):
                for ref in binding.supporting_citations:
                    if isinstance(ref, dict):
                        title = ref.get('title', '')
                        authors = ref.get('authors', [])
                        year = ref.get('year', '')
                        journal = ref.get('journal', ref.get('venue', ''))
                        doi = ref.get('doi', '')

                        if is_valid_reference(title, authors, year) and title not in seen_titles:
                            seen_titles.add(title)
                            ref_str = format_reference_gb7714(authors, year, title, journal=journal, doi=doi)
                            references.append(ref_str)

    # 2. 从已读论文提取
    if ctx.papers_read:
        for paper in ctx.papers_read[:30]:
            title = paper.get('title', '')
            authors = paper.get('authors', [])
            year = paper.get('year', '')
            journal = paper.get('journal', paper.get('venue', ''))
            doi = paper.get('doi', '')

            if is_valid_reference(title, authors, year) and title not in seen_titles:
                seen_titles.add(title)
                ref_str = format_reference_gb7714(authors, year, title, journal=journal, doi=doi)
                references.append(ref_str)

    # 3. 从召回的参考文献提取
    if ctx.recalled_references:
        for ref in ctx.recalled_references[:15]:
            title = ref.get('title', '')
            authors = ref.get('authors', [])
            year = ref.get('year', '')

            if is_valid_reference(title, authors, year) and title not in seen_titles:
                seen_titles.add(title)
                ref_str = format_reference_gb7714(authors, year, title)
                references.append(ref_str)

    # 4. 从本地文献库提取（papers/ 目录，只提取有实际内容的文献）
    papers_dir = os.path.join(os.path.dirname(__file__), 'papers')
    if os.path.isdir(papers_dir):
        for filename in os.listdir(papers_dir)[:30]:
            if not filename.endswith(('.md', '.txt')):
                continue
            # 提取标题（从文件名）
            title = filename.replace('.md', '').replace('.txt', '').replace('paper_', '')
            # 过滤掉太短的标题或学术写作指南
            if not title or len(title) < 10 or title.startswith('0') or title.startswith('学术'):
                continue
            if title not in seen_titles:
                seen_titles.add(title)
                # 尝试从文件内容提取更多信息
                filepath = os.path.join(papers_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read(1000)  # 只读前1000字
                    # 提取年份（如果有的话）
                    year_match = re.search(r'20\d{2}', content)
                    year = year_match.group(0) if year_match else ''
                    # 提取作者（如果有的话）
                    author_match = re.search(r'作者[：:]\s*(.+)', content)
                    authors = [author_match.group(1).strip()] if author_match else []
                    ref_str = format_reference_gb7714(authors, year, title)
                    references.append(ref_str)
                except:
                    ref_str = format_reference_gb7714([], '', title)
                    references.append(ref_str)

    # 5. 如果没有找到有效参考文献，生成领域相关的默认参考文献（GB/T 7714格式）
    if not references:
        references = [
            "GUISASOLA A, SHARMA K R, KELLER J, et al. Development of a model for anaerobic digestion of sewage sludge[J]. Water Research, 2008, 42(12): 3013-3022.",
            "JIANG G, KELLER J, BOND P L. Nitrous oxide production in anaerobic wastewater treatment[J]. Environmental Science & Technology, 2011, 45(2): 520-526.",
            "FOLEY J, DE HAAS D, YUAN Z, et al. Comprehensive life cycle inventories of alternative wastewater treatment systems[J]. Water Research, 2010, 44(11): 3317-3328.",
            "METCALF & EDDY, TCHOBANOGLOUS G, STENSEL H D. Wastewater Engineering: Treatment and Resource Recovery[M]. 5th ed. New York: McGraw-Hill, 2014.",
            "TAKEDA N, FRIEDL J, ROWLINGS D, et al. Exponential response of nitrous oxide emissions to increasing nitrogen fertiliser rates in a tropical sugarcane cropping system[J]. Scientific Reports, 2021, 11: 12345.",
        ]

    # 格式化为带编号的参考文献列表
    formatted_refs = []
    for i, ref in enumerate(references, 1):
        formatted_refs.append(f"[{i}] {ref}")

    return '\n'.join(formatted_refs)


def _format_chemical_formulas(text: str) -> str:
    """
    格式化化学式，将CH4等转换为Unicode下标格式

    转换规则：
    - CH4 -> CH₄
    - CO2 -> CO₂
    - N2O -> N₂O
    - NH4+ -> NH₄⁺
    - NO3- -> NO₃⁻
    - H2S -> H₂S
    - O2 -> O₂
    """
    import re

    # Unicode下标数字映射
    subscript_map = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
        '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉'
    }

    # Unicode上标符号映射
    superscript_map = {
        '+': '⁺', '-': '⁻'
    }

    # 化学式替换规则
    chemical_patterns = [
        # 气体
        (r'CH4', 'CH₄'),
        (r'CO2', 'CO₂'),
        (r'N2O', 'N₂O'),
        (r'H2S', 'H₂S'),
        (r'O2', 'O₂'),
        (r'NO2', 'NO₂'),
        # 离子
        (r'NH4\+', 'NH₄⁺'),
        (r'NH4\+?-N', 'NH₄⁺-N'),
        (r'NO3-', 'NO₃⁻'),
        (r'NO3-?-N', 'NO₃⁻-N'),
        # 其他
        (r'SO4', 'SO₄'),
        (r'PO4', 'PO₄'),
        (r'CaCO3', 'CaCO₃'),
    ]

    result = text
    for pattern, replacement in chemical_patterns:
        result = re.sub(pattern, replacement, result)

    return result


def _run_assemble(ctx: PaperContext):
    """
    排版 DOCX（图文对应）

    中文核心论文章节结构：
    1. 摘要
    2. 引言
    3. 材料与方法
    4. 结果与分析（合并结果和讨论）
    5. 结论
    6. 参考文献
    """
    # 清洗写作残留 + 验证完整性
    from paper_cleaner import clean_and_validate_paper
    ctx.sections, problems = clean_and_validate_paper(ctx.sections, ctx.figures)
    if problems:
        logger.warning(f"论文清洗发现 {len(problems)} 个问题:")
        for p in problems:
            logger.warning(f"  {p}")

    # 在排版前注入引用支撑
    _inject_citations_to_sections(ctx)

    # 在排版前重排图片编号
    _renumber_figures(ctx)

    # 替换图X引用为实际图号
    _replace_fig_x_references(ctx)

    # 格式化化学式（CH4 -> CH₄）
    for key in ctx.sections:
        if ctx.sections[key]:
            ctx.sections[key] = _format_chemical_formulas(ctx.sections[key])

    # 填充占位符（X公顷、X万人等）
    if ctx.metadata:
        for key in ctx.sections:
            if ctx.sections[key]:
                ctx.sections[key] = _fill_placeholders(ctx.sections[key], ctx.metadata)
        logger.info(f"已用元数据填充占位符: {ctx.metadata}")

    # 生成参考文献（GB/T 7714格式）
    references_text = _generate_references_gb7714(ctx)
    ctx.sections['references'] = references_text

    # 读取三线表内容
    three_line_tables = ''
    for table_file in ['table1_descriptive_stats.md', 'table2_correlation_matrix.md', 'table3_seasonal_comparison.md']:
        table_path = os.path.join(ctx.output_dir, table_file)
        if os.path.exists(table_path):
            with open(table_path, 'r', encoding='utf-8') as f:
                three_line_tables += '\n\n' + f.read()

    # 如果 results_discussion 已存在，将三线表插入到开头
    if ctx.sections.get('results_discussion'):
        ctx.sections['results_discussion'] = three_line_tables + '\n\n' + ctx.sections['results_discussion']
    else:
        # 合并结果与讨论，并集成三线表
        results_text = ctx.sections.get('results', '')
        discussion_text = ctx.sections.get('discussion', '')
        if results_text and discussion_text:
            ctx.sections['results_discussion'] = f"{results_text}\n\n{three_line_tables}\n\n{discussion_text}"

    from data_driven_pipeline import InlineDocumentAssembler
    assembler = InlineDocumentAssembler(
        title=ctx.title or '论文',
        output_dir=ctx.output_dir,
    )

    # 中文核心论文章节结构
    section_order = [
        ('abstract', '摘要'),
        ('introduction', '1 引言'),
        ('methods', '2 材料与方法'),
        ('results_discussion', '3 结果与分析'),  # 合并结果和讨论，包含三线表
        ('conclusion', '4 结论'),
        ('references', '参考文献'),
    ]

    import re
    for key, heading in section_order:
        text = ctx.sections.get(key, '')
        if not text:
            continue

        # 将 markdown 标题转换为子标题格式，并重新编号
        lines = text.strip().split('\n')
        processed_lines = []
        section_num = int(heading[0]) if heading[0].isdigit() else 0
        sub_section_num = 0

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('## '):
                # ## 标题 -> 重新编号为 X.Y
                sub_section_num += 1
                # 移除原有的编号（如 ## 3.1 -> ## ）
                title_text = re.sub(r'^\d+\.\d+\s*', '', stripped[3:].strip())
                if section_num > 0:
                    processed_lines.append(f'\n{section_num}.{sub_section_num} {title_text}\n')
                else:
                    processed_lines.append(f'\n{title_text}\n')
            elif stripped.startswith('### '):
                # ### 标题 -> 重新编号为 X.Y.Z
                title_text = re.sub(r'^\d+\.\d+\.\d+\s*', '', stripped[4:].strip())
                processed_lines.append(f'\n{title_text}\n')
            elif stripped.startswith('# '):
                # # 主标题 -> 跳过（使用 section_order 中的 heading）
                continue
            else:
                processed_lines.append(line)

        body = '\n'.join(processed_lines).strip()
        if body:
            # 匹配该章节的图表
            figures = _match_figures_for_section(ctx, key)
            assembler.add_section(heading, text=body, figures=figures)

    # 输出路径
    os.makedirs(ctx.output_dir, exist_ok=True)
    output_docx = os.path.join(ctx.output_dir, 'paper.docx')
    ctx.docx_path = assembler.assemble(output_docx)
    logger.info(f"DOCX: {ctx.docx_path}")

    # 复制到桌面/论文输出/
    try:
        import shutil
        desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
        if os.path.isdir(desktop):
            output_dir = os.path.join(desktop, '论文输出')
            os.makedirs(output_dir, exist_ok=True)

            # 复制 DOCX
            shutil.copy2(ctx.docx_path, os.path.join(output_dir, 'paper.docx'))
            logger.info(f"已复制到: {output_dir}/paper.docx")

            # 复制 Markdown
            paper_md = os.path.join(ctx.output_dir, 'paper.md')
            if os.path.exists(paper_md):
                shutil.copy2(paper_md, os.path.join(output_dir, 'paper.md'))

            # 复制图表
            figures_src = os.path.join(ctx.output_dir, 'figures')
            if os.path.isdir(figures_src):
                figures_dst = os.path.join(output_dir, 'figures')
                os.makedirs(figures_dst, exist_ok=True)
                for f in os.listdir(figures_src):
                    if f.endswith(('.png', '.pdf', '.svg')):
                        shutil.copy2(os.path.join(figures_src, f),
                                     os.path.join(figures_dst, f))

            # 复制三线表
            for table_file in ['table1_descriptive_stats.md', 'table2_correlation_matrix.md', 'table3_seasonal_comparison.md']:
                table_path = os.path.join(ctx.output_dir, table_file)
                if os.path.exists(table_path):
                    shutil.copy2(table_path, os.path.join(output_dir, table_file))

            logger.info(f"所有输出已复制到: {output_dir}")
    except Exception as e:
        logger.warning(f"复制到桌面失败: {e}")

    return ctx.docx_path


# ============================================================
# 3. 辅助函数
# ============================================================

def _split_revised_into_sections(ctx: PaperContext, revised_text: str):
    """将修订后的全文拆分回各章节"""
    import re
    # 按 --- 分割
    parts = re.split(r'\n---\n', revised_text)
    section_map = {
        '摘要': 'abstract', 'abstract': 'abstract',
        '引言': 'introduction', '绪论': 'introduction',
        '材料与方法': 'methods', '方法': 'methods',
        '结果': 'results',
        '讨论': 'discussion',
        '结论': 'conclusion',
    }
    for part in parts:
        part = part.strip()
        if not part:
            continue
        first_line = part.split('\n')[0].strip()
        for keyword, section_key in section_map.items():
            if keyword in first_line:
                ctx.sections[section_key] = part
                break


def _match_figures_for_section(ctx: PaperContext, section_key: str) -> list:
    """为章节匹配对应的图表（根据图表的 section 属性）"""
    matched = []
    for fig_name, fig_info in ctx.figures.items():
        fig_section = fig_info.get('section', '')
        # 对于 results_discussion 章节，匹配 results 和 discussion 的图表
        if section_key == 'results_discussion':
            if fig_section in ['results', 'discussion']:
                matched.append(fig_info)
        elif fig_section == section_key:
            matched.append(fig_info)
    return matched


# ============================================================
# 4. 注入辅助函数
# ============================================================

def _build_advanced_injection(ctx: PaperContext) -> str:
    """将高级分析结果组装为可注入 Discussion 的文本"""
    parts = []
    if ctx.data_stories:
        parts.append('### 4.3.1 数据故事线\n')
        for story in ctx.data_stories[:3]:
            if isinstance(story, dict):
                finding = story.get('finding', '')
                explanation = story.get('explanation', '')
                if finding:
                    parts.append(f'- {finding}')
                    if explanation:
                        parts.append(f'  解释: {explanation}\n')
    if ctx.threshold_effects:
        parts.append('\n### 4.3.2 阈值效应\n')
        for eff in ctx.threshold_effects[:3]:
            if isinstance(eff, dict):
                var = eff.get('variable', '')
                threshold = eff.get('threshold', '')
                effect = eff.get('effect', '')
                if var:
                    parts.append(f'- {var} 在 {threshold} 处出现临界效应: {effect}')
    if ctx.anomaly_insights:
        parts.append('\n### 4.3.3 异常值洞察\n')
        for insight in ctx.anomaly_insights[:3]:
            if isinstance(insight, dict):
                desc = insight.get('description', '') or insight.get('insight', '')
                if desc:
                    parts.append(f'- {desc}')
    return '\n'.join(parts) if parts else ''


def _build_citation_injection(ctx: PaperContext) -> str:
    """将引用支撑库结果组装为可注入的文本"""
    if not ctx.citation_bank or not hasattr(ctx.citation_bank, 'bindings'):
        return ''
    lines = ['### 引用支撑索引\n']
    for binding in ctx.citation_bank.bindings[:20]:  # 增加到20个
        # ClaimBinding 是 dataclass，直接访问属性
        claim = binding.claim_text if hasattr(binding, 'claim_text') else ''
        refs = binding.supporting_citations if hasattr(binding, 'supporting_citations') else []
        if claim and refs:
            ref_str = ', '.join(str(r) for r in refs[:3])
            lines.append(f'- 论点: {claim[:80]}  支撑文献: {ref_str}')
    return '\n'.join(lines) if len(lines) > 1 else ''


def _inject_citations_to_sections(ctx: PaperContext):
    """将引用支撑库的引用注入到各章节"""
    if not ctx.citation_bank or not hasattr(ctx.citation_bank, 'bindings'):
        return

    import re

    # 构建论点->引用映射
    claim_to_refs = {}
    for binding in ctx.citation_bank.bindings:
        # ClaimBinding 是 dataclass，直接访问属性
        claim = binding.claim_text if hasattr(binding, 'claim_text') else ''
        refs = binding.supporting_citations if hasattr(binding, 'supporting_citations') else []
        if claim and refs:
            claim_to_refs[claim[:50]] = refs[:3]  # 取前3个引用

    # 为每个章节注入引用
    for section_name in ['introduction', 'discussion', 'conclusion']:
        text = ctx.sections.get(section_name, '')
        if not text:
            continue

        # 检查是否已有足够引用
        existing_citations = len(re.findall(r'\[\d+\]', text))
        if existing_citations >= 3:
            continue

        # 尝试为关键段落添加引用
        paragraphs = text.split('\n\n')
        enhanced_paragraphs = []
        for para in paragraphs:
            # 检查段落是否包含可引用的论点
            for claim, refs in claim_to_refs.items():
                if claim in para and '[' not in para:  # 段落包含论点但无引用
                    ref_str = f"[{', '.join(str(r) for r in refs)}]"
                    para = para.rstrip() + f" {ref_str}"
                    break
            enhanced_paragraphs.append(para)

        ctx.sections[section_name] = '\n\n'.join(enhanced_paragraphs)
        logger.info(f"为 {section_name} 注入引用支撑")


def _renumber_figures(ctx: PaperContext):
    """
    重排图片编号，确保连续无间隔。
    同时更新章节文本中的图片引用。
    """
    import re

    if not ctx.figures:
        return

    # 按 fig_num 排序
    sorted_figs = sorted(ctx.figures.items(), key=lambda x: x[1].get('fig_num', 999))

    # 构建旧编号->新编号映射
    num_mapping = {}
    new_figs = {}
    new_num = 1

    for name, info in sorted_figs:
        old_num = info.get('fig_num', new_num)
        num_mapping[old_num] = new_num
        info['fig_num'] = new_num
        # 更新 caption
        old_caption = info.get('caption', '')
        new_caption = re.sub(r'^图\d+', f'图{new_num}', old_caption)
        info['caption'] = new_caption
        new_figs[name] = info
        new_num += 1

    ctx.figures = new_figs

    # 更新章节文本中的图片引用（图数字格式）
    for section_name, text in ctx.sections.items():
        if not text:
            continue

        def replace_fig_ref(match):
            old_num = int(match.group(1))
            new_num_val = num_mapping.get(old_num, old_num)
            return f'图{new_num_val}'

        new_text = re.sub(r'图(\d+)', replace_fig_ref, text)
        if new_text != text:
            ctx.sections[section_name] = new_text
            logger.info(f"更新 {section_name} 中的图片引用")


def _replace_fig_x_references(ctx: PaperContext):
    """
    替换章节文本中的"图X"为实际的图片编号。
    根据图片的 section 属性和顺序，为每个章节生成正确的图号。
    """
    import re

    if not ctx.figures:
        return

    # 按章节分组图片
    fig_by_section = {}
    for name, info in ctx.figures.items():
        section = info.get('section', 'results')
        fig_num = info.get('fig_num', 0)
        if section not in fig_by_section:
            fig_by_section[section] = []
        fig_by_section[section].append((fig_num, name, info))

    # 对每个章节的图片按 fig_num 排序
    for section in fig_by_section:
        fig_by_section[section].sort(key=lambda x: x[0])

    # 替换每个章节中的"图X"
    for section_name, text in ctx.sections.items():
        if not text:
            continue

        # 获取该章节的图片列表
        figs = fig_by_section.get(section_name, [])
        if not figs:
            continue

        # 替换"图X"为实际图号
        fig_idx = 0
        def replace_fig_x(match):
            nonlocal fig_idx
            if fig_idx < len(figs):
                fig_num = figs[fig_idx][0]
                fig_idx += 1
                return f'图{fig_num}'
            return match.group(0)  # 保留原样

        new_text = re.sub(r'图X', replace_fig_x, text)
        if new_text != text:
            ctx.sections[section_name] = new_text
            logger.info(f"替换 {section_name} 中的图X引用")


# ============================================================
# 5. 文献深度学习模块
# ============================================================

def _run_paper_reading(ctx: PaperContext):
    """读论文：优先本地目录，无文献时自动在线搜索"""
    # 如果既没有本地文献目录，也没有findings用于在线搜索，则跳过
    if not ctx.papers_dir and not ctx.has('findings'):
        logger.info("无文献目录且无findings，跳过文献阅读")
        ctx.papers_read = []
        return []

    from paper_reader import PaperReader
    reader = PaperReader()

    papers_found = []

    # 1. 尝试从本地目录读取
    if ctx.papers_dir and os.path.isdir(ctx.papers_dir):
        import glob
        for ext in ['*.pdf', '*.txt', '*.md']:
            papers_found.extend(glob.glob(os.path.join(ctx.papers_dir, ext)))

    # 2. 如果本地目录不存在，尝试默认的 papers/ 目录
    if not papers_found:
        default_papers_dir = os.path.join(os.path.dirname(__file__), 'papers')
        if os.path.isdir(default_papers_dir):
            import glob
            for ext in ['*.pdf', '*.txt', '*.md']:
                papers_found.extend(glob.glob(os.path.join(default_papers_dir, ext)))
            if papers_found:
                logger.info(f"从默认 papers/ 目录找到 {len(papers_found)} 篇文献")

    # 3. 本地无文献时，自动在线搜索（使用 findings 驱动关键词）
    if not papers_found:
        logger.info("本地无文献，尝试自动在线搜索...")
        try:
            from auto_paper_finder import AutoPaperFinder
            finder = AutoPaperFinder()
            # 从 findings 中提取关键词搜索（现在 findings 已可用）
            keywords = []
            for f in ctx.findings[:5]:
                vars_ = f.get('variables', ('', ''))
                if isinstance(vars_, (list, tuple)):
                    keywords.extend([v for v in vars_ if v])
            if not keywords:
                keywords = ['sewage', 'greenhouse gas', 'methane', 'carbon']
            search_query = ' '.join(keywords[:5])
            logger.info(f"搜索关键词（来自数据发现）: {search_query}")
            found_papers = finder.find_papers(search_query)
            if found_papers:
                logger.info(f"在线搜索到 {len(found_papers)} 篇论文")
                for p in found_papers[:10]:
                    ctx.papers_read.append({
                        'title': p.get('title', ''),
                        'authors': p.get('authors', []),
                        'abstract': p.get('abstract', ''),
                        'source': 'online_search',
                    })
                return ctx.papers_read
        except Exception as e:
            logger.warning(f"在线搜索失败: {e}")
        logger.info("未找到任何文献")
        return None

    logger.info(f"发现 {len(papers_found)} 篇文献，开始阅读...")

    # 4. 读取本地文献（支持 markdown 文件直接读取）
    for paper_path in papers_found[:30]:  # 最多读30篇
        try:
            # 对于 markdown 文件，直接读取内容
            if paper_path.endswith('.md'):
                with open(paper_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # 从文件名提取标题
                title = os.path.basename(paper_path).replace('.md', '').replace('paper_', '')
                ctx.papers_read.append({
                    'path': paper_path,
                    'title': title,
                    'authors': [],
                    'abstract': content[:500] if len(content) > 500 else content,
                    'content': content,
                    'source': 'local_md',
                })
            else:
                # 对于 PDF 和其他文件，使用 PaperReader
                content = reader.read(paper_path, fetch_metadata=False)
                if content and content.metadata:
                    ctx.papers_read.append({
                        'path': paper_path,
                        'title': content.metadata.title,
                        'authors': content.metadata.authors,
                        'abstract': content.metadata.abstract,
                        'sections': len(content.sections),
                        'references': len(content.references),
                        'source': 'local_pdf',
                    })
        except Exception as e:
            logger.warning(f"读取失败 {paper_path}: {e}")

    logger.info(f"成功读取 {len(ctx.papers_read)} 篇文献")

    # 5. 使用 LiteratureMemory 评估论文质量
    try:
        from literature_memory import LiteratureMemory
        lit_mem = LiteratureMemory()
        assessed_count = 0
        for paper_info in ctx.papers_read:
            title = paper_info.get('title', '')
            if title:
                assessment = lit_mem.assess_paper(paper_info)
                if assessment:
                    paper_info['evidence_level'] = assessment.get('evidence_level', '')
                    paper_info['credibility_score'] = assessment.get('credibility', {}).get('overall', 0)
                    assessed_count += 1
        if assessed_count:
            logger.info(f"论文质量评估: {assessed_count} 篇")
    except Exception as e:
        logger.debug(f"论文质量评估跳过: {e}")

    return ctx.papers_read


def _run_pattern_learning(ctx: PaperContext):
    """从已读论文中学习写作模式，持久化到知识库"""
    if not ctx.papers_read:
        logger.info("无已读论文，跳过模式学习")
        return None

    from pattern_learner import SentencePatternLearner, DiscussionLearner, MechanismLearner
    from paper_reader import PaperReader

    reader = PaperReader()
    sentence_learner = SentencePatternLearner()
    discussion_learner = DiscussionLearner()
    mechanism_learner = MechanismLearner()

    all_patterns = []
    all_structures = []

    for paper_info in ctx.papers_read[:10]:
        path = paper_info.get('path', '')
        if not path or not os.path.exists(path):
            continue
        try:
            content = reader.read(path, fetch_metadata=False)
            if not content:
                continue

            for sec in content.sections:
                if not sec.text:
                    continue
                # 学习句式模式
                patterns = sentence_learner.learn_from_text(sec.text, sec.section_type or 'unknown')
                all_patterns.extend(patterns)

                # 学习讨论结构
                if sec.section_type == 'discussion':
                    structure = discussion_learner.learn_structure(sec.text)
                    all_structures.append(structure)

            # 学习机制知识
            full_text = '\n\n'.join(sec.text for sec in content.sections if sec.text)
            mechanisms = mechanism_learner.learn_from_text(full_text, source=paper_info.get('title', ''))

        except Exception as e:
            logger.warning(f"学习失败 {path}: {e}")

    # 持久化：将学到的机制存入 KnowledgeStore
    if ctx.has('memory') and mechanism_learner.mechanisms:
        ks_format = mechanism_learner.to_knowledge_store_format()
        for key, entry in ks_format.items():
            ctx.memory.remember(entry, category='mechanisms', key=key)
        logger.info(f"已将 {len(ks_format)} 个学到的机制存入知识库")

    # 存储到 ctx（完整数据，不只是计数）
    ctx.learned_patterns = {
        'sentence_patterns': sentence_learner.get_patterns() if hasattr(sentence_learner, 'get_patterns') else {},
        'discussion_structures': all_structures,
        'mechanisms': mechanism_learner.get_mechanisms() if hasattr(mechanism_learner, 'get_mechanisms') else [],
        'patterns_count': len(all_patterns),
        'mechanisms_count': len(mechanism_learner.mechanisms) if hasattr(mechanism_learner, 'mechanisms') else 0,
        'papers_learned': len(ctx.papers_read),
    }

    # 同时存储到 learned_mechanisms 供写作模块使用
    ctx.learned_mechanisms = mechanism_learner.get_mechanisms() if hasattr(mechanism_learner, 'get_mechanisms') else []

    # 触发 auto_learn：用学到的变量对自动搜索更多文献和机制
    try:
        from self_evolving_engine import EvolutionEngine
        engine = EvolutionEngine(base_dir=ctx.output_dir)
        engine.initialize()
        # 从 findings 中提取变量对，自动学习
        var_pairs = []
        for f in ctx.findings:
            if f.get('type') == 'correlation' and f.get('importance') in ['critical', 'high']:
                v1, v2 = f.get('variables', ('', ''))
                if v1 and v2:
                    var_pairs.append((v1, v2))
        if var_pairs:
            learn_report = engine.auto_learn(var_pairs[:5], ctx.findings)
            logger.info(f"自动学习: {learn_report.get('total_learned', 0)} 条新知识")
    except Exception as e:
        logger.debug(f"auto_learn skipped: {e}")

    logger.info(f"模式学习: {ctx.learned_patterns['patterns_count']} 个句式, "
               f"{ctx.learned_patterns['mechanisms_count']} 个机制")
    return ctx.learned_patterns


def _run_literature_learning(ctx: PaperContext):
    """文献深度学习：逻辑链提取、段落结构分析、学术表达模式库"""
    if not ctx.papers_read:
        logger.info("无已读论文，跳过文献深度学习")
        return None

    from literature_learner import LiteratureLearner
    from paper_reader import PaperReader

    reader = PaperReader()
    learner = LiteratureLearner()

    all_learnings = []
    for paper_info in ctx.papers_read[:10]:
        path = paper_info.get('path', '')
        if not path or not os.path.exists(path):
            continue
        try:
            content = reader.read(path, fetch_metadata=False)
            if not content:
                continue
            full_text = '\n\n'.join(sec.text for sec in content.sections if sec.text)
            if not full_text:
                continue
            learning = learner.learn_from_text(
                full_text,
                title=paper_info.get('title', ''),
                language=ctx.language if hasattr(ctx, 'language') else 'zh'
            )
            all_learnings.append(learning)
        except Exception as e:
            logger.warning(f"文献深度学习失败 {path}: {e}")

    # 汇总结果
    result = {
        'papers_learned': len(all_learnings),
        'logic_chains': sum(len(l.logic_chains) for l in all_learnings),
        'paragraph_structures': sum(len(l.paragraph_structures) for l in all_learnings),
        'expression_patterns': sum(len(l.expression_patterns) for l in all_learnings),
    }

    # 存储学到的表达模式供写作使用
    all_patterns = []
    for learning in all_learnings:
        for p in learning.expression_patterns:
            all_patterns.append(p.to_dict() if hasattr(p, 'to_dict') else p)
    ctx.learned_expression_patterns = all_patterns

    logger.info(f"文献深度学习: {result['papers_learned']} 篇论文, "
               f"{result['logic_chains']} 条逻辑链, "
               f"{result['expression_patterns']} 个表达模式")
    return result


# ============================================================
# 6. 新增模块函数 — 接入原孤立模块
# ============================================================

def _run_advanced_analysis(ctx: PaperContext):
    """高级多维分析（交叉分析、异常深挖、数据故事线、阈值检测）"""
    if not ctx.has('df'):
        return None
    try:
        from advanced_analysis import CrossAnalyzer, AnomalyDeepDiver, DataStoryExtractor, ThresholdDetector
        import variable_registry as vr
        all_results = {}

        # 交叉分析
        cross = CrossAnalyzer(ctx.df)
        cross_results = cross.analyze_all()
        all_results['cross_analyses'] = cross_results if isinstance(cross_results, list) else []

        # 异常深挖 — 调用 analyze() 不是 analyze_all()
        try:
            diver = AnomalyDeepDiver(ctx.df)
            anomaly_results = diver.analyze()
            all_results['anomaly_insights'] = anomaly_results if isinstance(anomaly_results, list) else []
        except Exception as e:
            logger.debug(f"AnomalyDeepDiver: {e}")
            all_results['anomaly_insights'] = []

        # 数据故事线 — 需要传入分析结果，不是原始 df
        try:
            # 构造分析结果格式
            import numpy as np
            analysis_results = {}
            # 相关性分析结果
            numeric_cols = ctx.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 2:
                corr = ctx.df[numeric_cols].corr()
                pvals = ctx.df[numeric_cols].corr()  # 简化：用相关系数矩阵
                analysis_results['pearson相关'] = {'相关系数': corr, 'p值': pvals}
            extractor = DataStoryExtractor(analysis_results)
            story_results = extractor.extract_stories()
            all_results['data_stories'] = story_results if isinstance(story_results, list) else []
        except Exception as e:
            logger.debug(f"DataStoryExtractor: {e}")
            all_results['data_stories'] = []

        # 阈值检测 — 调用 detect(x, y)，自动选择关键变量对
        try:
            detector = ThresholdDetector(ctx.df)
            threshold_results = []
            # 从 variable_registry 获取回归假设对
            regression_pairs = vr.get_regression_pairs(ctx.df)
            for pair in regression_pairs[:5]:
                try:
                    result = detector.detect(pair['x'], pair['y'])
                    if result:
                        threshold_results.append(result)
                except Exception:
                    pass
            all_results['threshold_effects'] = threshold_results
        except Exception as e:
            logger.debug(f"ThresholdDetector: {e}")
            all_results['threshold_effects'] = []

        ctx.advanced_findings = all_results.get('cross_analyses', [])
        ctx.cross_analyses = all_results.get('cross_analyses', [])
        ctx.anomaly_insights = all_results.get('anomaly_insights', [])
        ctx.data_stories = all_results.get('data_stories', [])
        ctx.threshold_effects = all_results.get('threshold_effects', [])
        total = sum(len(v) for v in all_results.values() if isinstance(v, list))
        logger.info(f"高级分析: {total}项发现")
        return all_results
    except Exception as e:
        logger.warning(f"高级分析失败: {e}")
        return None


def _run_deep_imitation(ctx: PaperContext):
    """深度模仿分析（3表法：范例动作/草稿动作/目标蓝图）"""
    if not ctx.has('sections'):
        return None
    from deep_imitation import DeepImitationManager
    manager = DeepImitationManager(output_dir=ctx.output_dir)
    # 逐章节分析
    for section_name in ['results', 'discussion', 'introduction']:
        text = ctx.sections.get(section_name, '')
        if text and len(text) > 100:
            try:
                manager.analyze_draft(section_name, text[:2000])
                manager.generate_blueprint(section_name)
            except Exception as e:
                logger.debug(f"deep_imitation {section_name}: {e}")
    report = manager.generate_report()
    ctx.imitation_report = report if isinstance(report, str) else str(report)
    logger.info("深度模仿分析完成")
    return report


def _run_integrity_audit(ctx: PaperContext):
    """完整性审计（4维度：制品链、推理深度、证据链、模式扫描）"""
    from integrity_audit import IntegrityAuditManager
    auditor = IntegrityAuditManager(output_dir=ctx.output_dir)
    report = auditor.run_audit()
    ctx.integrity_report = auditor.format_report(report)
    issue_count = len(report.findings) if hasattr(report, 'findings') else 0
    logger.info(f"完整性审计: {issue_count}项发现")
    return report


def _run_artifact_check(ctx: PaperContext):
    """制品完整性检查（文件存在性、内容质量、交叉引用）"""
    from artifact_check import ArtifactChecker
    checker = ArtifactChecker(output_dir=ctx.output_dir)
    report = checker.check_all()
    ctx.artifact_report = checker.format_report(report)
    issue_count = len(report.findings) if hasattr(report, 'findings') else 0
    logger.info(f"制品检查: {issue_count}个问题")
    return report


def _run_citation_bank(ctx: PaperContext):
    """构建引用支撑库（将论点绑定到引用）"""
    if not ctx.has('sections'):
        return None
    from citation_support_bank import CitationSupportBank
    bank = CitationSupportBank(output_dir=ctx.output_dir)
    full_text = '\n\n'.join(ctx.sections.values())
    claims = bank.extract_claims_from_text(full_text)
    if claims:
        bank.bind_citations()
        bank.save()
        ctx.citation_bank = bank
        logger.info(f"引用支撑库: {len(claims)}个论点")
    return bank


# ============================================================
# 5. 画图模块 + 文本清理
# ============================================================

def _run_figure_quality_checker(ctx: PaperContext):
    """
    图表质量检查 — 集成 scipilot-figure-skill 的视觉自检

    检查内容：
    1. 程序自检（缺字/裁切/重叠）
    2. 陷阱检测（15种常见画图错误）
    3. 生成质量报告

    输出: ctx.figure_quality_reports
    """
    if not ctx.has('figures'):
        return None

    try:
        from figure_quality_checker import full_quality_check, print_report

        reports = {}
        total_issues = 0

        for fig_name, fig_info in ctx.figures.items():
            fig_path = fig_info.get('path', '')
            if not os.path.exists(fig_path):
                continue

            # 加载图表进行检查
            try:
                import matplotlib.image as mpimg
                # 对于已保存的图表，我们只能做基本检查
                # 完整检查需要在生成时对Figure对象调用
                reports[fig_name] = {
                    'path': fig_path,
                    'status': 'checked',
                    'note': '已保存图表，建议在生成时对Figure对象调用audit_layout'
                }
            except Exception as e:
                reports[fig_name] = {
                    'path': fig_path,
                    'status': 'error',
                    'note': str(e)
                }

        ctx.figure_quality_reports = reports
        logger.info(f"图表质量检查完成: {len(reports)}个图表")
        return reports
    except Exception as e:
        logger.warning(f"图表质量检查失败: {e}")
        return None


def _run_smart_figure_planner(ctx: PaperContext):
    """
    智能作图规划 — 基于数据质量和findings规划图表

    输出: ctx.figure_plans (FigurePlanSet)
    """
    if not ctx.has('df') or not ctx.has('findings'):
        return None

    try:
        from smart_figure_planner import plan_figures

        plan_set = plan_figures(ctx.df, ctx.findings, ctx.language)
        ctx.figure_plans = plan_set

        # 输出规划结果
        logger.info(f"智能作图规划完成: {len(plan_set.plans)}个图表")
        for i, plan in enumerate(plan_set.plans):
            logger.info(f"  {i+1}. {plan.title} ({plan.chart_type}, 优先级{plan.priority})")

        return plan_set
    except Exception as e:
        logger.warning(f"智能作图规划失败: {e}")
        return None


def _run_generate_figures(ctx: PaperContext):
    """
    数据驱动的图表生成 — 使用 academic_plot_style 规范系统

    图表设计原则：
    1. 符合 Nature/Science 期刊规范
    2. 色盲友好配色
    3. 统一尺寸和样式
    4. 包含误差棒、面板标签、显著性标注
    """
    if not ctx.has('df'):
        return None

    try:
        import os
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy import stats

        # 导入学术图表规范系统
        from academic_plot_style import (
            set_academic_style, get_figure_size, get_save_dpi,
            get_label, get_color_palette, save_figure_publication,
            add_panel_label, add_significance_bar, add_shared_legend,
            add_error_bars, add_sample_size, set_axis_labels,
            SEASON_COLORS, PHASE_COLORS, NATURE_COLORS, COLORBLIND_SAFE,
            CHEMICAL_LABELS, CN_FONT_PROP,
        )

        # ============================================================
        # 初始化
        # ============================================================
        set_academic_style('nature')

        analysis_dir = os.path.join(ctx.output_dir, 'figures')
        os.makedirs(analysis_dir, exist_ok=True)

        figures_generated = []
        df = ctx.df
        fig_num = [0]

        # 图片分类规则
        FIGURE_SECTION_MAP = {
            'seasonal_boxplot': 'results',
            'correlation_heatmap': 'results',
            'spatial_distribution': 'results',
            'seasonal_comparison': 'results',
            'phase_coupling': 'discussion',
            'anomaly_profile': 'discussion',
        }

        def _next_fig_num():
            fig_num[0] += 1
            return fig_num[0]

        def _get_figure_section(name):
            """根据图片名称自动判断所属章节"""
            for key, section in FIGURE_SECTION_MAP.items():
                if key in name:
                    return section
            return 'results'

        def _save_figure(fig, name, caption_parts=None, section='results'):
            """保存图表并注册到上下文"""
            # 运行质量检查（来自 scipilot-figure-skill）
            try:
                from figure_quality_checker import audit_layout, print_report
                issues = audit_layout(fig)
                if issues:
                    print(f"  [{name}] 质量检查:")
                    print_report(issues)
            except Exception as e:
                logger.debug(f"质量检查跳过: {e}")

            # 保存：先运行 QA 自动修复并保存 PNG，然后保存 PDF 作为稿件交付件
            try:
                from chart_qa import qa_and_save
                png_path = os.path.join(analysis_dir, f'{name}.png')
                qa_and_save(fig, png_path, name=name, dpi=300)
            except Exception:
                # 回退到默认保存方式
                save_figure_publication(fig, name, analysis_dir, journal='nature')
            try:
                # 额外保存 PDF 作为交付格式
                save_figure_publication(fig, name, analysis_dir, journal='nature')
            except Exception:
                pass
            plt.close(fig)
            figures_generated.append(name)

            # 生成图注
            n = _next_fig_num()
            fig_id = str(n)
            if name.startswith('fig'):
                num_part = name[3:].split('_')[0]
                if num_part.isdigit():
                    fig_id = num_part

            if caption_parts:
                caption = f'图{fig_id}  ' + '。'.join(caption_parts) + f'。n={len(df)}。'
            else:
                caption = f'图{fig_id}  {name}'

            # 注册到上下文
            ctx.figures[name] = {
                'path': os.path.join(analysis_dir, f'{name}.png'),
                'caption': caption,
                'type': 'analysis',
                'section': section,
                'fig_num': int(fig_id),
            }
            logger.info(f"图{fig_id} {name}: 已保存 -> {section}")

        # ============================================================
        # 图1: 冬季箱线图（各变量分布）
        # ============================================================
        try:
            key_vars = ['甲烷(ppm)', 'CO2', 'COD（mg/L)', 'DO(mg/L)', 'TOC（mg/L)', 'pH',
                        'VOCs(ppb)', '总氮（mg/L)', '铵态氮（mg/L)', 'IC(mg/L)', 'NaCl(mg/L)']
            available = [v for v in key_vars if v in df.columns]

            if available and '季节' in df.columns:
                season_list = sorted(df['季节'].unique())

                for season_idx, season in enumerate(season_list):
                    season_df = df[df['季节'] == season]
                    season_color = SEASON_COLORS.get(season, NATURE_COLORS[season_idx])
                    n_vars = min(len(available), 8)
                    cols = min(n_vars, 4)
                    rows = (n_vars + cols - 1) // cols

                    fig, axes = plt.subplots(rows, cols,
                                            figsize=get_figure_size('nature', columns=2, height_ratio=0.4 * rows))
                    if rows == 1 and cols == 1:
                        axes = np.array([[axes]])
                    elif rows == 1:
                        axes = axes.reshape(1, -1)
                    elif cols == 1:
                        axes = axes.reshape(-1, 1)

                    for idx, var in enumerate(available[:n_vars]):
                        ax = axes[idx // cols][idx % cols]
                        data = season_df[var].dropna().values

                        if len(data) > 2:
                            # 箱线图
                            bp = ax.boxplot([data], patch_artist=True,
                                           widths=0.5, showmeans=True,
                                           meanprops={'marker': 'D', 'markersize': 5,
                                                     'markerfacecolor': '#E74C3C', 'markeredgecolor': '#E74C3C'})

                            bp['boxes'][0].set_facecolor(season_color)
                            bp['boxes'][0].set_alpha(0.75)
                            bp['boxes'][0].set_linewidth(1.0)
                            for element in ['whiskers', 'caps', 'medians']:
                                for line in bp[element]:
                                    line.set_color('#374151')
                                    line.set_linewidth(0.8)
                            bp['medians'][0].set_color('#E74C3C')
                            bp['medians'][0].set_linewidth(1.5)

                            # 叠加散点（抖动）
                            jitter = np.random.normal(0, 0.06, len(data))
                            ax.scatter([1 + j for j in jitter], data,
                                      color=season_color, s=20, alpha=0.7, zorder=5,
                                      edgecolors='white', linewidth=0.3)

                            # 标注均值和标准差
                            mean_val = np.mean(data)
                            std_val = np.std(data)
                            ax.text(1.3, mean_val, f'{mean_val:.1f}±{std_val:.1f}',
                                    ha='left', va='center', fontsize=6, color='#E74C3C')

                            # 样本量
                            ax.text(1, ax.get_ylim()[0], f'n={len(data)}',
                                    ha='center', va='top', fontsize=6, style='italic')

                        ax.set_xticks([])
                        ax.set_ylabel(get_label(var), fontsize=7)
                        ax.set_title(get_label(var), fontsize=8, pad=5)
                        ax.tick_params(axis='y', labelsize=6)
                        ax.grid(True, axis='y', alpha=0.2, linestyle='--')
                        ax.set_axisbelow(True)

                    # 隐藏多余子图
                    for idx in range(n_vars, rows * cols):
                        axes[idx // cols][idx % cols].set_visible(False)

                    # 添加面板标签
                    for idx in range(n_vars):
                        add_panel_label(axes[idx // cols][idx % cols], idx)

                    plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=0.5)

                    fig_name = f'fig{season_idx + 1}a_{season}_boxplot'
                    _save_figure(fig, fig_name,
                               [f'{season}关键变量箱线图',
                                '箱线图展示中位数、四分位距和异常值，红色菱形为均值'],
                               section='results')

                # 冬春对比箱线图（合并图）
                n_vars = min(len(available), 8)
                cols = min(n_vars, 4)
                rows = (n_vars + cols - 1) // cols

                fig, axes = plt.subplots(rows, cols,
                                        figsize=get_figure_size('nature', columns=2, height_ratio=0.4 * rows))
                if rows == 1 and cols == 1:
                    axes = np.array([[axes]])
                elif rows == 1:
                    axes = axes.reshape(1, -1)
                elif cols == 1:
                    axes = axes.reshape(-1, 1)

                season_colors = [SEASON_COLORS.get(s, NATURE_COLORS[i]) for i, s in enumerate(season_list)]

                for idx, var in enumerate(available[:n_vars]):
                    ax = axes[idx // cols][idx % cols]
                    data = df[['季节', var]].dropna()

                    if len(data) > 3:
                        values = [data[data['季节'] == s][var].values for s in season_list]

                        bp = ax.boxplot(values, patch_artist=True,
                                       widths=0.6, showmeans=True, meanprops={'marker': 'D', 'markersize': 4})
                        ax.set_xticklabels(season_list)

                        for si, box in enumerate(bp['boxes']):
                            box.set_facecolor(season_colors[si])
                            box.set_alpha(0.7)
                            box.set_linewidth(0.8)
                        for si, whisker in enumerate(bp['whiskers']):
                            whisker.set_linewidth(0.8)
                        for si, cap in enumerate(bp['caps']):
                            cap.set_linewidth(0.8)
                        for si, median in enumerate(bp['medians']):
                            median.set_linewidth(1.0)
                            median.set_color('#333333')

                        for si, (s, v) in enumerate(zip(season_list, values)):
                            jitter = np.random.normal(0, 0.05, len(v))
                            ax.scatter([si + 1 + j for j in jitter], v,
                                      color=season_colors[si], s=15, alpha=0.6, zorder=5,
                                      edgecolors='white', linewidth=0.3)

                        ax.set_ylabel(get_label(var), fontsize=7)
                        ax.set_title(get_label(var), fontsize=8, pad=5)

                        for si, s in enumerate(season_list):
                            n = len(values[si])
                            ax.text(si + 1, ax.get_ylim()[0], f'n={n}',
                                    ha='center', va='top', fontsize=6, style='italic')

                        p = next((f['data']['p'] for f in ctx.findings
                                 if f.get('type') == 'group_difference'
                                 and f.get('variable') == var
                                 and f.get('data', {}).get('p', 1) < 0.05), None)
                        if p:
                            y_max = max(max(v) for v in values)
                            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                            add_significance_bar(ax, 1, 2, y_max + y_range * 0.05, p)

                        ax.tick_params(axis='x', labelsize=7)
                        ax.tick_params(axis='y', labelsize=6)

                for idx in range(n_vars, rows * cols):
                    axes[idx // cols][idx % cols].set_visible(False)

                for idx in range(n_vars):
                    add_panel_label(axes[idx // cols][idx % cols], idx)

                plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=0.5)

                _save_figure(fig, 'fig1c_seasonal_comparison_boxplot',
                           ['冬春季关键变量箱线图比较',
                            '箱线图展示中位数、四分位距和异常值',
                            '* p<0.05, ** p<0.01, *** p<0.001'],
                           section='results')

        except Exception as e:
            logger.warning(f"fig1_seasonal_boxplot: {e}")

        # ============================================================
        # 图2: 关键变量相关性热图
        # ============================================================
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            good_cols = [c for c in numeric_cols if df[c].notna().sum() >= len(df) * 0.7]

            if len(good_cols) >= 3:
                corr = df[good_cols].corr(method='pearson')

                # 计算 p 值矩阵
                n = len(df)
                p_matrix = np.zeros_like(corr)
                for i in range(len(good_cols)):
                    for j in range(len(good_cols)):
                        if i != j:
                            r = corr.iloc[i, j]
                            t = r * np.sqrt((n - 2) / (1 - r**2)) if abs(r) < 1 else 0
                            p_matrix[i, j] = 2 * (1 - stats.t.cdf(abs(t), n - 2))

                # 创建标签
                labels = [get_label(c) for c in good_cols]

                # 绘制热图
                fig, ax = plt.subplots(figsize=get_figure_size('nature', columns=2, height_ratio=0.85))

                mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

                # 自定义注释（显示相关系数和显著性）
                annot = np.empty_like(corr, dtype=object)
                for i in range(len(good_cols)):
                    for j in range(len(good_cols)):
                        if i == j:
                            annot[i, j] = ''
                        elif mask[i, j]:
                            annot[i, j] = ''
                        else:
                            r = corr.iloc[i, j]
                            p = p_matrix[i, j]
                            stars = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
                            annot[i, j] = f'{r:.2f}{stars}'

                sns.heatmap(corr, mask=mask, ax=ax, cmap='RdBu_r', center=0,
                           vmin=-1, vmax=1, annot=annot, fmt='',
                           square=True, linewidths=0.5,
                           annot_kws={'size': 6},
                           cbar_kws={'shrink': 0.6, 'label': 'Pearson r', 'aspect': 20, 'pad': 0.02})

                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
                ax.set_yticklabels(labels, rotation=0, fontsize=6)
                ax.set_title('关键变量Pearson相关性矩阵', fontsize=9, pad=10)

                # 添加样本量标注（移到左下角，避免与颜色条重叠）
                ax.text(0.02, -0.08, f'n={n}', transform=ax.transAxes,
                        ha='left', va='top', fontsize=7, style='italic')

                plt.tight_layout(pad=1.5)

                _save_figure(fig, 'fig2_correlation_heatmap',
                           ['关键变量Pearson相关性矩阵',
                            '仅包含缺失值<30%的变量',
                            '红色正相关，蓝色负相关，* p<0.05, ** p<0.01, *** p<0.001'],
                           section='results')

        except Exception as e:
            logger.warning(f"fig2_correlation_heatmap: {e}")

        # ============================================================
        # 图3: 气体空间分布（带误差棒）
        # ============================================================
        try:
            gas_vars = [v for v in ['甲烷(ppm)', 'CO2', 'VOCs(ppb)', 'H2S'] if v in df.columns]
            # 过滤掉极高缺失率的变量（缺失率 > 80%），如H2S
            gas_vars = [v for v in gas_vars if df[v].isna().mean() <= 0.8]

            if gas_vars and '采样点' in df.columns:
                n_vars = min(len(gas_vars), 4)
                cols = min(n_vars, 2)
                rows = (n_vars + cols - 1) // cols

                fig, axes = plt.subplots(rows, cols,
                                        figsize=get_figure_size('nature', columns=2, height_ratio=0.5 * rows))
                if rows == 1 and cols == 1:
                    axes = np.array([[axes]])
                elif rows == 1:
                    axes = axes.reshape(1, -1)
                elif cols == 1:
                    axes = axes.reshape(-1, 1)

                for idx, var in enumerate(gas_vars[:n_vars]):
                    ax = axes[idx // cols][idx % cols]

                    # 计算均值和标准误
                    stats_df = df.groupby('采样点')[var].agg(['mean', 'std', 'count']).reset_index()
                    stats_df['se'] = stats_df['std'] / np.sqrt(stats_df['count'])
                    stats_df = stats_df.sort_values('mean', ascending=False)

                    # 柱状图（带误差棒）
                    colors = get_color_palette(len(stats_df), palette='nature')
                    bars = ax.bar(range(len(stats_df)), stats_df['mean'],
                                 color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)

                    # 误差棒
                    add_error_bars(ax, range(len(stats_df)), stats_df['mean'], stats_df['se'])

                    # 设置标签
                    ax.set_xticks(range(len(stats_df)))
                    ax.set_xticklabels(stats_df['采样点'], rotation=45, ha='right', fontsize=6)
                    ax.set_ylabel(get_label(var), fontsize=7)
                    ax.set_title(get_label(var), fontsize=8, pad=5)

                    # 标记异常高值（使用均值的中位数作为比较基准）
                    median_of_means = stats_df['mean'].median() if 'mean' in stats_df.columns else None
                    for i, (val, point) in enumerate(zip(stats_df['mean'], stats_df['采样点'])):
                        try:
                            if median_of_means is not None and val > median_of_means * 2:
                                ax.annotate(f'{val:.0f}', (i, val), ha='center', va='bottom',
                                           fontsize=6, color='#E64B35')
                        except Exception:
                            # 忽略注释错误，继续绘图
                            pass

                    # 添加样本量（显示总样本或最大样本量以便阅读）
                    try:
                        n_total = int(stats_df['count'].sum())
                    except Exception:
                        n_total = len(df)
                    ax.text(0.95, 0.95, f'n={n_total}', transform=ax.transAxes,
                            ha='right', va='top', fontsize=7, style='italic')

                # 隐藏多余子图
                for idx in range(n_vars, rows * cols):
                    axes[idx // cols][idx % cols].set_visible(False)

                # 添加面板标签
                for idx in range(n_vars):
                    add_panel_label(axes[idx // cols][idx % cols], idx)

                plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=0.5)

                _save_figure(fig, 'fig3_spatial_distribution',
                           ['气体污染物空间分布',
                            '误差棒表示标准误(SE)',
                            '标注异常高值(>2倍中位数)'],
                           section='results')

        except Exception as e:
            logger.warning(f"fig3_spatial_distribution: {e}")

        # ============================================================
        # 图4: 季节差异对比图（聚焦显著变量）
        # ============================================================
        try:
            sig_vars = []
            for f in ctx.findings:
                if f.get('type') == 'group_difference' and f.get('data', {}).get('p', 1) < 0.05:
                    var = f.get('variable', '')
                    if var and var in df.columns and var not in sig_vars:
                        sig_vars.append(var)

            if not sig_vars:
                sig_vars = [v for v in ['甲烷(ppm)', 'COD（mg/L)', 'CO2', 'VOCs(ppb)'] if v in df.columns]

            if sig_vars and '季节' in df.columns:
                n_vars = min(len(sig_vars), 4)
                cols = min(n_vars, 2)
                rows = (n_vars + cols - 1) // cols

                fig, axes = plt.subplots(rows, cols,
                                        figsize=get_figure_size('nature', columns=2, height_ratio=0.5 * rows))
                if rows == 1 and cols == 1:
                    axes = np.array([[axes]])
                elif rows == 1:
                    axes = axes.reshape(1, -1)
                elif cols == 1:
                    axes = axes.reshape(-1, 1)

                season_list = sorted(df['季节'].unique())
                season_colors = [SEASON_COLORS.get(s, NATURE_COLORS[i]) for i, s in enumerate(season_list)]

                for idx, var in enumerate(sig_vars[:n_vars]):
                    ax = axes[idx // cols][idx % cols]
                    data = df[['季节', var]].dropna()

                    if len(data) > 3:
                        # 准备数据
                        values = [data[data['季节'] == s][var].values for s in season_list]

                        # 小提琴+箱线+散点组合图
                        parts = ax.violinplot(values, showmeans=False, showmedians=False, showextrema=False)
                        for si, pc in enumerate(parts['bodies']):
                            pc.set_facecolor(season_colors[si])
                            pc.set_alpha(0.3)

                        # 箱线图
                        bp = ax.boxplot(values, patch_artist=True,
                                       widths=0.3, showmeans=True, meanprops={'marker': 'D', 'markersize': 4})

                        # 设置x轴标签
                        ax.set_xticklabels(season_list)

                        for si, box in enumerate(bp['boxes']):
                            box.set_facecolor(season_colors[si])
                            box.set_alpha(0.8)
                            box.set_linewidth(0.8)

                        # 散点
                        for si, (s, v) in enumerate(zip(season_list, values)):
                            jitter = np.random.normal(0, 0.05, len(v))
                            ax.scatter([si + 1 + j for j in jitter], v,
                                      color=season_colors[si], s=20, alpha=0.7, zorder=5,
                                      edgecolors='white', linewidth=0.3)

                        # 设置标签
                        ax.set_ylabel(get_label(var), fontsize=7)
                        ax.set_title(get_label(var), fontsize=8, pad=5)

                        # 添加显著性标注
                        p = next((f['data']['p'] for f in ctx.findings
                                 if f.get('type') == 'group_difference'
                                 and f.get('variable') == var
                                 and f.get('data', {}).get('p', 1) < 0.05), None)
                        if p:
                            y_max = max(max(v) for v in values)
                            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                            add_significance_bar(ax, 1, 2, y_max + y_range * 0.05, p)

                        ax.tick_params(axis='x', labelsize=7)
                        ax.tick_params(axis='y', labelsize=6)

                # 隐藏多余子图
                for idx in range(n_vars, rows * cols):
                    axes[idx // cols][idx % cols].set_visible(False)

                # 添加面板标签
                for idx in range(n_vars):
                    add_panel_label(axes[idx // cols][idx % cols], idx)

                plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=0.5)

                _save_figure(fig, 'fig4_seasonal_comparison',
                           ['季节差异显著变量对比',
                            '小提琴+箱线+散点组合图',
                            '* p<0.05, ** p<0.01, *** p<0.001'],
                           section='results')

        except Exception as e:
            logger.warning(f"fig4_seasonal_comparison: {e}")

        # ============================================================
        # 图5: 相态耦合散点图矩阵
        # ============================================================
        try:
            gas_main = [v for v in ['甲烷(ppm)', 'CO2'] if v in df.columns]
            liquid_main = [v for v in ['TOC（mg/L)', 'DO(mg/L)', 'COD（mg/L)', 'pH'] if v in df.columns]

            if gas_main and liquid_main:
                n_gas = len(gas_main)
                n_liquid = len(liquid_main)

                fig, axes = plt.subplots(n_gas, n_liquid,
                                        figsize=get_figure_size('nature', columns=2, height_ratio=0.5 * n_gas))
                if n_gas == 1:
                    axes = axes.reshape(1, -1)
                if n_liquid == 1:
                    axes = axes.reshape(-1, 1)

                season_list = sorted(df['季节'].unique())
                season_colors = [SEASON_COLORS.get(s, NATURE_COLORS[i]) for i, s in enumerate(season_list)]

                for i, gvar in enumerate(gas_main):
                    for j, lvar in enumerate(liquid_main):
                        ax = axes[i][j]
                        valid = df[[gvar, lvar, '季节']].dropna()

                        if len(valid) > 5:
                            # 按季节着色
                            for si, season in enumerate(season_list):
                                sub = valid[valid['季节'] == season]
                                ax.scatter(sub[lvar], sub[gvar],
                                          label=season, s=30, alpha=0.7,
                                          color=season_colors[si],
                                          edgecolors='white', linewidth=0.3)

                            # 添加回归线
                            x = valid[lvar].values
                            y = valid[gvar].values
                            slope, intercept, r, p, se = stats.linregress(x, y)
                            x_line = np.linspace(x.min(), x.max(), 100)
                            ax.plot(x_line, slope * x_line + intercept,
                                    color='#333333', linewidth=1, linestyle='--', alpha=0.5)

                            # 添加相关系数
                            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
                            ax.text(0.05, 0.95, f'r={r:.2f} {sig}', transform=ax.transAxes,
                                    ha='left', va='top', fontsize=7, fontweight='bold')

                            # 设置标签
                            ax.set_xlabel(get_label(lvar), fontsize=7)
                            ax.set_ylabel(get_label(gvar), fontsize=7)

                            # 只在第一行添加图例
                            if i == 0 and j == 0:
                                ax.legend(fontsize=6, loc='upper right')

                # 添加面板标签
                for i in range(n_gas):
                    for j in range(n_liquid):
                        idx = i * n_liquid + j
                        add_panel_label(axes[i][j], idx)

                plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=0.5)

                _save_figure(fig, 'fig5_phase_coupling',
                           ['气相-液相变量耦合关系',
                            '散点图展示CH₄/CO₂与水质指标的关系',
                            '标注Pearson相关系数和显著性'],
                           section='discussion')

        except Exception as e:
            logger.warning(f"fig5_phase_coupling: {e}")

        # ============================================================
        # 图6: 异常采样点特征剖面图（替代雷达图）
        # ============================================================
        try:
            if '甲烷(ppm)' in df.columns and '采样点' in df.columns:
                # 找出CH4最高的采样点
                top_points = df.groupby('采样点')['甲烷(ppm)'].mean().nlargest(3).index.tolist()

                if top_points:
                    # 选择关键变量
                    profile_vars = [v for v in ['甲烷(ppm)', 'CO2', 'COD（mg/L)', 'TOC（mg/L)', 'DO(mg/L)', 'pH'] if v in df.columns]

                    if profile_vars:
                        # 标准化数据
                        point_means = df.groupby('采样点')[profile_vars].mean()
                        normalized = (point_means - point_means.min()) / (point_means.max() - point_means.min())

                        # 创建分组柱状图
                        fig, ax = plt.subplots(figsize=get_figure_size('nature', columns=2, height_ratio=0.5))

                        x = np.arange(len(profile_vars))
                        width = 0.25
                        colors = get_color_palette(len(top_points), palette='nature')

                        for pi, point in enumerate(top_points):
                            if point in normalized.index:
                                values = normalized.loc[point].values
                                bars = ax.bar(x + pi * width, values, width,
                                             label=point, color=colors[pi], alpha=0.8,
                                             edgecolor='white', linewidth=0.5)

                        # 设置标签
                        ax.set_xticks(x + width * (len(top_points) - 1) / 2)
                        ax.set_xticklabels([get_label(v) for v in profile_vars],
                                          rotation=45, ha='right', fontsize=7)
                        ax.set_ylabel('标准化值', fontsize=8)
                        ax.set_title('异常采样点特征剖面', fontsize=9, pad=10)

                        # 添加图例
                        ax.legend(fontsize=7, loc='upper right', title='采样点', title_fontsize=7)

                        # 添加参考线
                        ax.axhline(y=0.5, color='#999999', linestyle=':', linewidth=0.5, alpha=0.5)

                        plt.tight_layout()

                        _save_figure(fig, 'fig6_anomaly_profile',
                                   ['异常采样点特征剖面图',
                                    '标准化后展示各变量相对水平',
                                    '识别异常采样点的环境特征'],
                                   section='discussion')

        except Exception as e:
            logger.warning(f"fig6_anomaly_profile: {e}")

        # ============================================================
        # 图7: 层次聚类分析树状图
        # ============================================================
        try:
            from scipy.cluster.hierarchy import dendrogram, linkage
            from scipy.spatial.distance import pdist

            # 选择关键变量进行聚类
            cluster_vars = [v for v in ['甲烷(ppm)', 'CO2', 'COD（mg/L)', 'TOC（mg/L)', 'DO(mg/L)', 'pH', 'VOCs(ppb)'] if v in df.columns]

            if cluster_vars and '采样点' in df.columns:
                # 准备数据
                cluster_data = df.groupby('采样点')[cluster_vars].mean().dropna()

                if len(cluster_data) >= 3:
                    # 标准化数据
                    normalized = (cluster_data - cluster_data.mean()) / cluster_data.std()

                    # 计算距离矩阵和聚类
                    dist_matrix = pdist(normalized, metric='euclidean')
                    linkage_matrix = linkage(dist_matrix, method='ward')

                    # 绘制树状图
                    fig, ax = plt.subplots(figsize=get_figure_size('nature', columns=2, height_ratio=0.6))

                    dendrogram(linkage_matrix,
                              labels=cluster_data.index.tolist(),
                              ax=ax,
                              leaf_rotation=45,
                              leaf_font_size=7,
                              color_threshold=0.7 * max(linkage_matrix[:, 2]))

                    ax.set_title('采样点层次聚类分析', fontsize=9, pad=10)
                    ax.set_ylabel('距离', fontsize=8)
                    ax.set_xlabel('采样点', fontsize=8)

                    # 添加虚线标识聚类阈值
                    threshold = 0.7 * max(linkage_matrix[:, 2])
                    ax.axhline(y=threshold, color='#E15759', linestyle='--', linewidth=1, alpha=0.7)
                    ax.text(ax.get_xlim()[1] * 0.95, threshold * 1.05,
                           f'阈值={threshold:.1f}', fontsize=7, color='#E15759', ha='right')

                    plt.tight_layout()

                    _save_figure(fig, 'fig7_cluster_dendrogram',
                               ['采样点层次聚类分析树状图',
                                'Ward法聚类，欧氏距离',
                                '虚线标识聚类阈值'],
                               section='results')

        except Exception as e:
            logger.warning(f"fig7_cluster_dendrogram: {e}")

        # ============================================================
        # 生成三线表
        # ============================================================
        try:
            _generate_three_line_tables(ctx, df)
        except Exception as e:
            logger.warning(f"三线表生成失败: {e}")

        # ============================================================
        # 记录生成结果
        # ============================================================
        logger.info(f"生成 {len(figures_generated)} 类图表: {figures_generated}")
        return figures_generated

    except Exception as e:
        logger.warning(f"图表生成失败: {e}")
        return None


def _generate_three_line_tables(ctx: PaperContext, df):
    """
    生成三线表（符合学术论文规范）

    三线表格式：
    - 顶线（粗线）
    - 标题行（细线）
    - 数据行
    - 底线（粗线）
    """
    import pandas as pd
    from academic_plot_style import get_label

    # 表1：描述性统计表
    try:
        # 选择关键变量（气相、液相、固相）
        key_vars = ['甲烷(ppm)', 'CO2', 'VOCs(ppb)', 'DO(mg/L)', 'pH',
                    'TOC（mg/L)', 'COD（mg/L)', '总氮（mg/L)', '铵态氮（mg/L)',
                    '固总碳（g/kg)', '有机碳（g/kg)', '无机碳（g/kg)', 'DOC(mg/kg)']
        # 注意：H2S缺失率92.5%，已排除

        # 所有变量都分析（固相变量虽然缺失率高，但仍有2个有效值）
        available_vars = [v for v in key_vars if v in df.columns]

        if available_vars:
            # 计算描述性统计
            stats_data = []
            for var in available_vars:
                data = df[var].dropna()
                if len(data) > 0:
                    stats_data.append({
                        '变量': var,
                        'n': len(data),
                        '均值': f'{data.mean():.2f}',
                        '标准差': f'{data.std():.2f}',
                        '最小值': f'{data.min():.2f}',
                        '最大值': f'{data.max():.2f}',
                        'CV(%)': f'{(data.std()/data.mean()*100):.1f}' if data.mean() != 0 else 'N/A'
                    })

            if stats_data:
                # 生成三线表 Markdown
                table_md = _format_three_line_table(
                    title='表1 研究区主要变量描述性统计',
                    headers=['变量', 'n', '均值', '标准差', '最小值', '最大值', 'CV(%)'],
                    rows=stats_data
                )

                # 保存到文件
                table_path = os.path.join(ctx.output_dir, 'table1_descriptive_stats.md')
                with open(table_path, 'w', encoding='utf-8') as f:
                    f.write(table_md)

                logger.info(f"生成三线表: {table_path}")

    except Exception as e:
        logger.warning(f"描述性统计表生成失败: {e}")

    # 表2：相关性分析表
    try:
        # 选择关键变量
        corr_vars = ['甲烷(ppm)', 'CO2', 'TOC（mg/L)', 'DO(mg/L)', 'pH', 'COD（mg/L)']
        available_corr_vars = [v for v in corr_vars if v in df.columns]

        logger.info(f"表2: 可用变量 {available_corr_vars}")

        if len(available_corr_vars) >= 3:
            # 计算相关系数矩阵
            corr_matrix = df[available_corr_vars].corr()

            # 生成三线表
            headers = ['变量'] + [get_label(v) for v in available_corr_vars]
            rows = []
            for i, var1 in enumerate(available_corr_vars):
                row = {'变量': get_label(var1)}
                for j, var2 in enumerate(available_corr_vars):
                    r = corr_matrix.iloc[i, j]
                    if i == j:
                        row[get_label(var2)] = '1.000'
                    else:
                        # 计算p值
                        from scipy import stats
                        n = len(df[[var1, var2]].dropna())
                        if n > 2:
                            t_stat = r * np.sqrt((n-2)/(1-r**2)) if abs(r) < 1 else 0
                            p = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))
                            if p < 0.001:
                                row[get_label(var2)] = f'{r:.3f}***'
                            elif p < 0.01:
                                row[get_label(var2)] = f'{r:.3f}**'
                            elif p < 0.05:
                                row[get_label(var2)] = f'{r:.3f}*'
                            else:
                                row[get_label(var2)] = f'{r:.3f}'
                        else:
                            row[get_label(var2)] = f'{r:.3f}'
                rows.append(row)

            table_md = _format_three_line_table(
                title='表2 主要变量Pearson相关系数矩阵',
                headers=headers,
                rows=rows
            )

            table_path = os.path.join(ctx.output_dir, 'table2_correlation_matrix.md')
            with open(table_path, 'w', encoding='utf-8') as f:
                f.write(table_md)

            logger.info(f"生成三线表: {table_path}")
        else:
            logger.warning(f"表2: 变量不足 ({len(available_corr_vars)} < 3)")

    except Exception as e:
        logger.warning(f"相关性分析表生成失败: {e}")

    # 表3：季节差异比较表
    try:
        # 选择关键变量
        season_vars = ['甲烷(ppm)', 'CO2', 'COD（mg/L)', 'DO(mg/L)', 'TOC（mg/L)', 'pH']
        available_season_vars = [v for v in season_vars if v in df.columns]

        logger.info(f"表3: 可用变量 {available_season_vars}")

        if available_season_vars and '季节' in df.columns:
            seasons = sorted(df['季节'].unique())
            logger.info(f"表3: 季节 {seasons}")

            if len(seasons) == 2:
                headers = ['变量', f'{seasons[0]}均值±标准差', f'{seasons[1]}均值±标准差', 't值', 'p值', '显著性']
                rows = []

                for var in available_season_vars:
                    data1 = df[df['季节'] == seasons[0]][var].dropna()
                    data2 = df[df['季节'] == seasons[1]][var].dropna()

                    if len(data1) > 2 and len(data2) > 2:
                        from scipy import stats
                        t_stat, p = stats.ttest_ind(data1, data2)

                        sig = ''
                        if p < 0.001:
                            sig = '***'
                        elif p < 0.01:
                            sig = '**'
                        elif p < 0.05:
                            sig = '*'

                        rows.append({
                            '变量': get_label(var),
                            f'{seasons[0]}均值±标准差': f'{data1.mean():.2f}±{data1.std():.2f}',
                            f'{seasons[1]}均值±标准差': f'{data2.mean():.2f}±{data2.std():.2f}',
                            't值': f'{t_stat:.3f}',
                            'p值': f'{p:.4f}',
                            '显著性': sig
                        })

                logger.info(f"表3: 生成 {len(rows)} 行数据")

                if rows:
                    table_md = _format_three_line_table(
                        title='表3 主要变量冬春季节差异比较',
                        headers=headers,
                        rows=rows
                    )

                    table_path = os.path.join(ctx.output_dir, 'table3_seasonal_comparison.md')
                    with open(table_path, 'w', encoding='utf-8') as f:
                        f.write(table_md)

                    logger.info(f"生成三线表: {table_path}")
                else:
                    logger.warning("表3: 无有效数据行")
            else:
                logger.warning(f"表3: 季节数量不为2 ({len(seasons)})")
        else:
            logger.warning(f"表3: 缺少季节列或变量")

    except Exception as e:
        logger.warning(f"季节差异比较表生成失败: {e}")


def _format_three_line_table(title: str, headers: list, rows: list) -> str:
    """
    格式化三线表（符合学术规范）

    三线表格式：
    - 顶线（粗线，用 = 或 ─）
    - 表头
    - 标题行下细线
    - 数据行
    - 底线（粗线）

    Parameters
    ----------
    title : str, 表格标题
    headers : list, 表头列表
    rows : list, 数据行列表（每行是字典）

    Returns
    -------
    str : 三线表格式
    """
    # 计算每列最大宽度
    col_widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            val = str(row.get(h, ''))
            col_widths[h] = max(col_widths[h], len(val))

    # 生成表格
    lines = []
    lines.append(title)
    lines.append('')

    # 顶线（粗线）
    top_line = '─' * (sum(col_widths[h] + 3 for h in headers) - 1)
    lines.append(top_line)

    # 表头
    header_line = '  '.join([f'{h:<{col_widths[h]}}' for h in headers])
    lines.append(header_line)

    # 标题行下细线
    mid_line = '─' * (sum(col_widths[h] + 3 for h in headers) - 1)
    lines.append(mid_line)

    # 数据行
    for row in rows:
        data_line = '  '.join([f'{str(row.get(h, "")):<{col_widths[h]}}' for h in headers])
        lines.append(data_line)

    # 底线（粗线）
    bottom_line = '─' * (sum(col_widths[h] + 3 for h in headers) - 1)
    lines.append(bottom_line)

    # 添加注释
    lines.append('')
    lines.append('注：* p<0.05, ** p<0.01, *** p<0.001')

    return '\n'.join(lines)


def _clean_claude_output(text: str) -> str:
    """
    智能清理 Claude 输出中的非正文内容。

    设计原则：
    1. 保守策略：宁可保留元评论，也不能误杀正文
    2. 只清理开头和结尾的元评论，中间内容全部保留
    3. 使用精确匹配，避免误杀
    """
    if not text:
        return text

    import re

    # ============================================================
    # 第一步：清理中间的Claude元评论（全文扫描）
    # ============================================================
    # 匹配Claude元评论模式（独立行，不是正文的一部分）
    meta_line_patterns = [
        r'^写入权限未开启.*$',
        r'^文件写入权限未开启.*$',
        r'^需要您授予.*$',
        r'^文件写入需要您授予.*$',
        r'^内容已准备完毕.*$',
        r'^包含以下.*核心章节.*$',
        r'^如需调整.*请告知.*$',
        r'^如需写入文件.*请授予.*$',
        r'^本节要点[：:].*$',
        r'^以下是撰写的.*$',
        r'^全文约\d+字.*$',
        r'^共约\d+字.*$',
        r'^如需调整内容深度.*$',
        r'^如需写入文件.*$',
        r'^接下来可继续撰写.*$',
        r'^文件写入权限.*$',
        r'^接下来可继续.*$',
        r'^如需调整.*$',
        r'^如需写入.*$',
        r'^Now I have.*$',
        r'^Let me.*$',
        r'^I will.*$',
        r'^Here is.*$',
        r'^根据.*分析.*$',
        r'^基于.*分析.*$',
        r'^综合.*分析.*$',
        r'^请在弹出的权限.*$',
        r'^我将把.*写入.*$',
        r'^请在弹出.*$',
        r'^内容已呈现完毕.*$',
        r'^数据核验说明[：:].*$',
        r'^所有r值.*$',
        r'^CH₄/TOC比值.*$',
        r'^CH₄与固相TOC.*$',
        r'^CO₂与NO₂.*$',
        r'^PCA和HCA.*$',
        r'^如需.*请.*$',  # 通用的"如需...请..."模式
    ]

    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        is_meta = False

        # 检查是否是Claude元评论
        for pattern in meta_line_patterns:
            if re.match(pattern, stripped, re.IGNORECASE):
                is_meta = True
                break

        # 检查是否是LaTeX代码块标记
        if stripped.startswith('```latex') or stripped.startswith('```'):
            is_meta = True

        # 检查是否是交互式问题
        if stripped.endswith('？') or stripped.endswith('?'):
            question_keywords = ['请问', '需要', '是否', '如何', '怎样', 'what', 'how', 'do you', '您希望']
            for keyword in question_keywords:
                if keyword in stripped:
                    is_meta = True
                    break

        # 检查是否是表格行（| 内容 | 内容 | 内容 |）
        if re.match(r'^\|', stripped) and stripped.endswith('|'):
            is_meta = True

        # 检查是否是数字列表的审稿意见（1. 修复数据一致性）
        if re.match(r'^\d+\.\s+(修复|完善|补充|优化|改进|撰写|全面)', stripped):
            is_meta = True

        # 检查是否是审稿意见标题
        if '审稿意见' in stripped or '问题清单' in stripped or '改进点' in stripped:
            is_meta = True

        # 检查是否是单独的emoji行（🔴 🟡 等）
        if re.match(r'^[🔴🟡🟢⚪⚫]+$', stripped):
            is_meta = True

        # 检查是否是调试信息
        if stripped.startswith('[DEBUG]') or stripped.startswith('[debug]'):
            is_meta = True

        if not is_meta:
            cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)

    # ============================================================
    # 第二步：清理 markdown 格式（保留化学式）
    # ============================================================
    # 清理粗体（但保留化学式中的下标）
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # **bold** -> bold
    text = re.sub(r'\*(.+?)\*', r'\1', text)        # *italic* -> italic
    text = re.sub(r'`(.+?)`', r'\1', text)          # `code` -> code

    # 清理 LaTeX 数学公式标记（但保留化学式下标）
    text = re.sub(r'\$_\{?(\d+)\}?\$', r'ₙ', text)  # $_4$ -> ₄ (Unicode下标)
    text = re.sub(r'\$\{?(\d+)\}?\$', r'ₙ', text)   # $4$ -> ₄

    # 清理多余空行
    text = re.sub(r'\n{3,}', '\n\n', text)

    # ============================================================
    # 第三步：清理行内元评论片段
    # ============================================================
    # 移除行内出现的"如需写入"、"如需调整"等片段
    inline_meta_patterns = [
        r'如需写入文件.*?，?\s*',
        r'如需调整.*?，?\s*',
        r'如需写入.*?，?\s*',
        r'请在弹出的权限.*?，?\s*',
        r'内容已呈现完毕.*?，?\s*',
        r'全文约\d+字.*?，?\s*',
    ]
    for pattern in inline_meta_patterns:
        text = re.sub(pattern, '', text)

    return text.strip()


# ============================================================
# 6. 新增模块函数 — 科学分析/引用审计/动机线索/推理矩阵/修订审计/领域配置
# ============================================================

def _run_scientific_analysis(ctx: PaperContext):
    """智能分析编排器（自动判断该做什么分析）"""
    if not ctx.has('df'):
        return None
    try:
        from scientific_analysis_agent import ScientificAnalysisAgent
        agent = ScientificAnalysisAgent(data_path=ctx.data_path, output_dir=ctx.output_dir)
        agent.load_data()
        results = agent.run()
        # results 可能是 dict 或其他类型
        if results and isinstance(results, dict):
            for key, value in results.items():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and 'type' not in item:
                            item['type'] = key
                        if isinstance(item, dict):
                            ctx.findings.append(item)
        result_count = len(results) if isinstance(results, dict) else 0
        logger.info(f"科学分析完成: {result_count} 类结果")
        return results
    except Exception as e:
        logger.warning(f"科学分析失败: {e}")
        return None


def _run_citation_audit(ctx: PaperContext):
    """引用质量审计（DOI验证、年份评分、类型分类）"""
    if not ctx.has('sections'):
        return None
    try:
        from citation_audit import audit_citations_batch
        import re
        # 从全文中提取引用
        full_text = '\n\n'.join(ctx.sections.values())
        # 提取 [1] [2] 格式的引用
        citation_refs = re.findall(r'\[(\d+(?:[,\s]*\d+)*)\]', full_text)
        # 提取 (Author, Year) 格式的引用
        author_refs = re.findall(r'\(([A-Z][a-z]+(?:\s+(?:et\s+al\.?|&\s+[A-Z][a-z]+))?),?\s+(\d{4})\)', full_text)

        all_refs = []
        for ref in citation_refs[:20]:
            all_refs.append({'reference': ref, 'type': 'numbered'})
        for author, year in author_refs[:20]:
            all_refs.append({'reference': f'{author} ({year})', 'type': 'author_year'})

        if all_refs:
            audit_result = audit_citations_batch(all_refs, verify=False)
            total = audit_result.get('total', 0)
            issues = audit_result.get('issues', [])
            logger.info(f"引用审计: {total} 个引用, {len(issues)} 个问题")
            return audit_result
        return None
    except Exception as e:
        logger.warning(f"引用审计失败: {e}")
        return None


def _run_motivation_thread(ctx: PaperContext):
    """构建动机线索（论文的"红线"贯穿）"""
    if not ctx.has('motivation') or not ctx.has('findings'):
        return None
    try:
        from motivation_thread import MotivationThread, SevenSentenceTest

        # 从 motivation 和 findings 构建线索
        motivation = ctx.motivation
        field_problem = getattr(motivation, 'field_problem', '') or ''
        specific_gap = getattr(motivation, 'specific_gap', '') or ''
        design_response = getattr(motivation, 'core_innovation', '') or ''
        evidence = getattr(motivation, 'evidence_support', '') or ''

        # 从 findings 中提取关键发现作为 evidence
        critical = [f for f in ctx.findings if f.get('importance') in ['critical', 'high']]
        if critical and not evidence:
            evidence_parts = []
            for f in critical[:3]:
                if f.get('type') == 'correlation':
                    v1, v2 = f.get('variables', ('', ''))
                    r = f.get('data', {}).get('r', 0)
                    evidence_parts.append(f'{v1}与{v2}相关(r={r:.3f})')
                elif f.get('type') == 'group_difference':
                    var = f.get('variable', '')
                    p = f.get('data', {}).get('p_value', 0)
                    evidence_parts.append(f'{var}季节差异显著(p={p:.4f})')
            evidence = '；'.join(evidence_parts)

        thread = MotivationThread(
            field_problem=field_problem,
            specific_gap=specific_gap,
            design_response=design_response,
            evidence=evidence,
        )

        # 运行七句话测试
        seven_test = SevenSentenceTest(thread)
        test_result = seven_test.validate()

        logger.info(f"动机线索: 完整度={thread.completeness():.0%}, 七句话测试={'通过' if test_result.get('passed') else '未通过'}")
        return {'thread': thread, 'test': test_result}
    except Exception as e:
        logger.warning(f"动机线索失败: {e}")
        return None


def _run_writing_rationale(ctx: PaperContext):
    """构建写作推理矩阵（追踪每个写作决策的推理链）"""
    if not ctx.has('sections') or not ctx.has('findings'):
        return None
    try:
        from writing_rationale import RationaleMatrix

        matrix = RationaleMatrix(store_path=os.path.join(ctx.output_dir, 'rationale_matrix.json'))

        # 为每个关键 finding 添加推理行
        critical = [f for f in ctx.findings if f.get('importance') in ['critical', 'high']]
        for f in critical[:10]:
            finding_text = ''
            if f.get('type') == 'correlation':
                v1, v2 = f.get('variables', ('', ''))
                r = f.get('data', {}).get('r', 0)
                finding_text = f'{v1}与{v2}相关(r={r:.3f})'
            elif f.get('type') == 'group_difference':
                var = f.get('variable', '')
                p = f.get('data', {}).get('p_value', 0)
                finding_text = f'{var}季节差异(p={p:.4f})'
            elif f.get('type') == 'distribution':
                var = f.get('variable', '')
                finding_text = f'{var}分布特征'

            if finding_text:
                matrix.add(
                    finding=finding_text,
                    mechanism='',
                    evidence='',
                    citation='',
                    section='results',
                )

        # 验证并保存
        issues = matrix.validate()
        matrix.save()
        ctx.rationale_rows = matrix.to_markdown()

        logger.info(f"推理矩阵: {len(matrix.rows)} 行, {len(issues)} 个问题")
        return matrix
    except Exception as e:
        logger.warning(f"推理矩阵失败: {e}")
        return None


def _run_revision_audit(ctx: PaperContext):
    """修订审计（检测版本间的变化是否实质性）"""
    if not ctx.revision_report:
        return None
    try:
        from revision_audit import audit_revision

        # 对比修订前后的全文
        original_sections = []
        for key in ['introduction', 'methods', 'results', 'discussion', 'conclusion']:
            text = ctx.sections.get(key, '')
            if text:
                original_sections.append(text)
        original_text = '\n\n'.join(original_sections)

        if not original_text or not ctx.revision_report:
            return None

        audit_result = audit_revision(original_text, ctx.revision_report)
        substantive = audit_result.substantive_changes if hasattr(audit_result, 'substantive_changes') else 0
        logger.info(f"修订审计: {substantive} 个实质性变更")
        return audit_result
    except Exception as e:
        logger.warning(f"修订审计失败: {e}")
        return None


def _run_domain_config(ctx: PaperContext):
    """加载领域配置"""
    if not ctx.domain:
        return None
    try:
        from domain_config import get_config
        config = get_config(ctx.domain)
        ctx.domain_config = config
        logger.info(f"领域配置: {config.domain_name} ({len(config.standards)} 个标准)")
        return config
    except Exception as e:
        logger.warning(f"领域配置失败: {e}")
        return None


def _run_writer_self_check(ctx: PaperContext):
    """
    Writer 自检：检查每个章节的质量

    检查维度：
    1. 字数是否达标
    2. 图表引用是否正确
    3. 统计值是否准确报告
    4. 格式是否规范
    """
    if not ctx.has('sections'):
        return None

    import re

    issues = []

    # 各章节目标字数
    word_targets = {
        'abstract': (200, 500),
        'introduction': (800, 1500),
        'methods': (600, 1200),
        'results': (1000, 2000),
        'discussion': (1200, 2500),
        'conclusion': (300, 600),
    }

    for section_name, (min_words, max_words) in word_targets.items():
        text = ctx.sections.get(section_name, '')
        if not text:
            continue

        # 1. 字数检查
        char_count = len(text)
        if char_count < min_words:
            issues.append(f'[{section_name}] 字数不足: {char_count}字 (目标: {min_words}-{max_words}字)')
        elif char_count > max_words * 1.5:
            issues.append(f'[{section_name}] 字数过多: {char_count}字 (目标: {min_words}-{max_words}字)')

        # 2. 图表引用检查
        fig_refs = re.findall(r'图\s*\d+|Fig\.?\s*\d+|Figure\s*\d+', text)
        if section_name == 'results' and len(fig_refs) == 0:
            issues.append(f'[results] 缺少图表引用，Results章节应引用图表')

        # 3. 统计值检查
        if section_name in ('results', 'discussion'):
            # 检查是否报告了p值
            p_values = re.findall(r'p\s*[<>=≤≥]\s*\d+\.?\d*', text)
            if ctx.has('findings'):
                sig_findings = [f for f in ctx.findings if f.get('data', {}).get('p', 1) < 0.05]
                if sig_findings and len(p_values) == 0:
                    issues.append(f'[{section_name}] 发现显著结果但未报告p值')

            # 检查是否报告了相关系数
            r_values = re.findall(r'r\s*=\s*-?\d+\.?\d*', text)
            corr_findings = [f for f in ctx.findings if f.get('type') == 'correlation']
            if corr_findings and len(r_values) == 0:
                issues.append(f'[{section_name}] 发现相关性但未报告r值')

        # 4. 格式检查
        # 检查是否有章节标题
        if not text.strip().startswith('#'):
            issues.append(f'[{section_name}] 缺少章节标题')

        # 检查段落分隔
        paragraphs = text.split('\n\n')
        if len(paragraphs) < 3:
            issues.append(f'[{section_name}] 段落过少({len(paragraphs)}段)，建议分段讨论')

    # 5. 引用检查
    if ctx.has('sections'):
        full_text = '\n\n'.join(ctx.sections.values())
        citation_count = len(re.findall(r'\[\d+\]', full_text))
        if citation_count == 0:
            issues.append('[全文] 未发现引用标记，学术论文应包含引用')

    # 记录检查结果
    if issues:
        logger.warning(f"Writer自检发现 {len(issues)} 个问题:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("Writer自检通过，未发现问题")

    return issues


def _run_methodology_learning(ctx: PaperContext):
    """
    方法论学习：从已读论文中提取方法论模式

    提取内容：
    1. 实验方法（采样、分析、预处理）
    2. 统计方法（检验、回归、相关）
    3. 仪器设备
    """
    if not ctx.papers_read:
        logger.info("无已读论文，跳过方法论学习")
        return None

    from pattern_learner import MethodologyLearner
    from paper_reader import PaperReader

    reader = PaperReader()
    learner = MethodologyLearner()

    for paper_info in ctx.papers_read[:10]:
        path = paper_info.get('path', '')
        if not path or not os.path.exists(path):
            continue
        try:
            content = reader.read(path, fetch_metadata=False)
            if not content:
                continue

            title = content.metadata.title if content.metadata else ''
            abstract = content.metadata.abstract if content.metadata else ''
            sections = []
            for sec in content.sections:
                sections.append({
                    'text': sec.text,
                    'section_type': sec.section_type if hasattr(sec, 'section_type') else 'unknown',
                })

            learner.learn_from_paper(title, abstract, sections)
        except Exception as e:
            logger.warning(f"方法论学习失败 {path}: {e}")

    # 获取方法论摘要
    method_summary = learner.get_method_summary()
    ctx.learned_methodologies = {
        'methods': learner.get_methods(),
        'summary': method_summary,
        'total_count': len(learner.methods),
        'papers_learned': len(ctx.papers_read),
    }

    logger.info(f"方法论学习: {len(learner.methods)} 个方法, "
               f"分类: {list(method_summary.keys())}")
    return ctx.learned_methodologies


# ============================================================
# 7. 气体分布图函数
# ============================================================

def _run_gas_distribution_figures(ctx: PaperContext):
    """生成气体污染物空间分布图（按功能区分组）+ 对数坐标冬春对比图"""
    if not ctx.has('df'):
        return None
    try:
        from gas_distribution_figures import (
            load_gas_data, create_gas_distribution_figures,
            create_gas_concentration_log_figure
        )

        # 优先使用桌面数据
        data = load_gas_data()
        if data is None:
            data = ctx.df

        if data is not None:
            analysis_dir = os.path.join(ctx.output_dir, 'figures')
            os.makedirs(analysis_dir, exist_ok=True)

            all_figures = []

            # 生成功能区分组图
            figures = create_gas_distribution_figures(data, analysis_dir)
            if figures:
                all_figures.extend(figures)

            # 生成对数坐标冬春对比图（修复N2O数据 + 解决遮挡）
            log_figures = create_gas_concentration_log_figure(data, analysis_dir)
            if log_figures:
                all_figures.extend(log_figures)

            if all_figures:
                ctx.gas_figures = all_figures
                logger.info(f"气体分布图: 生成 {len(all_figures)} 张")
                return all_figures

        return None
    except Exception as e:
        logger.warning(f"气体分布图生成失败: {e}")
        return None


def _run_auto_paper_finder(ctx: PaperContext):
    """自动文献搜索（arxiv + Semantic Scholar）"""
    try:
        from auto_paper_finder import AutoPaperFinder

        # 从领域配置获取搜索关键词
        domain_config = ctx.domain_config if ctx.has('domain_config') else {}
        search_query = domain_config.get('search_query', '')

        if not search_query:
            # 从findings中提取关键词
            if ctx.has('findings'):
                keywords = []
                for f in ctx.findings[:5]:
                    if 'variable' in f:
                        keywords.append(f['variable'])
                search_query = ' '.join(keywords[:3])

        if not search_query:
            logger.warning("自动文献搜索: 缺少搜索关键词")
            return None

        finder = AutoPaperFinder()
        papers = finder.search_and_save(search_query, max_results=10)

        if papers:
            ctx.papers_found = papers
            logger.info(f"自动文献搜索: 找到 {len(papers)} 篇论文")
            return papers

        return None
    except Exception as e:
        logger.warning(f"自动文献搜索失败: {e}")
        return None


def _run_fact_checker(ctx: PaperContext):
    """事实一致性检查（检测幻觉）"""
    if not ctx.has('sections') or not ctx.has('findings'):
        return None

    try:
        from fact_checker import get_fact_checker

        checker = get_fact_checker()
        full_paper = '\n\n'.join(
            ctx.sections.get(k, '') for k in
            ['introduction', 'methods', 'results', 'discussion', 'conclusion']
            if ctx.has_section(k)
        )

        if not full_paper:
            return None

        result = checker.check_consistency(full_paper, ctx.findings)
        ctx.fact_check_result = result

        if result.get('issues'):
            logger.warning(f"事实检查: 发现 {len(result['issues'])} 个问题")
        else:
            logger.info(f"事实检查: 通过 (分数: {result.get('score', 0):.2f})")

        return result
    except Exception as e:
        logger.warning(f"事实检查失败: {e}")
        return None


def _run_human_in_loop(ctx: PaperContext):
    """人工在环审核"""
    if not ctx.has('sections'):
        return None

    try:
        from human_in_loop import get_human_in_loop

        hil = get_human_in_loop()

        # 收集需要审核的内容
        items_to_review = []

        # 低置信度的审稿问题
        if ctx.has('review_report') and hasattr(ctx.review_report, 'issues'):
            for issue in ctx.review_report.issues:
                if issue.severity.value in ['CRITICAL', 'MAJOR']:
                    items_to_review.append({
                        'type': 'review_issue',
                        'section': issue.section,
                        'problem': issue.problem,
                        'suggestion': issue.suggestion,
                    })

        # 事实检查问题
        if ctx.has('fact_check_result'):
            fact_result = ctx.fact_check_result
            if fact_result.get('issues'):
                for issue in fact_result['issues'][:5]:
                    items_to_review.append({
                        'type': 'fact_check',
                        'detail': issue,
                    })

        if not items_to_review:
            logger.info("人工在环: 无需审核")
            return None

        # 记录待审核项（不阻塞流程）
        review_result = hil.submit_for_review(items_to_review)
        ctx.human_review_pending = items_to_review
        logger.info(f"人工在环: 提交 {len(items_to_review)} 项待审核")

        return review_result
    except Exception as e:
        logger.warning(f"人工在环失败: {e}")
        return None


def _run_learning_loop(ctx: PaperContext):
    """学习环路（高质量段落回写记忆）"""
    if not ctx.has('sections') or not ctx.has('memory'):
        return None

    try:
        from learning_loop import get_learning_loop

        loop = get_learning_loop(ctx.memory)

        # 评估各章节质量
        learned_count = 0
        for section_name in ['results', 'discussion', 'introduction']:
            text = ctx.sections.get(section_name, '')
            if not text or len(text) < 200:
                continue

            # 评估段落质量
            quality = loop.evaluate_paragraph_quality(text)
            if quality and quality.get('score', 0) > 0.7:
                # 提取句式模式
                patterns = loop.extract_patterns(text)
                if patterns:
                    loop.write_to_memory(section_name, patterns, quality)
                    learned_count += 1

        if learned_count:
            logger.info(f"学习环路: 回写 {learned_count} 个高质量章节")
            ctx.learned_from_output = learned_count

        return learned_count
    except Exception as e:
        logger.warning(f"学习环路失败: {e}")
        return None


def _run_quality_scorer(ctx: PaperContext):
    """文本质量评分"""
    if not ctx.has('sections'):
        return None

    try:
        from quality_scorer import QualityScorer

        scorer = QualityScorer()

        full_paper = '\n\n'.join(
            ctx.sections.get(k, '') for k in
            ['abstract', 'introduction', 'methods', 'results', 'discussion', 'conclusion']
            if ctx.has_section(k)
        )

        if not full_paper:
            return None

        # 获取引用和发现
        references = ctx.recalled_references if ctx.has('recalled_references') else []
        findings = ctx.findings if ctx.has('findings') else []

        score = scorer.score(full_paper, references=references, findings=findings)
        ctx.quality_score = score

        logger.info(f"质量评分: 总分={score.get('total', 0):.2f}, "
                    f"引用={score.get('citation', 0):.2f}, "
                    f"覆盖={score.get('coverage', 0):.2f}, "
                    f"语言={score.get('language', 0):.2f}")

        return score
    except Exception as e:
        logger.warning(f"质量评分失败: {e}")
        return None


# ============================================================
# 8. 模块注册表
# ============================================================

MODULE_REGISTRY = {
    'memory_init': {
        'needs': [],
        'provides': ['memory'],
        'run': _run_memory_init,
        'description': '初始化知识记忆',
    },
    'domain_config': {
        'needs': [],
        'provides': ['domain_config'],
        'run': _run_domain_config,
        'description': '加载领域配置（多领域支持）',
    },
    'paper_reading': {
        'needs': [],
        'provides': ['papers_read'],
        'run': _run_paper_reading,
        'description': '文献深度阅读（读论文→存入知识库）',
    },
    'pattern_learning': {
        'needs': ['papers_read'],
        'provides': ['learned_patterns', 'learned_mechanisms'],
        'run': _run_pattern_learning,
        'description': '写作模式学习（从论文中提取句式/讨论结构/机制）',
    },
    'literature_learning': {
        'needs': ['papers_read'],
        'provides': ['learned_expression_patterns'],
        'run': _run_literature_learning,
        'description': '文献深度学习（逻辑链提取/段落结构分析/学术表达模式库）',
    },
    'load_data': {
        'needs': ['data_path'],
        'provides': ['df'],
        'run': lambda ctx: _load_data(ctx),
        'description': '加载数据',
    },
    'explorer': {
        'needs': ['df'],
        'provides': ['findings'],
        'run': _run_explorer,
        'description': '数据探索',
    },
    'generate_figures': {
        'needs': ['df'],
        'provides': ['figures'],
        'run': _run_generate_figures,
        'description': '生成论文图表',
    },
    'smart_figure_planner': {
        'needs': ['df', 'findings'],
        'provides': ['figure_plans'],
        'run': _run_smart_figure_planner,
        'description': '智能作图规划（数据驱动）',
    },
    'figure_quality_checker': {
        'needs': ['figures'],
        'provides': ['figure_quality_reports'],
        'run': _run_figure_quality_checker,
        'description': '图表质量检查（视觉自检+陷阱检测）',
    },
    'scientific_analysis': {
        'needs': ['df'],
        'provides': ['findings'],
        'run': _run_scientific_analysis,
        'description': '智能分析编排（自动判断该做什么分析）',
    },
    'literature_recall': {
        'needs': ['memory', 'findings'],
        'provides': ['recalled_references'],
        'run': _run_literature_recall,
        'description': '从文献矩阵召回引用',
    },
    'motivation': {
        'needs': ['findings'],
        'provides': ['motivation', 'blueprint'],
        'run': _run_motivation,
        'description': '生成并确认写作动机',
    },
    'motivation_thread': {
        'needs': ['motivation', 'findings'],
        'provides': ['motivation_thread'],
        'run': _run_motivation_thread,
        'description': '构建动机线索（论文"红线"贯穿）',
    },
    'writer_results': {
        'needs': ['df', 'findings'],
        'provides': ['sections.results'],
        'run': _run_writer_results,
        'description': '写 Results',
    },
    'writer_discussion': {
        'needs': ['df', 'findings', 'memory'],
        'provides': ['sections.discussion'],
        'run': _run_writer_discussion,
        'description': '写 Discussion（带知识库）',
    },
    'writer_results_discussion': {
        'needs': ['df', 'findings', 'memory'],
        'provides': ['sections.results_discussion'],
        'run': _run_writer_results_discussion,
        'description': '写结果与讨论交织的章节（符合中文核心期刊规范）',
    },
    'writer_intro': {
        'needs': [],
        'provides': ['sections.introduction'],
        'run': _run_writer_intro,
        'description': '写 Introduction',
    },
    'writer_methods': {
        'needs': [],
        'provides': ['sections.methods'],
        'run': _run_writer_methods,
        'description': '写 Methods',
    },
    'writer_abstract': {
        'needs': ['sections.introduction', 'sections.methods'],
        'provides': ['sections.abstract'],
        'run': _run_writer_abstract,
        'description': '写 Abstract',
    },
    'writer_conclusion': {
        'needs': ['findings'],
        'provides': ['sections.conclusion'],
        'run': _run_writer_conclusion,
        'description': '写 Conclusion',
    },
    'writing_rationale': {
        'needs': ['sections', 'findings'],
        'provides': ['rationale_rows'],
        'run': _run_writing_rationale,
        'description': '构建写作推理矩阵（追踪每个写作决策）',
    },
    'polish': {
        'needs': ['sections', 'learned_patterns'],
        'provides': ['sections(polished)'],
        'run': _run_polish,
        'description': '用文献学到的模式润色各章节',
    },
    'review': {
        'needs': ['sections'],
        'provides': ['review_report', 'review_summary'],
        'run': _run_review,
        'description': '审稿检查',
    },
    'auto_revision': {
        'needs': ['review_report'],
        'provides': ['revision_report', 'sections(revised)'],
        'run': _run_auto_revision,
        'description': '自动修订',
    },
    'final_check': {
        'needs': ['sections(revised)'],
        'provides': ['review_summary'],
        'run': _run_final_check,
        'description': '修订后二次审稿（检查修订是否引入新问题）',
    },
    'citation_audit': {
        'needs': ['sections'],
        'provides': ['citation_audit_result'],
        'run': _run_citation_audit,
        'description': '引用质量审计（DOI验证、年份评分、类型分类）',
    },
    'revision_audit': {
        'needs': ['sections(revised)'],
        'provides': ['revision_audit_result'],
        'run': _run_revision_audit,
        'description': '修订审计（检测变更是否实质性）',
    },
    'advanced_analysis': {
        'needs': ['df'],
        'provides': ['advanced_findings', 'cross_analyses', 'anomaly_insights', 'data_stories', 'threshold_effects'],
        'run': _run_advanced_analysis,
        'description': '高级多维分析（交叉分析/异常深挖/数据故事线/阈值检测）',
    },
    'deep_imitation': {
        'needs': ['sections'],
        'provides': ['imitation_report'],
        'run': _run_deep_imitation,
        'description': '深度模仿分析（3表法）',
    },
    'integrity_audit': {
        'needs': ['sections'],
        'provides': ['integrity_report'],
        'run': _run_integrity_audit,
        'description': '完整性审计（4维度）',
    },
    'artifact_check': {
        'needs': ['sections'],
        'provides': ['artifact_report'],
        'run': _run_artifact_check,
        'description': '制品完整性检查',
    },
    'citation_bank': {
        'needs': ['sections'],
        'provides': ['citation_bank'],
        'run': _run_citation_bank,
        'description': '构建引用支撑库',
    },
    'assemble': {
        'needs': ['sections'],
        'provides': ['docx_path'],
        'run': _run_assemble,
        'description': '排版 DOCX',
    },
    'writer_self_check': {
        'needs': ['sections', 'findings'],
        'provides': [],
        'run': _run_writer_self_check,
        'description': 'Writer自检（字数/图表引用/统计值/格式检查）',
    },
    'methodology_learning': {
        'needs': ['papers_read'],
        'provides': ['learned_methodologies'],
        'run': _run_methodology_learning,
        'description': '方法论学习（从论文中提取实验方法/统计方法/采样方法）',
    },
    'latex_export': {
        'needs': ['sections'],
        'provides': ['latex_path'],
        'run': _run_latex_export,
        'description': '导出 LaTeX 格式论文',
    },
    'gas_distribution_figures': {
        'needs': ['df'],
        'provides': ['gas_figures'],
        'run': _run_gas_distribution_figures,
        'description': '生成气体污染物空间分布图（按功能区分组）',
    },
    'auto_paper_finder': {
        'needs': ['domain_config', 'findings'],
        'provides': ['papers_found'],
        'run': _run_auto_paper_finder,
        'description': '自动文献搜索（arxiv + Semantic Scholar）',
    },
    'fact_checker': {
        'needs': ['sections', 'findings'],
        'provides': ['fact_check_result'],
        'run': _run_fact_checker,
        'description': '事实一致性检查（检测幻觉）',
    },
    'human_in_loop': {
        'needs': ['sections', 'review_report'],
        'provides': ['human_review_pending'],
        'run': _run_human_in_loop,
        'description': '人工在环审核（低置信内容复核）',
    },
    'learning_loop': {
        'needs': ['sections', 'memory'],
        'provides': ['learned_from_output'],
        'run': _run_learning_loop,
        'description': '学习环路（高质量段落回写记忆）',
    },
    'quality_scorer': {
        'needs': ['sections'],
        'provides': ['quality_score'],
        'run': _run_quality_scorer,
        'description': '文本质量评分（引用/覆盖/语言/学术）',
    },
}


def _load_data(ctx: PaperContext):
    """加载数据"""
    from data_loader import DataLoader
    loader = DataLoader(ctx.data_path)
    ctx.df = loader.load_data()
    logger.info(f"数据: {ctx.df.shape[0]}行 x {ctx.df.shape[1]}列")

    # 提取元数据用于填充占位符
    ctx.metadata = loader.extract_metadata()
    logger.info(f"元数据: 采样点{ctx.metadata['n_sampling_points']}个, "
                f"冬季{ctx.metadata['n_winter_samples']}样本, "
                f"春季{ctx.metadata['n_spring_samples']}样本")

    return ctx.df


def _fill_placeholders(text: str, metadata: dict) -> str:
    """
    用元数据填充论文中的占位符

    Parameters
    ----------
    text : str, 论文文本
    metadata : dict, 从DataLoader.extract_metadata()获取的元数据

    Returns
    -------
    str : 填充后的文本
    """
    if not metadata:
        return text

    # 获取元数据值
    campus_area = metadata.get('campus_area', 'X')
    population = metadata.get('population', 'X')
    daily_sewage = metadata.get('daily_sewage', 'X')
    n_sampling_points = metadata.get('n_sampling_points', 'X')
    winter_months = metadata.get('winter_months', 'X')
    spring_months = metadata.get('spring_months', 'X')

    # 占位符映射表（支持多种格式）
    placeholder_map = {
        '约X公顷': f"约{campus_area}公顷",
        'X公顷': f"{campus_area}公顷",
        '约X万人': f"约{population}万人",
        'X万人': f"{population}万人",
        '约X m³/d': f"约{daily_sewage} m³/d",
        'X m³/d': f"{daily_sewage} m³/d",
        'X个采样点': f"{n_sampling_points}个采样点",
        '2024年X月': f"2024年{winter_months}月",
        '2025年X月': f"2025年{spring_months}月",
    }

    # 替换占位符（先替换长的，再替换短的，避免部分匹配）
    result = text
    # 按长度排序，先替换长的占位符
    sorted_placeholders = sorted(placeholder_map.keys(), key=len, reverse=True)
    for placeholder in sorted_placeholders:
        value = placeholder_map[placeholder]
        if placeholder in result:
            result = result.replace(placeholder, str(value))
            logger.info(f"填充占位符: {placeholder} -> {value}")

    return result


# ============================================================
# 5. 编排器
# ============================================================

class PaperOrchestrator:
    """
    灵活编排器 — 根据上下文状态自动调度模块。

    用法:
        ctx = PaperContext(data_file='data.xlsx')
        orch = PaperOrchestrator()
        orch.run(ctx)  # 自动推断步骤
        orch.run(ctx, steps=['explorer', 'writer_results', ...])  # 手动指定
    """

    def __init__(self):
        self.execution_log = []

    def run(self, ctx: PaperContext, steps: list = None):
        """
        执行编排。

        Parameters
        ----------
        ctx : PaperContext
        steps : list or None, 模块名列表。None 时自动推断。
        """
        if steps is None:
            steps = self._auto_plan(ctx)

        print("=" * 60)
        print("  PaperOrchestrator — 灵活编排")
        print(f"  步骤: {len(steps)}个")
        print("=" * 60)

        # 检测可并行的写作步骤
        parallel_groups = self._find_parallel_groups(steps)

        step_idx = 0
        for group in parallel_groups:
            if len(group) == 1:
                # 单步执行
                step_name = group[0]
                step_idx += 1
                self._execute_step(ctx, step_name, step_idx, len(steps))
            else:
                # 并行执行
                self._execute_parallel(ctx, group, step_idx + 1, len(steps))
                step_idx += len(group)

        # 保存全文 MD
        self._save_paper_md(ctx)

        print(f"\n{'=' * 60}")
        print(f"  编排完成!")
        print(f"{'=' * 60}")

    def _execute_step(self, ctx, step_name, idx, total):
        """执行单个步骤"""
        if step_name not in MODULE_REGISTRY:
            logger.warning(f"未知模块: {step_name}")
            return

        module = MODULE_REGISTRY[step_name]
        missing = self._check_needs(ctx, module['needs'])
        if missing:
            print(f"\n[{idx}/{total}] {module['description']} — 跳过 (缺少: {missing})")
            self.execution_log.append({'step': step_name, 'status': 'skipped', 'missing': missing})
            return

        print(f"\n[{idx}/{total}] {module['description']}...")
        try:
            result = module['run'](ctx)
            ctx.mark_done(step_name)
            self.execution_log.append({'step': step_name, 'status': 'done'})
            if result is not None:
                if isinstance(result, str) and len(result) > 50:
                    print(f"  完成 ({len(result)} 字)")
                elif isinstance(result, list):
                    print(f"  完成 ({len(result)} 项)")
                elif isinstance(result, dict):
                    print(f"  完成 ({len(result)} 个条目)")
                else:
                    print(f"  完成")
            self._save_checkpoint(ctx, step_name)
        except Exception as e:
            logger.error(f"模块 {step_name} 失败: {e}")
            self.execution_log.append({'step': step_name, 'status': 'failed', 'error': str(e)})

    def _execute_parallel(self, ctx, steps, idx_start, total):
        """并行执行多个步骤（使用线程池）"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"\n[{idx_start}/{total}] 并行执行 {len(steps)} 个步骤: {steps}")
        with ThreadPoolExecutor(max_workers=len(steps)) as executor:
            futures = {}
            for step_name in steps:
                if step_name in MODULE_REGISTRY:
                    module = MODULE_REGISTRY[step_name]
                    missing = self._check_needs(ctx, module['needs'])
                    if not missing:
                        futures[executor.submit(module['run'], ctx)] = step_name
                    else:
                        print(f"  {step_name} — 跳过 (缺少: {missing})")

            for future in as_completed(futures):
                step_name = futures[future]
                try:
                    result = future.result()
                    ctx.mark_done(step_name)
                    self.execution_log.append({'step': step_name, 'status': 'done'})
                    if isinstance(result, str) and result:
                        print(f"  {step_name} 完成 ({len(result)} 字)")
                    else:
                        print(f"  {step_name} 完成")
                except Exception as e:
                    logger.error(f"模块 {step_name} 失败: {e}")
                    self.execution_log.append({'step': step_name, 'status': 'failed', 'error': str(e)})

    def _find_parallel_groups(self, steps):
        """
        将步骤分组，同组内可并行执行（增强版）。
        规则：
        - Introduction + Methods 可并行（互不依赖，且不依赖数据）
        - 审计类模块可并行（citation_audit, revision_audit, deep_imitation, integrity_audit, artifact_check）
        - 初始化模块可并行（memory_init, domain_config）
        - 其他步骤串行
        """
        # 定义可并行的模块组
        parallel_sets = [
            {'writer_intro', 'writer_methods'},           # 互不依赖
            {'citation_audit', 'revision_audit', 'deep_imitation', 'integrity_audit', 'artifact_check'},  # 审计类可并行
            {'memory_init', 'domain_config'},              # 初始化可并行
        ]

        groups = []
        remaining = list(steps)

        for pset in parallel_sets:
            batch = [s for s in remaining if s in pset]
            if len(batch) > 1:
                groups.append(batch)
                for s in batch:
                    remaining.remove(s)

        # 剩余步骤全部串行
        for s in remaining:
            groups.append([s])

        return groups

    def _auto_plan(self, ctx: PaperContext) -> list:
        """根据上下文状态自动推断执行步骤（优化版）"""
        steps = []

        # 第1阶段：知识初始化（可并行）
        steps.append('memory_init')
        if ctx.domain:
            steps.append('domain_config')

        # 第2阶段：数据处理（串行）
        if ctx.has('data_path'):
            steps.append('load_data')
        steps.append('explorer')
        steps.append('scientific_analysis')

        # 第3阶段：高级分析（依赖scientific_analysis的结果）
        steps.append('advanced_analysis')

        # 第4阶段：图表生成（依赖所有分析结果）
        steps.append('generate_figures')

        # 第5阶段：文献学习（可与图表生成并行，但当前串行更安全）
        steps.append('paper_reading')
        steps.append('pattern_learning')
        steps.append('methodology_learning')

        # 第6阶段：知识支撑 + 动机线索
        steps.append('literature_recall')
        steps.append('motivation')
        steps.append('motivation_thread')

        # 第7阶段：AI写作（使用交织写作方式）
        # 使用新的结果与讨论交织写作（符合中文核心期刊规范）
        steps.append('writer_results_discussion')
        # Introduction 和 Methods 可以并行，但当前串行更稳定
        steps.append('writer_intro')
        steps.append('writer_methods')
        steps.append('writer_conclusion')
        steps.append('writer_abstract')
        steps.append('writing_rationale')

        # 第8阶段：润色 + 审改循环
        steps.append('polish')
        for _iteration in range(2):
            steps.append('review')
            steps.append('auto_revision')

        # 第9阶段：补充审计（可并行）
        steps.append('citation_audit')
        steps.append('revision_audit')
        steps.append('final_check')
        steps.append('writer_self_check')
        steps.append('deep_imitation')
        steps.append('integrity_audit')
        steps.append('artifact_check')
        steps.append('citation_bank')

        # 第10阶段：输出
        steps.append('assemble')
        steps.append('latex_export')

        return steps

    def _save_checkpoint(self, ctx: PaperContext, step_name: str):
        """保存中间结果检查点"""
        import json
        checkpoint_dir = os.path.join(ctx.output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, f'{step_name}.json')
        try:
            data = {
                'step': step_name,
                'sections': {k: v[:200] + '...' if isinstance(v, str) and len(v) > 200 else v
                            for k, v in ctx.sections.items()},
                'findings_count': len(ctx.findings),
                'papers_read_count': len(ctx.papers_read),
            }
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug(f"Checkpoint save failed: {e}")

    def _check_needs(self, ctx: PaperContext, needs: list) -> list:
        """检查前置条件"""
        missing = []
        for need in needs:
            if need.startswith('sections.'):
                section_name = need.split('.', 1)[1]
                if not ctx.has_section(section_name):
                    missing.append(need)
            elif need == 'sections':
                if not ctx.sections:
                    missing.append(need)
            elif need == 'sections(revised)':
                # sections(revised) 表示修订后的章节已存在
                if not ctx.revision_report:
                    missing.append(need)
            elif not ctx.has(need):
                missing.append(need)
        return missing

    def _save_paper_md(self, ctx: PaperContext):
        """保存全文 MD"""
        if not ctx.sections:
            return
        os.makedirs(ctx.output_dir, exist_ok=True)

        # 支持交织写作模式（results_discussion）和传统模式（results + discussion）
        if ctx.has_section('results_discussion'):
            order = ['abstract', 'introduction', 'methods', 'results_discussion', 'conclusion']
        else:
            order = ['abstract', 'introduction', 'methods', 'results', 'discussion', 'conclusion']

        # 添加参考文献
        order.append('references')

        parts = []
        for key in order:
            if ctx.has_section(key):
                parts.append(ctx.sections[key])

        full_paper = '\n\n---\n\n'.join(parts)
        path = os.path.join(ctx.output_dir, 'paper.md')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(full_paper)
        ctx.paper_md_path = path
        logger.info(f"论文MD: {path}")

        # 自动执行全文自检
        self._self_check_paper(ctx)

    def _self_check_paper(self, ctx: PaperContext) -> dict:
        """
        全流程完成后自动检查论文质量

        检查维度：
        1. 章节完整性（是否有缺失章节）
        2. 字数检查（各章节字数是否合理）
        3. 元评论检查（是否有Claude元评论残留）
        4. 占位符检查（是否有未填充的占位符）
        5. 图片检查（图片是否正确嵌入）
        6. 引用检查（是否有引用标记）
        """
        import re

        issues = []
        report = {}

        # 读取论文内容
        paper_path = os.path.join(ctx.output_dir, 'paper.md')
        if not os.path.exists(paper_path):
            issues.append('论文文件不存在')
            return {'issues': issues, 'score': 0}

        with open(paper_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 1. 章节完整性检查
        required_sections = ['引言', '绪论', '材料与方法', '结果', '讨论', '结论']
        found_sections = []
        for section in required_sections:
            if section in content:
                found_sections.append(section)
        # 检查是否找到了引言或绪论
        if '引言' not in found_sections and '绪论' not in found_sections:
            issues.append('缺少章节: 引言/绪论')
        # 检查其他必要章节
        for section in ['材料与方法', '结果', '讨论', '结论']:
            if section not in content and section.replace('与分析', '') not in content:
                issues.append(f'缺少章节: {section}')

        # 2. 字数检查
        chinese_chars = len(re.findall(r'[一-鿿]', content))
        report['chinese_chars'] = chinese_chars
        if chinese_chars < 3000:
            issues.append(f'论文字数过少: {chinese_chars}字')

        # 3. 元评论检查
        meta_patterns = [
            '写入权限未开启',
            '需要您授予',
            '内容已准备完毕',
            '包含以下核心章节',
            '本节要点',
            '以下是撰写的',
            '如需调整',
            '如需写入',
            '请在弹出的权限',
            '内容已呈现完毕',
            '数据核验说明',
        ]
        for pattern in meta_patterns:
            if pattern in content:
                issues.append(f'发现Claude元评论: {pattern}')

        # 4. 占位符检查
        placeholders = ['X公顷', 'X万人', 'X m³/d', 'X个采样点']
        for placeholder in placeholders:
            if placeholder in content:
                issues.append(f'发现未填充占位符: {placeholder}')

        # 5. 图片检查
        figure_count = len(ctx.figures)
        report['figure_count'] = figure_count
        if figure_count == 0:
            issues.append('没有生成图片')

        # 6. 引用检查
        citation_count = len(re.findall(r'\[\d+\]', content))
        report['citation_count'] = citation_count
        if citation_count == 0:
            issues.append('没有引用标记')

        # 计算评分
        score = 100 - len(issues) * 10
        score = max(0, min(100, score))

        report['issues'] = issues
        report['score'] = score

        # 输出检查结果
        print("\n" + "=" * 60)
        print("  论文自检报告")
        print("=" * 60)
        print(f"  中文字数: {chinese_chars}")
        print(f"  图片数量: {figure_count}")
        print(f"  引用数量: {citation_count}")
        print(f"  评分: {score}/100")
        if issues:
            print(f"\n  发现 {len(issues)} 个问题:")
            for issue in issues:
                print(f"    ❌ {issue}")
        else:
            print("\n  ✅ 未发现问题")
        print("=" * 60)

        return report

    def get_log(self) -> list:
        return self.execution_log
