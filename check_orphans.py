# -*- coding: utf-8 -*-
"""
孤立模块检查器
==============
检查根目录下所有 .py 模块是否被其他模块导入（即是否接入管线）。
每次新增模块或合并分支后运行此脚本。

用法:
    python check_orphans.py

退出码:
    0 = 无孤立模块
    1 = 有孤立模块
"""

import os
import sys


def find_orphans(root_dir='.'):
    """找出所有未被任何其他 .py 文件导入的模块"""
    # 收集根目录所有 .py 文件（排除 test_ 开头和 check_orphans 自身）
    all_py = [
        f for f in os.listdir(root_dir)
        if f.endswith('.py')
        and not f.startswith('test_')
        and f != 'check_orphans.py'
        and f != 'conftest.py'
    ]

    # 收集所有文件的导入语句
    imports_in_files = {}
    for f in all_py:
        filepath = os.path.join(root_dir, f)
        try:
            with open(filepath, encoding='utf-8', errors='ignore') as fh:
                content = fh.read()
            imports_in_files[f] = content
        except Exception:
            imports_in_files[f] = ''

    # 检查每个模块是否被其他文件导入
    orphaned = []
    connected = []

    # 工具库/基础设施模块（不需要注册到MODULE_REGISTRY，但被其他模块内部调用）
    utility_modules = {
        'academic_plot_style',    # 期刊规范作图系统
        'assertion_control',      # 断言强度控制
        'audit_logger',           # 审计日志服务
        'chart_qa',               # 图表质量检测
        'citation_guard',         # 引用安全防护
        'claude_writer',          # Claude CLI写作引擎
        'cn_core_rules',          # 中文核心期刊规范
        'knowledge_memory',       # 知识记忆系统
        'proxy_config',           # 代理配置
        'review_rules',           # 审稿规则库
        'runtime_metrics',        # 运行时指标
        'self_evolving_engine',   # 自进化引擎核心
        'text_utils',             # 公共文本工具
        'variable_registry',      # 变量注册中心
        'writer_schema',          # 写作接口Schema
        'statistical_analysis',   # 统计分析工具库
        'generate_docx',          # DOCX生成脚本
        'web_app',                # Flask Web界面
        'ai_trace_enhanced',      # AI痕迹检测（被review_agent调用）
        'document_assembler',     # 文档组装器（被assemble步骤调用）
        'literature_memory',      # 文献记忆（被literature_recall步骤调用）
        'motivation_planner',     # 动机规划器（被motivation步骤调用）
        'paper_reader',           # 论文阅读器（被paper_reading步骤调用）
        'paper_writing_agent',    # 论文写作Agent（被writer_*步骤调用）
        'pattern_learner',        # 模式学习器（被pattern_learning步骤调用）
        'scientific_analysis_agent',  # 科研分析Agent（被scientific_analysis步骤调用）
        'scientific_visualization_agent',  # 可视化Agent（被generate_figures步骤调用）
        'citation_support_bank',  # 引用支撑库（被citation_bank步骤调用）
        'enhanced_pdf_parser',    # 增强PDF解析器（被paper_reader调用）
    }

    # 已在MODULE_REGISTRY中注册的模块（通过不同名称）
    registered_aliases = {
        'data_driven_pipeline': 'explorer',
        'data_loader': 'load_data',
        'latex_exporter': 'latex_export',
        'literature_memory': 'literature_recall',
        'motivation_planner': 'motivation',
        'paper_reader': 'paper_reading',
        'paper_writing_agent': 'writer_*',
        'pattern_learner': 'pattern_learning',
        'scientific_analysis_agent': 'scientific_analysis',
        'scientific_visualization_agent': 'generate_figures',
        'document_assembler': 'assemble',
        'citation_support_bank': 'citation_bank',
    }

    for f in sorted(all_py):
        module_name = f[:-3]  # 去掉 .py
        is_imported = False
        importers = []

        for other_f, content in imports_in_files.items():
            if other_f == f:
                continue
            # 检查 import xxx 或 from xxx import
            if (f'import {module_name}' in content or
                f'from {module_name}' in content):
                is_imported = True
                importers.append(other_f)

        if is_imported:
            connected.append((f, importers))
        else:
            # 入口脚本（不需要注册到MODULE_REGISTRY）
            if module_name in ('paper_context', 'run_pipeline'):
                connected.append((f, ['[entry point]']))
            # 工具库/基础设施模块
            elif module_name in utility_modules:
                connected.append((f, ['[utility/infrastructure]']))
            # 已注册的模块（通过不同名称）
            elif module_name in registered_aliases:
                alias = registered_aliases[module_name]
                connected.append((f, [f'[registered as: {alias}]']))
            else:
                orphaned.append(f)

    return orphaned, connected


def main():
    print("=" * 60)
    print("  孤立模块检查器")
    print("=" * 60)
    print()

    orphaned, connected = find_orphans()

    print(f"[已连通] {len(connected)} 个模块:")
    for f, importers in connected:
        print(f"  OK  {f:35s} <- {', '.join(importers)}")

    print()

    if orphaned:
        print(f"[孤立] {len(orphaned)} 个模块未接入管线:")
        for f in orphaned:
            print(f"  XX  {f}")
        print()
        print("请在 paper_context.py 中注册这些模块。")
        print("参考 CLAUDE.md 第5条：模块注册规则。")
        sys.exit(1)
    else:
        print("[OK] 0 个孤立模块，全部已连通。")
        sys.exit(0)


if __name__ == '__main__':
    main()
