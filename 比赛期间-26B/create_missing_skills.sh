#!/bin/bash

# 为缺失的教程创建技能体

echo "创建缺失的技能体..."

# 定义需要创建的技能体（教程:技能体名称）
declare -A NEW_SKILLS=(
    ["第4课-模糊综合评价"]="fuzzy-evaluation"
    ["第5课-灰色关联分析"]="grey-relation"
    ["第9课-非线性规划"]="nonlinear-programming"
    ["第10课-整数规划和0-1规划模型"]="integer-programming"
    ["第11课-最大最小化规划模型"]="minimax-programming"
    ["第13课-动态规划"]="dynamic-programming"
    ["第14课-图论"]="graph-theory"
    ["第16课-最小生成树"]="minimum-spanning-tree"
)

for lesson in "${!NEW_SKILLS[@]}"; do
    skill="${NEW_SKILLS[$lesson]}"
    echo "创建技能体: $skill"
    
    # 创建目录结构
    mkdir -p ".github/skills/$skill/references"
    mkdir -p ".github/skills/$skill/examples"
    
    # 复制PDF教程
    if ls "Models-instruction/$lesson"/*.pdf 1> /dev/null 2>&1; then
        cp "Models-instruction/$lesson"/*.pdf ".github/skills/$skill/references/"
        echo "  ✓ 复制PDF教程"
    fi
    
    # 复制Python代码
    if ls "Models-instruction/$lesson"/*.py 1> /dev/null 2>&1; then
        cp "Models-instruction/$lesson"/*.py ".github/skills/$skill/examples/"
        echo "  ✓ 复制Python代码"
    fi
    
    # 复制MATLAB代码
    if ls "Models-instruction/$lesson"/*.m 1> /dev/null 2>&1; then
        cp "Models-instruction/$lesson"/*.m ".github/skills/$skill/examples/"
        echo "  ✓ 复制MATLAB代码"
    fi
    
    # 复制Jupyter Notebook
    if ls "Models-instruction/$lesson"/*.ipynb 1> /dev/null 2>&1; then
        cp "Models-instruction/$lesson"/*.ipynb ".github/skills/$skill/examples/"
        echo "  ✓ 复制Jupyter Notebook"
    fi
    
    # 复制MATLAB Live Script
    if ls "Models-instruction/$lesson"/*.mlx 1> /dev/null 2>&1; then
        cp "Models-instruction/$lesson"/*.mlx ".github/skills/$skill/examples/"
        echo "  ✓ 复制MATLAB Live Script"
    fi
    
    # 创建基础SKILL.md（如果不存在）
    if [ ! -f ".github/skills/$skill/SKILL.md" ]; then
        cat > ".github/skills/$skill/SKILL.md" << 'EOFSKILL'
---
name: SKILL_NAME
description: SKILL_DESCRIPTION
---

# SKILL_TITLE

## Quick Start

[待补充]

## Examples

See `examples/` folder for code implementations.

## References

- `references/`: Complete tutorial PDFs with theory and examples

## Competition Tips

[待补充]

## Time Budget

- Implementation: 30-60 min
- Validation: 30-60 min

EOFSKILL
        # 替换占位符
        sed -i "s/SKILL_NAME/$skill/g" ".github/skills/$skill/SKILL.md"
        sed -i "s/SKILL_DESCRIPTION/MCM\/ICM skill for $lesson/g" ".github/skills/$skill/SKILL.md"
        sed -i "s/SKILL_TITLE/${skill^}/g" ".github/skills/$skill/SKILL.md"
        echo "  ✓ 创建基础SKILL.md"
    fi
    
    echo "  完成: $skill"
    echo ""
done

echo "所有缺失技能体已创建！"
echo ""
echo "注意：SKILL.md文件已创建基础框架，需要后续补充详细内容。"
