#!/bin/bash

# 技能体与教程整合脚本

echo "开始整合技能体与教程..."

# 定义映射关系（教程文件夹:技能体文件夹）
declare -A MAPPINGS=(
    ["第1课-层次分析法"]="ahp-method"
    ["第2课-Topsis"]="topsis-scorer"
    ["第3课-熵权法"]="entropy-weight-method"
    ["第6课-主成分分析"]="pca-analyzer"
    ["第8课-蒙特卡罗法"]="monte-carlo-engine"
    ["第12课 多目标规划模型及代码"]="multi-objective-optimization"
    ["第15课-最短路径"]="shortest-path"
    ["第21课-灰色预测GM(1,1)模型"]="grey-forecaster"
    ["第22课-时间序列AIRMA模型"]="arima-forecaster"
)

# 整合已有技能体
for lesson in "${!MAPPINGS[@]}"; do
    skill="${MAPPINGS[$lesson]}"
    echo "处理: $lesson -> $skill"
    
    # 创建目录
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
    
    echo "  完成: $skill"
    echo ""
done

echo "已有技能体整合完成！"
echo ""
echo "需要创建的新技能体："
echo "  - linear-programming (第7课)"
echo "  - nonlinear-programming (第9课)"
echo "  - integer-programming (第10课)"
echo "  - minimax-programming (第11课)"
echo "  - dynamic-programming (第13课)"
echo "  - graph-theory (第14课)"
echo "  - minimum-spanning-tree (第16课)"
echo "  - fuzzy-evaluation (第4课)"
echo "  - grey-relation (第5课)"
