import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

class EnvironmentalAnalysis:
    def __init__(self):
        # 基础参数
        self.total_material = 100e6  # 总材料需求（吨）
        
        # 太空电梯参数
        self.space_elevator_capacity_per_year = 179000  # 每年运输能力（吨）
        self.num_galactic_harbors = 3  # 银河港口数量
        self.space_elevator_cost_per_ton = 500  # 每吨运输成本（美元）
        self.space_elevator_emissions_per_ton = 0.1  # 太空电梯每吨碳排放（吨）
        self.space_elevator_renewable_factor = 0.5  # 可再生能源使用比例
        
        # 传统火箭参数
        self.rocket_payload_range = [100, 150]  # 火箭有效载荷范围（吨）
        self.rocket_cost_per_ton = 10000  # 每吨运输成本（美元）
        self.rocket_emissions_per_ton = 10  # 火箭每吨碳排放（吨）
        self.num_rocket_bases = 10  # 火箭发射基地数量
        
        # 结果保存路径
        self.results_path = '/Users/rzn/Desktop/2026_MCM-ICM_Problems/B题/Problem4/results'
    
    def calculate_environmental_impact(self, space_fraction, renewable_factor=1.0):
        """计算环境影响（碳排放）"""
        material_space = self.total_material * space_fraction
        material_rocket = self.total_material * (1 - space_fraction)
        
        # 计算碳排放
        emissions_space = material_space * self.space_elevator_emissions_per_ton * renewable_factor
        emissions_rocket = material_rocket * self.rocket_emissions_per_ton
        total_emissions = emissions_space + emissions_rocket
        
        # 计算成本
        cost = material_space * self.space_elevator_cost_per_ton + material_rocket * self.rocket_cost_per_ton
        
        return total_emissions, cost
    
    def generate_results(self):
        """生成环境影响分析结果"""
        # 太空电梯比例范围
        space_fractions = np.linspace(0, 1, 11)
        env_data = []
        
        # 不同可再生能源使用比例
        renewable_factors = [0.5, 0.75, 1.0]
        renewable_data = {}
        
        for factor in renewable_factors:
            factor_data = []
            for fraction in space_fractions:
                emissions, cost = self.calculate_environmental_impact(fraction, factor)
                factor_data.append({
                    '太空电梯比例': fraction,
                    '碳排放（万吨）': emissions/1e4,
                    '成本（亿美元）': cost/1e8
                })
            renewable_data[factor] = factor_data
        
        # 基础情况（无可再生能源优化）
        base_data = []
        for fraction in space_fractions:
            emissions, cost = self.calculate_environmental_impact(fraction, 1.0)
            base_data.append({
                '太空电梯比例': fraction,
                '碳排放（万吨）': emissions/1e4,
                '成本（亿美元）': cost/1e8
            })
        
        return base_data, renewable_data
    
    def save_results(self, base_data, renewable_data):
        """保存结果到文件"""
        # 保存基础情况为CSV
        base_df = pd.DataFrame(base_data)
        base_df.to_csv(f'{self.results_path}/environmental_impact_base.csv', index=False)
        
        # 保存可再生能源优化情况
        for factor, data in renewable_data.items():
            df = pd.DataFrame(data)
            df.to_csv(f'{self.results_path}/environmental_impact_renewable_{factor}.csv', index=False)
        
        # 生成图表
        self.generate_charts(base_data, renewable_data)
    
    def generate_charts(self, base_data, renewable_data):
        """生成环境影响分析图表"""
        # 基础情况数据
        fractions = [item['太空电梯比例'] for item in base_data]
        emissions = [item['碳排放（万吨）'] for item in base_data]
        costs = [item['成本（亿美元）'] for item in base_data]
        
        # 双Y轴图：碳排放和成本
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        color = 'tab:red'
        ax1.set_xlabel('太空电梯比例', fontsize=12)
        ax1.set_ylabel('碳排放（万吨）', color=color, fontsize=12)
        ax1.plot(fractions, emissions, color=color, linewidth=3, marker='o', markersize=8, label='碳排放')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('成本（亿美元）', color=color, fontsize=12)
        ax2.plot(fractions, costs, color=color, linewidth=3, marker='s', markersize=8, label='成本')
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()
        plt.title('太空电梯比例对环境影响和成本的影响', fontsize=16)
        fig.legend(loc='upper left', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f'{self.results_path}/emissions_cost_relation.png', dpi=300)
        plt.close()
        
        # 可再生能源影响
        plt.figure(figsize=(12, 8))
        for factor, data in renewable_data.items():
            emissions = [item['碳排放（万吨）'] for item in data]
            plt.plot(fractions, emissions, linewidth=2, marker='o', markersize=6, label=f'可再生能源比例: {factor}')
        
        plt.title('可再生能源使用对碳排放的影响', fontsize=16)
        plt.xlabel('太空电梯比例', fontsize=12)
        plt.ylabel('碳排放（万吨）', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/renewable_energy_impact.png', dpi=300)
        plt.close()
        
        # 环境影响减少百分比
        base_emissions = base_data[0]['碳排放（万吨）']  # 纯火箭的碳排放
        reduction_percentages = [(base_emissions - item['碳排放（万吨）']) / base_emissions * 100 for item in base_data]
        
        plt.figure(figsize=(12, 8))
        plt.plot(fractions, reduction_percentages, linewidth=3, color='green', marker='^', markersize=8)
        plt.title('环境影响减少百分比', fontsize=16)
        plt.xlabel('太空电梯比例', fontsize=12)
        plt.ylabel('碳排放减少百分比（%）', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/emissions_reduction.png', dpi=300)
        plt.close()
        
        # 环境友好型运输方案示意图
        self.generate_environmental_solution_diagram()
    
    def generate_environmental_solution_diagram(self):
        """生成环境友好型运输方案示意图"""
        plt.figure(figsize=(12, 10))
        
        # 绘制环境友好型运输方案
        plt.title('环境友好型月球殖民地运输方案', fontsize=16)
        
        # 左侧：太空电梯系统
        plt.subplot(1, 2, 1)
        plt.title('太空电梯系统（环保）', fontsize=14)
        
        # 地球
        plt.plot([0.5], [0.1], 'o', markersize=30, color='blue', alpha=0.7)
        plt.text(0.5, 0.1, '地球', ha='center', va='center', color='white', fontsize=10)
        
        # 系绳
        plt.plot([0.5, 0.5], [0.1, 0.9], 'k-', linewidth=2)
        
        # 太阳能板
        plt.plot([0.3, 0.7], [0.7, 0.7], 's', markersize=20, color='yellow', alpha=0.7)
        plt.text(0.5, 0.7, '太阳能', ha='center', va='center', color='black', fontsize=8)
        
        # 顶点锚
        plt.plot([0.5], [0.9], 'o', markersize=15, color='red')
        plt.text(0.5, 0.9, '顶点锚', ha='center', va='center', color='white', fontsize=8)
        
        # 绿色箭头表示低排放
        plt.arrow(0.5, 0.3, 0, 0.2, width=0.02, head_width=0.05, color='green')
        plt.text(0.6, 0.4, '低排放', ha='left', va='center', color='green', fontsize=10)
        
        plt.axis('off')
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        
        # 右侧：火箭系统
        plt.subplot(1, 2, 2)
        plt.title('火箭系统（高排放）', fontsize=14)
        
        # 发射台
        plt.plot([0.3], [0.1], 's', markersize=20, color='gray')
        plt.text(0.3, 0.1, '发射台', ha='center', va='center', color='white', fontsize=10)
        
        # 火箭轨迹
        trajectory = np.linspace(0, 1, 100)
        height = 0.1 + trajectory ** 2 * 0.8
        plt.plot(0.3 + trajectory * 0.4, height, 'k-', linewidth=2)
        
        # 红色箭头表示高排放
        plt.arrow(0.5, 0.3, 0, 0.2, width=0.02, head_width=0.05, color='red')
        plt.text(0.6, 0.4, '高排放', ha='left', va='center', color='red', fontsize=10)
        
        # 烟雾效果
        for i in range(5):
            plt.plot(0.3 + i*0.05, 0.15 + i*0.05, 'o', markersize=5+i, color='gray', alpha=0.5)
        
        plt.axis('off')
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/environmental_solution_diagram.png', dpi=300)
        plt.close()

# 主函数
if __name__ == "__main__":
    analysis = EnvironmentalAnalysis()
    base_data, renewable_data = analysis.generate_results()
    analysis.save_results(base_data, renewable_data)
    
    # 打印结果
    print("问题4：环境影响分析")
    print("=" * 60)
    print("不同太空电梯比例的环境影响：")
    print("太空电梯比例 | 碳排放（万吨） | 成本（亿美元）")
    print("-" * 60)
    for item in base_data:
        print(f"{item['太空电梯比例']:.2f} | {item['碳排放（万吨）']:.2f} | {item['成本（亿美元）']:.2f}")
    
    print("\n环境影响最小化建议：")
    print("1. 优先使用太空电梯系统，减少火箭使用")
    print("2. 为太空电梯系统提供可再生能源供电")
    print("3. 建立月球水资源循环利用系统")
    print("4. 优化火箭燃料，减少排放")
    
    print("\n结果已保存到 Problem4/results 文件夹")
