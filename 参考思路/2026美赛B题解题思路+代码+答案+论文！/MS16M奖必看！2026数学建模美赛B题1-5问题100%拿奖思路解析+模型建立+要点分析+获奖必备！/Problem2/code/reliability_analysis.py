import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

class ReliabilityAnalysis:
    def __init__(self):
        # 基础参数
        self.total_material = 100e6  # 总材料需求（吨）
        
        # 太空电梯参数
        self.space_elevator_capacity_per_year = 179000  # 每年运输能力（吨）
        self.num_galactic_harbors = 3  # 银河港口数量
        self.space_elevator_reliability = 0.98  # 可靠性
        
        # 传统火箭参数
        self.rocket_payload_range = [100, 150]  # 火箭有效载荷范围（吨）
        self.rocket_reliability = 0.95  # 可靠性
        self.num_rocket_bases = 10  # 火箭发射基地数量
        
        # 结果保存路径
        self.results_path = '/Users/rzn/Desktop/2026_MCM-ICM_Problems/B题/Problem2/results'
    
    def calculate_reliability_impact(self, reliability_factor):
        """计算可靠性对运输的影响（添加波动）"""
        # 调整后的太空电梯容量 - 添加波动
        base_space_capacity = self.space_elevator_capacity_per_year * reliability_factor * self.space_elevator_reliability
        # 添加±5%的随机波动
        space_variation = base_space_capacity * (0.95 + 0.1 * np.random.random())
        total_space_capacity = space_variation * self.num_galactic_harbors
        
        # 调整后的火箭容量 - 添加波动
        avg_payload = np.mean(self.rocket_payload_range)
        # 添加±8%的随机波动
        payload_variation = avg_payload * (0.92 + 0.16 * np.random.random())
        launches_per_year_per_base = 50  # 每个基地每年发射次数
        # 添加±10%的随机波动
        launches_variation = launches_per_year_per_base * (0.9 + 0.2 * np.random.random())
        base_rocket_capacity = payload_variation * launches_variation * self.num_rocket_bases * reliability_factor * self.rocket_reliability
        # 添加±12%的随机波动
        rocket_variation = base_rocket_capacity * (0.88 + 0.24 * np.random.random())
        
        # 计算运输时间
        space_time = self.total_material / total_space_capacity
        rocket_time = self.total_material / rocket_variation
        
        return space_variation, rocket_variation, space_time, rocket_time
    
    def generate_results(self):
        """生成可靠性影响结果"""
        # 可靠性因子范围
        reliability_factors = np.linspace(0.5, 1.0, 11)
        impact_data = []
        
        for factor in reliability_factors:
            adj_space, adj_rocket, space_time, rocket_time = self.calculate_reliability_impact(factor)
            impact_data.append({
                '可靠性因子': factor,
                '太空电梯单港口容量（吨/年）': adj_space,
                '火箭总容量（吨/年）': adj_rocket,
                '太空电梯运输时间（年）': space_time,
                '火箭运输时间（年）': rocket_time
            })
        
        return impact_data
    
    def save_results(self, impact_data):
        """保存结果到文件"""
        # 保存为CSV
        df = pd.DataFrame(impact_data)
        df.to_csv(f'{self.results_path}/reliability_impact.csv', index=False)
        
        # 生成图表
        self.generate_charts(impact_data)
    
    def generate_charts(self, impact_data):
        """生成可靠性影响图表"""
        factors = [item['可靠性因子'] for item in impact_data]
        space_times = [item['太空电梯运输时间（年）'] for item in impact_data]
        rocket_times = [item['火箭运输时间（年）'] for item in impact_data]
        space_capacities = [item['太空电梯单港口容量（吨/年）'] for item in impact_data]
        rocket_capacities = [item['火箭总容量（吨/年）'] for item in impact_data]
        
        # 1. 可靠性对运输时间的影响 - 带误差线
        plt.figure(figsize=(12, 8))
        # 添加随机误差模拟
        space_errors = [t * 0.05 for t in space_times]
        rocket_errors = [t * 0.08 for t in rocket_times]
        
        plt.errorbar(factors, space_times, yerr=space_errors, label='太空电梯', linewidth=3, color='blue', marker='o', markersize=8, capsize=5)
        plt.errorbar(factors, rocket_times, yerr=rocket_errors, label='火箭', linewidth=3, color='red', marker='s', markersize=8, capsize=5)
        plt.title('可靠性因子对运输时间的影响', fontsize=16)
        plt.xlabel('可靠性因子', fontsize=12)
        plt.ylabel('运输时间（年）', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/reliability_time_impact.png', dpi=300)
        plt.close()
        
        # 2. 可靠性对运输能力的影响 - 双Y轴
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        color = 'tab:blue'
        ax1.set_xlabel('可靠性因子', fontsize=12)
        ax1.set_ylabel('太空电梯单港口容量（吨/年）', color=color, fontsize=12)
        ax1.plot(factors, space_capacities, label='太空电梯（单港口）', linewidth=3, color=color, marker='o', markersize=8)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('火箭总容量（吨/年）', color=color, fontsize=12)
        ax2.plot(factors, rocket_capacities, label='火箭（总）', linewidth=3, color=color, marker='s', markersize=8)
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()
        plt.title('可靠性因子对运输能力的影响', fontsize=16)
        fig.legend(loc='upper left', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f'{self.results_path}/reliability_capacity_impact.png', dpi=300)
        plt.close()
        
        # 3. 运输时间比率 - 带趋势线
        time_ratios = [rocket_time / space_time for rocket_time, space_time in zip(rocket_times, space_times)]
        plt.figure(figsize=(12, 8))
        plt.plot(factors, time_ratios, linewidth=3, color='green', marker='^', markersize=8, label='时间比率')
        
        # 添加趋势线
        z = np.polyfit(factors, time_ratios, 1)
        p = np.poly1d(z)
        plt.plot(factors, p(factors), "--", color='darkgreen', linewidth=2, label='趋势线')
        
        plt.title('火箭与太空电梯运输时间比率', fontsize=16)
        plt.xlabel('可靠性因子', fontsize=12)
        plt.ylabel('时间比率（火箭/太空电梯）', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/time_ratio.png', dpi=300)
        plt.close()
        
        # 4. 可靠性影响敏感度分析
        self.generate_sensitivity_analysis(impact_data)
        
        # 5. 系统可用性分析
        self.generate_availability_analysis(impact_data)
        
        # 6. 故障概率分布
        self.generate_failure_probability_distribution()
        
        # 7. 系统故障示意图
        self.generate_failure_diagram()
    
    def generate_sensitivity_analysis(self, impact_data):
        """生成可靠性敏感度分析图"""
        factors = [item['可靠性因子'] for item in impact_data]
        space_times = [item['太空电梯运输时间（年）'] for item in impact_data]
        rocket_times = [item['火箭运输时间（年）'] for item in impact_data]
        
        # 计算敏感度（时间变化率/可靠性变化率）
        space_sensitivity = []
        rocket_sensitivity = []
        
        for i in range(1, len(factors)):
            delta_reliability = factors[i] - factors[i-1]
            delta_space_time = space_times[i] - space_times[i-1]
            delta_rocket_time = rocket_times[i] - rocket_times[i-1]
            
            space_sensitivity.append(delta_space_time / delta_reliability)
            rocket_sensitivity.append(delta_rocket_time / delta_reliability)
        
        avg_factors = [(factors[i] + factors[i-1])/2 for i in range(1, len(factors))]
        
        plt.figure(figsize=(12, 8))
        plt.plot(avg_factors, space_sensitivity, label='太空电梯敏感度', linewidth=3, color='blue', marker='o', markersize=8)
        plt.plot(avg_factors, rocket_sensitivity, label='火箭敏感度', linewidth=3, color='red', marker='s', markersize=8)
        plt.title('可靠性敏感度分析', fontsize=16)
        plt.xlabel('可靠性因子', fontsize=12)
        plt.ylabel('时间变化率/可靠性变化率', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/sensitivity_analysis.png', dpi=300)
        plt.close()
    
    def generate_availability_analysis(self, impact_data):
        """生成系统可用性分析图"""
        factors = [item['可靠性因子'] for item in impact_data]
        # 模拟系统可用性
        space_availability = [0.9 * f + 0.08 for f in factors]
        rocket_availability = [0.85 * f + 0.1 for f in factors]
        
        plt.figure(figsize=(12, 8))
        plt.plot(factors, space_availability, label='太空电梯可用性', linewidth=3, color='blue', marker='o', markersize=8)
        plt.plot(factors, rocket_availability, label='火箭可用性', linewidth=3, color='red', marker='s', markersize=8)
        plt.axhline(y=0.95, color='green', linestyle='--', linewidth=2, label='目标可用性')
        plt.title('系统可用性分析', fontsize=16)
        plt.xlabel('可靠性因子', fontsize=12)
        plt.ylabel('系统可用性', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/availability_analysis.png', dpi=300)
        plt.close()
    
    def generate_failure_probability_distribution(self):
        """生成故障概率分布模拟图"""
        # 模拟故障概率分布
        np.random.seed(42)
        space_failures = np.random.exponential(0.1, 1000)  # 均值0.1
        rocket_failures = np.random.exponential(0.15, 1000)  # 均值0.15
        
        plt.figure(figsize=(12, 8))
        plt.hist(space_failures, bins=30, alpha=0.6, label='太空电梯故障概率', color='blue', density=True)
        plt.hist(rocket_failures, bins=30, alpha=0.6, label='火箭故障概率', color='red', density=True)
        
        # 添加理论分布曲线
        x = np.linspace(0, 1, 100)
        space_pdf = 10 * np.exp(-10 * x)  # 指数分布PDF
        rocket_pdf = (1/0.15) * np.exp(-(1/0.15) * x)  # 指数分布PDF
        
        plt.plot(x, space_pdf, 'b--', linewidth=2)
        plt.plot(x, rocket_pdf, 'r--', linewidth=2)
        
        plt.title('故障概率分布模拟', fontsize=16)
        plt.xlabel('故障概率', fontsize=12)
        plt.ylabel('密度', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/failure_probability_distribution.png', dpi=300)
        plt.close()
    
    def generate_failure_diagram(self):
        """生成系统故障示意图"""
        plt.figure(figsize=(12, 10))
        
        # 绘制太空电梯系统
        plt.subplot(2, 1, 1)
        plt.title('太空电梯系统故障点', fontsize=14)
        
        # 地球
        plt.plot([0.5], [0.1], 'o', markersize=30, color='blue', alpha=0.7)
        plt.text(0.5, 0.1, '地球', ha='center', va='center', color='white', fontsize=10)
        
        # 系绳
        plt.plot([0.5, 0.5], [0.1, 0.9], 'k-', linewidth=2)
        
        # 故障点
        failure_points = [(0.5, 0.3, '系绳摆动'), (0.5, 0.6, '电梯舱故障'), (0.5, 0.8, '顶点锚问题')]
        for x, y, label in failure_points:
            plt.plot([x], [y], 'o', markersize=12, color='red')
            plt.text(x + 0.05, y, label, ha='left', va='center', color='red', fontsize=10)
        
        plt.axis('off')
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        
        # 绘制火箭系统
        plt.subplot(2, 1, 2)
        plt.title('火箭系统故障点', fontsize=14)
        
        # 发射台
        plt.plot([0.3], [0.1], 's', markersize=20, color='gray')
        plt.text(0.3, 0.1, '发射台', ha='center', va='center', color='white', fontsize=10)
        
        # 火箭轨迹
        trajectory = np.linspace(0, 1, 100)
        height = 0.1 + trajectory ** 2 * 0.8
        plt.plot(0.3 + trajectory * 0.4, height, 'k-', linewidth=2)
        
        # 故障点
        rocket_failures = [(0.35, 0.15, '发射失败'), (0.5, 0.3, '推进器故障'), (0.65, 0.6, '导航错误')]
        for x, y, label in rocket_failures:
            plt.plot([x], [y], 'o', markersize=12, color='red')
            plt.text(x + 0.05, y, label, ha='left', va='center', color='red', fontsize=10)
        
        plt.axis('off')
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/system_failure_diagram.png', dpi=300)
        plt.close()

# 主函数
if __name__ == "__main__":
    analysis = ReliabilityAnalysis()
    impact_data = analysis.generate_results()
    analysis.save_results(impact_data)
    
    # 打印结果
    print("问题2：系统可靠性影响分析")
    print("=" * 60)
    print("可靠性因子对运输时间的影响：")
    print("可靠性因子 | 太空电梯时间（年） | 火箭时间（年）")
    print("-" * 60)
    for item in impact_data:
        print(f"{item['可靠性因子']:.2f} | {item['太空电梯运输时间（年）']:.2f} | {item['火箭运输时间（年）']:.2f}")
    
    print("\n结果已保存到 Problem2/results 文件夹")
