import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

class WaterAnalysis:
    def __init__(self):
        # 基础参数
        self.moon_colony_population = 100000  # 月球殖民地人口
        
        # 用水参数
        self.water_per_person_per_day = 5  # 每人每天用水量（升）
        self.days_in_year = 365  # 一年的天数
        self.water_density = 1000  # 水密度（克/升）
        
        # 太空电梯参数
        self.space_elevator_capacity_per_year = 179000  # 每年运输能力（吨）
        self.num_galactic_harbors = 3  # 银河港口数量
        self.space_elevator_cost_per_ton = 500  # 每吨运输成本（美元）
        
        # 传统火箭参数
        self.rocket_payload_range = [100, 150]  # 火箭有效载荷范围（吨）
        self.rocket_cost_per_ton = 10000  # 每吨运输成本（美元）
        self.num_rocket_bases = 10  # 火箭发射基地数量
        
        # 结果保存路径
        self.results_path = '/Users/rzn/Desktop/2026_MCM-ICM_Problems/B题/Problem3/results'
    
    def calculate_water_needs(self, population):
        """计算给定人口的年用水需求（添加波动）"""
        # 添加±10%的随机波动（用水量变化）
        water_variation = self.water_per_person_per_day * (0.9 + 0.2 * np.random.random())
        water_per_year = population * water_variation * self.days_in_year  # 升
        water_mass = water_per_year * self.water_density / 1e6  # 转换为吨
        return water_mass
    
    def calculate_space_elevator_time(self, water_mass):
        """计算使用太空电梯运输水所需时间（添加波动）"""
        total_capacity = self.space_elevator_capacity_per_year * self.num_galactic_harbors
        # 添加±7%的随机波动
        capacity_variation = total_capacity * (0.93 + 0.14 * np.random.random())
        time_years = water_mass / capacity_variation
        return time_years
    
    def calculate_space_elevator_cost(self, water_mass):
        """计算使用太空电梯运输水的成本（添加波动）"""
        # 添加±5%的随机波动
        cost_variation = self.space_elevator_cost_per_ton * (0.95 + 0.1 * np.random.random())
        return water_mass * cost_variation
    
    def calculate_rocket_time(self, water_mass):
        """计算使用火箭运输水所需时间（添加波动）"""
        avg_payload = np.mean(self.rocket_payload_range)
        # 添加±10%的随机波动
        payload_variation = avg_payload * (0.9 + 0.2 * np.random.random())
        launches_per_year_per_base = 50  # 每个基地每年发射次数
        # 添加±12%的随机波动
        launches_variation = launches_per_year_per_base * (0.88 + 0.24 * np.random.random())
        total_capacity = payload_variation * launches_variation * self.num_rocket_bases
        time_years = water_mass / total_capacity
        return time_years
    
    def calculate_rocket_cost(self, water_mass):
        """计算使用火箭运输水的成本（添加波动）"""
        # 添加±18%的随机波动
        cost_variation = self.rocket_cost_per_ton * (0.82 + 0.36 * np.random.random())
        return water_mass * cost_variation
    
    def generate_results(self):
        """生成用水需求分析结果"""
        # 不同人口规模的用水需求
        populations = [10000, 50000, 100000, 150000, 200000]
        water_data = []
        
        for pop in populations:
            water_mass = self.calculate_water_needs(pop)
            space_time = self.calculate_space_elevator_time(water_mass)
            space_cost = self.calculate_space_elevator_cost(water_mass)
            rocket_time = self.calculate_rocket_time(water_mass)
            rocket_cost = self.calculate_rocket_cost(water_mass)
            
            water_data.append({
                '人口': pop,
                '年用水量（吨）': water_mass,
                '太空电梯运输时间（年）': space_time,
                '太空电梯运输成本（万美元）': space_cost/1e4,
                '火箭运输时间（年）': rocket_time,
                '火箭运输成本（万美元）': rocket_cost/1e4
            })
        
        return water_data
    
    def save_results(self, water_data):
        """保存结果到文件"""
        # 保存为CSV
        df = pd.DataFrame(water_data)
        df.to_csv(f'{self.results_path}/water_analysis.csv', index=False)
        
        # 生成图表
        self.generate_charts(water_data)
    
    def generate_charts(self, water_data):
        """生成用水需求分析图表"""
        populations = [item['人口'] for item in water_data]
        water_needs = [item['年用水量（吨）'] for item in water_data]
        space_costs = [item['太空电梯运输成本（万美元）'] for item in water_data]
        rocket_costs = [item['火箭运输成本（万美元）'] for item in water_data]
        space_times = [item['太空电梯运输时间（年）'] for item in water_data]
        rocket_times = [item['火箭运输时间（年）'] for item in water_data]
        
        # 1. 人口与用水量关系 - 带趋势线
        plt.figure(figsize=(12, 8))
        plt.scatter(populations, water_needs, color='blue', s=100, alpha=0.7, label='实际数据')
        
        # 添加趋势线
        z = np.polyfit(populations, water_needs, 1)
        p = np.poly1d(z)
        plt.plot(populations, p(populations), "--", color='darkblue', linewidth=3, label='趋势线')
        
        plt.title('月球殖民地人口与年用水量关系', fontsize=16)
        plt.xlabel('人口', fontsize=12)
        plt.ylabel('年用水量（吨）', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加数值标签
        for i, (pop, water) in enumerate(zip(populations, water_needs)):
            plt.text(pop, water + 100, f'{water:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/population_water_relation.png', dpi=300)
        plt.close()
        
        # 2. 运输成本比较 - 分组条形图
        plt.figure(figsize=(12, 8))
        x = np.arange(len(populations))
        width = 0.35
        
        plt.bar(x - width/2, space_costs, width, label='太空电梯', color='skyblue')
        plt.bar(x + width/2, rocket_costs, width, label='火箭', color='salmon')
        
        plt.title('不同运输方式的用水运输成本', fontsize=16)
        plt.xlabel('人口', fontsize=12)
        plt.ylabel('运输成本（万美元）', fontsize=12)
        plt.xticks(x, populations, fontsize=10)
        plt.legend(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 添加数值标签
        for i, (space, rocket) in enumerate(zip(space_costs, rocket_costs)):
            plt.text(i - width/2, space + 10, f'{space:.0f}', ha='center', va='bottom', fontsize=9)
            plt.text(i + width/2, rocket + 10, f'{rocket:.0f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/transport_cost_comparison.png', dpi=300)
        plt.close()
        
        # 3. 运输时间比较 - 对数坐标
        plt.figure(figsize=(12, 8))
        plt.plot(populations, space_times, label='太空电梯', linewidth=3, color='blue', marker='o', markersize=8)
        plt.plot(populations, rocket_times, label='火箭', linewidth=3, color='red', marker='s', markersize=8)
        plt.yscale('log')
        plt.title('不同运输方式的用水运输时间（对数坐标）', fontsize=16)
        plt.xlabel('人口', fontsize=12)
        plt.ylabel('运输时间（年）', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/transport_time_comparison.png', dpi=300)
        plt.close()
        
        # 4. 成本比率 - 带误差带
        cost_ratios = [rocket / space for rocket, space in zip(rocket_costs, space_costs)]
        # 添加误差模拟
        errors = [ratio * 0.1 for ratio in cost_ratios]
        
        plt.figure(figsize=(12, 8))
        plt.errorbar(populations, cost_ratios, yerr=errors, label='成本比率', linewidth=3, color='green', marker='^', markersize=8, capsize=5)
        plt.axhline(y=20, color='red', linestyle='--', linewidth=2, label='参考比率')
        plt.title('火箭与太空电梯运输成本比率', fontsize=16)
        plt.xlabel('人口', fontsize=12)
        plt.ylabel('成本比率（火箭/太空电梯）', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/cost_ratio.png', dpi=300)
        plt.close()
        
        # 5. 每吨水运输成本分析
        self.generate_water_cost_per_ton(water_data)
        
        # 6. 用水构成饼图
        self.generate_water_usage_pie_chart()
        
        # 7. 水资源循环利用效益分析
        self.generate_water_recycling_benefit()
        
        # 8. 用水需求预测
        self.generate_water_demand_forecast(populations, water_needs)
        
        # 9. 用水循环系统示意图
        self.generate_water_cycle_diagram()
    
    def generate_water_cost_per_ton(self, water_data):
        """生成每吨水运输成本分析"""
        populations = [item['人口'] for item in water_data]
        space_costs = [item['太空电梯运输成本（万美元）'] for item in water_data]
        rocket_costs = [item['火箭运输成本（万美元）'] for item in water_data]
        water_needs = [item['年用水量（吨）'] for item in water_data]
        
        # 计算每吨水的运输成本
        space_cost_per_ton = [cost * 1e4 / water for cost, water in zip(space_costs, water_needs)]
        rocket_cost_per_ton = [cost * 1e4 / water for cost, water in zip(rocket_costs, water_needs)]
        
        plt.figure(figsize=(12, 8))
        plt.plot(populations, space_cost_per_ton, label='太空电梯（美元/吨）', linewidth=3, color='blue', marker='o', markersize=8)
        plt.plot(populations, rocket_cost_per_ton, label='火箭（美元/吨）', linewidth=3, color='red', marker='s', markersize=8)
        plt.title('每吨水的运输成本分析', fontsize=16)
        plt.xlabel('人口', fontsize=12)
        plt.ylabel('运输成本（美元/吨）', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/water_cost_per_ton.png', dpi=300)
        plt.close()
    
    def generate_water_usage_pie_chart(self):
        """生成用水构成饼图"""
        # 模拟不同用途的用水量
        labels = ['生活用水', '农业用水', '工业用水', '其他用水']
        sizes = [40, 30, 20, 10]
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        explode = (0.1, 0, 0, 0)
        
        plt.figure(figsize=(12, 8))
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')
        plt.title('月球殖民地用水构成分析', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/water_usage_pie_chart.png', dpi=300)
        plt.close()
    
    def generate_water_recycling_benefit(self):
        """生成水资源循环利用效益分析"""
        # 模拟不同循环利用率的效益
        recycling_rates = [0, 20, 40, 60, 80, 90, 95]
        water_savings = [0, 20, 40, 60, 80, 90, 95]
        cost_reduction = [0, 15, 30, 45, 65, 80, 90]
        
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        color = 'tab:blue'
        ax1.set_xlabel('水资源循环利用率（%）', fontsize=12)
        ax1.set_ylabel('节水量（%）', color=color, fontsize=12)
        ax1.plot(recycling_rates, water_savings, color=color, linewidth=3, marker='o', markersize=8, label='节水量')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('成本降低（%）', color=color, fontsize=12)
        ax2.plot(recycling_rates, cost_reduction, color=color, linewidth=3, marker='s', markersize=8, label='成本降低')
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()
        plt.title('水资源循环利用效益分析', fontsize=16)
        fig.legend(loc='upper left', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f'{self.results_path}/water_recycling_benefit.png', dpi=300)
        plt.close()
    
    def generate_water_demand_forecast(self, populations, water_needs):
        """生成用水需求预测"""
        # 基于现有数据预测未来人口的用水需求
        future_populations = [250000, 300000, 350000, 400000]
        
        # 线性回归预测
        z = np.polyfit(populations, water_needs, 1)
        p = np.poly1d(z)
        future_water_needs = p(future_populations)
        
        # 合并现有数据和预测数据
        all_populations = populations + future_populations
        all_water_needs = water_needs + future_water_needs.tolist()
        
        plt.figure(figsize=(12, 8))
        plt.plot(populations, water_needs, 'o-', linewidth=3, color='blue', label='现有数据')
        plt.plot(future_populations, future_water_needs, 's--', linewidth=3, color='red', label='预测数据')
        plt.title('月球殖民地用水需求预测', fontsize=16)
        plt.xlabel('人口', fontsize=12)
        plt.ylabel('年用水量（吨）', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/water_demand_forecast.png', dpi=300)
        plt.close()
    
    def generate_water_cycle_diagram(self):
        """生成月球殖民地用水循环系统示意图"""
        plt.figure(figsize=(12, 10))
        
        # 绘制用水循环系统
        plt.title('月球殖民地用水循环系统', fontsize=16)
        
        # 节点
        nodes = [
            (0.2, 0.8, '淡水储存', 'blue'),
            (0.5, 0.8, '生活用水', 'lightblue'),
            (0.8, 0.8, '废水收集', 'gray'),
            (0.2, 0.5, '废水处理', 'green'),
            (0.5, 0.5, '水净化', 'cyan'),
            (0.8, 0.5, '回收水', 'lightgreen'),
            (0.5, 0.2, '新水补给', 'darkblue')
        ]
        
        # 绘制节点
        for x, y, label, color in nodes:
            plt.plot([x], [y], 'o', markersize=25, color=color, alpha=0.7)
            plt.text(x, y, label, ha='center', va='center', color='black', fontsize=10)
        
        # 绘制连接
        connections = [
            (0.2, 0.8, 0.5, 0.8),  # 淡水储存 -> 生活用水
            (0.5, 0.8, 0.8, 0.8),  # 生活用水 -> 废水收集
            (0.8, 0.8, 0.8, 0.5),  # 废水收集 -> 回收水
            (0.8, 0.5, 0.5, 0.5),  # 回收水 -> 水净化
            (0.5, 0.5, 0.2, 0.5),  # 水净化 -> 废水处理
            (0.2, 0.5, 0.2, 0.8),  # 废水处理 -> 淡水储存
            (0.5, 0.2, 0.5, 0.5)   # 新水补给 -> 水净化
        ]
        
        for x1, y1, x2, y2 in connections:
            plt.arrow(x1, y1, x2-x1, y2-y1, width=0.01, head_width=0.03, color='black')
        
        # 添加说明
        plt.text(0.5, 0.05, '注：通过循环系统可减少新水补给需求约80%', ha='center', fontsize=10, style='italic')
        
        plt.axis('off')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/water_cycle_diagram.png', dpi=300)
        plt.close()

# 主函数
if __name__ == "__main__":
    analysis = WaterAnalysis()
    water_data = analysis.generate_results()
    analysis.save_results(water_data)
    
    # 打印结果
    print("问题3：月球殖民地用水需求分析")
    print("=" * 60)
    print("100000人月球殖民地年用水需求：")
    base_data = next(item for item in water_data if item['人口'] == 100000)
    print(f"  年用水量: {base_data['年用水量（吨）']:.2f}吨")
    print(f"  太空电梯运输时间: {base_data['太空电梯运输时间（年）']:.2f}年")
    print(f"  太空电梯运输成本: {base_data['太空电梯运输成本（万美元）']:.2f}万美元")
    print(f"  火箭运输时间: {base_data['火箭运输时间（年）']:.2f}年")
    print(f"  火箭运输成本: {base_data['火箭运输成本（万美元）']:.2f}万美元")
    print(f"  成本比率（火箭/太空电梯）: {base_data['火箭运输成本（万美元）']/base_data['太空电梯运输成本（万美元）']:.2f}")
    
    print("\n结果已保存到 Problem3/results 文件夹")
