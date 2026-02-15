%% 复现SCI柱状图插图
%% 清空工作区与所有窗口
clc;
clear;
close all;
%% PPT颜色提取
bar_color = [227 188 61;170 195 166]./255;
scatter_color = [1 1 1]./255;
%% 自定义柱状图数据
X = [0.5 1 1.5;2.5 3 3.5]; % 1  2代表有两大组柱状图
Y = [1.8,1.5,1;2.3,2.5,2.2]; %每组有三个数
%% 自定义散点图数据
scatter_X = [0.4,0.5,0.6,  0.9,1,1.1,  1.43,1.5,1.58,...
    2.5,2.5,2.5,   2.9,3,3.1,   3.4,3.5,3.6];
scatter_Y = [1.83,1.75,1.82,   1.42,1.60,1.44,   0.89,1.17,0.93,...
    2.13,2.3,2.47,   2.37,2.68,2.45,   2.3,2.01,2.3];

%% 自定义误差棒的值
Y_low = [0,0.07,0.08;0.1,0.1,0.1];
Y_high = [0,0.07,0.08;0.1,0.1,0.1];
%% 开始绘图
%%初始化显示位置 [x,y,dx,dy] 从 x,y坐标开始，dx,dy为沿升的长度
figure('color',[1 1 1 ]);
for i = 1:2
    b = bar(X(i,:),Y(i,:),0.7,'stacked','FaceColor',bar_color(i,:),'EdgeColor',...
        scatter_color,'Linewidth',1.5);
    hold on;
    errorbar(X(i,:),Y(i,:),Y_low(i,:),Y_high(i,:),'LineStyle','none','Color',scatter_color,'Linewidth',...
        1.5,'Capsize',12);
end
for i = 1:length(scatter_X)
    scatter(scatter_X(i),scatter_Y(i),35,'filled','MarkerFaceColor',scatter_color);
end
hold on;
plot([0.5,1.5],[2.1,2.1],'Color',scatter_color,'Linewidth',1);
plot([0.5,0.5],[2.0 2.11],'Color',scatter_color,'Linewidth',1);
plot([1.5,1.5],[2.0 2.11],'Color',scatter_color,'Linewidth',1);
text(0.8,2.3,'***','FontSize',18,'Fontname','微软雅黑');
hold on;
plot([0.5,2.5],[2.8,2.8],'Color',scatter_color,'Linewidth',1);

plot([0.5,0.5],[2.7 2.81],'Color',scatter_color,'Linewidth',1);
plot([2.5,2.5],[2.7 2.81],'Color',scatter_color,'Linewidth',1);

text(1.4,3,'**','FontSize',18,'Fontname','微软雅黑');
hold on;
plot([1,3],[3.28,3.28],'Color',scatter_color,'Linewidth',1);
plot([1,1],[3.18 3.29],'Color',scatter_color,'Linewidth',1);
plot([3,3],[3.18 3.29],'Color',scatter_color,'Linewidth',1);
text(1.9,3.5,'**','FontSize',18,'Fontname','微软雅黑');
hold on;
plot([1.5,3.5],[3.7,3.7],'Color',scatter_color,'Linewidth',1);
plot([1.5,1.5],[3.6 3.71],'Color',scatter_color,'Linewidth',1);
plot([3.5,3.5],[3.6 3.71],'Color',scatter_color,'Linewidth',1);
text(2.3,3.9,'***','FontSize',18,'Fontname','微软雅黑');
    

%% 设置图例
set(gca, 'Box', 'off','XGrid', 'off', 'YGrid', 'off','TickDir', 'out', 'TickLength', [.005 .005]);          
% 对Y轴显示范围与刻度设置
set(gca,'YTick', 0:1:4, 'Ylim',[0,4]);
% 对X轴显示范围与横坐标显示设置
set(gca,'Xlim',[0 4],'Xtick', [0:0.5:4], 'Xticklabel',{' ','0.5','4','8',' ','0.5','4','8'});
% 对坐标轴添加图例
ylabel(['Keyannantx30-Matlab']);
set(gca, 'FontName', 'Arial', 'FontSize', 15)
set(gca,'linewidth',1.5)