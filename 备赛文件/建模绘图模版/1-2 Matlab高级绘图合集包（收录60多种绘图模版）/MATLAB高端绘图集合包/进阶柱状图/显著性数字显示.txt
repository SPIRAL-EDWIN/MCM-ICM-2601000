%% 复现SCI柱状图插图
%% 清空工作区与所有窗口-三件套
clc;
clear;
close all;
%% 可以在PPT中进行颜色提取 对喜欢的图片与论文颜色提取
% 柱状图配色
bar_color = [255 242 189 ;190 216 255]./255;
% 散点图配色
scatter_color = [1 1 1]./255;

%% 自定义柱状图数据
X = [0.5 1 1.5;2.5 3 3.5]; % 1  2代表有两大组柱状图
Y = [1.8,1.5,1;2.3,2.5,2.2]; %每组有三个数
%% 自定义散点图数据
scatter_X = [0.4,0.5,0.6,  0.9,1,1.1,  1.43,1.5,1.58,...
    2.5,2.5,2.5,   2.9,3,3.1,   3.4,3.5,3.6];
scatter_Y = [1.83,1.75,1.82,   1.42,1.60,1.44,   0.89,1.17,0.93,...
    2.13,2.3,2.47,   2.37,2.68,2.45,   2.3,2.01,2.3];
%% 自定义误差棒的值 相对于Y上下数值
Y_low = [0,0.07,0.08;0.1,0.1,0.1];
Y_high =[0,0.07,0.08;0.1,0.1,0.1];
%% 开始绘图
% 设置背景为白色
figure('color',[1 1 1]);
% bar(X,Y,width) 宽度为柱状图宽度0.7，参数依次为颜色、边缘颜色与线宽
for i = 1:2
    b = bar(X(i,:),Y(i,:),0.7,'stacked','FaceColor',bar_color(i,:),'EdgeColor',...
        scatter_color,'Linewidth',1.5);
    hold on;
    % errorbar函数 绘制误差棒
    %errorbar(X,Y,Low,High)；参数以柱状图为基础、上下限高度、线宽、误差棒的长短
    errorbar(X(i,:),Y(i,:),Y_low(i,:),Y_high(i,:),...
        'LineStyle','none',...
        'Color',scatter_color,...
        'Linewidth',1.5,'Capsize',12);
end
%% 绘制散点图
% 绘制实心散点
% scatter(X,Y) 参数：点的大小、填充颜色
for i = 1:length(scatter_X)
    scatter(scatter_X(i),scatter_Y(i),35,'filled','MarkerFaceColor',scatter_color);
end
hold on;
%% 绘制直线 函数——plot:绘制直线  text：添加文本
plot([0.5,1.5],[2.1,2.1],'Color',scatter_color,'Linewidth',1.5);
text(0.51,2.3,'p = 0.0078','FontSize',15,'Fontname','微软雅黑');
hold on;
plot([0.5,2.5],[2.8,2.8],'Color',scatter_color,'Linewidth',1.5);
text(1.0,3,'p = 0.0315','FontSize',15,'Fontname','微软雅黑');
hold on;
plot([1,3],[3.28,3.28],'Color',scatter_color,'Linewidth',1.5);
text(1.6,3.5,'p = 0.0044','FontSize',15,'Fontname','微软雅黑');
hold on;
plot([1.5,3.5],[3.7,3.7],'Color',scatter_color,'Linewidth',1.5);
text(2,3.9,'p = 0.0057','FontSize',15,'Fontname','微软雅黑');
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

mycolor = [0,0.501960784313726,1;...
          1,1,1;....
          0.00392156862745098,0.00392156862745098,0.00392156862745098;...
          0.745098039215686,0.847058823529412,1;...
          1,0.949019607843137,0.741176470588235];