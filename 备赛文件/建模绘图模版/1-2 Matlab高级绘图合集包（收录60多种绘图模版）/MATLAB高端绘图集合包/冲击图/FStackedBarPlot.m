clc;
clear;
close all;
%冲击图

%% 数据准备
% 读取数据
load data.mat
% 初始化绘图参数
y = A(:,1:6);

%% 颜色定义
map = colormap(nclCM(486));%color包里选颜色
C = map(1:4,:);

%% 图片尺寸设置（单位：厘米）
close all;
figureUnits = 'centimeters';
figureWidth = 15;
figureHeight = 12;

%% 窗口设置
figureHandle = figure('color','w');
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);
hold on

%% 冲击图绘制
GO = Fbarstacked(y,0.4,C,0.7);
hTitle = title('Filled stacked bar chart');
hXLabel = xlabel('Samples');
hYLabel = ylabel('RMSE (m)');

%% 细节优化
% 坐标区基本属性调整
set(gca, 'Box', 'off', ...                                         % 边框
         'LineWidth', 1, ...                                       % 线宽
         'XGrid', 'off', 'YGrid', 'on', ...                        % 网格
         'TickDir', 'out', 'TickLength', [.01 .01], ...            % 刻度
         'XMinorTick', 'off', 'YMinorTick', 'off', ...             % 小刻度
         'XColor', [.1 .1 .1],  'YColor', [.1 .1 .1])              % 坐标轴颜色
% 坐标轴刻度调整
set(gca, 'YTick', 0:0.3:1.5,...
         'Ylim' , [0 1.5], ...
         'Xlim' , [0.5 6.5], ...
         'XTick', 1:7,...
         'Xticklabel',{1:7},...
         'Yticklabel',{0:0.3:1.5})
% legend
hLegend = legend([GO(1),GO(2),GO(3),GO(4)],...
         {'A', 'B', 'C', 'D'});
% 字体和字号
set(gca, 'FontName', 'Arial', 'FontSize', 10)
set([hLegend, hXLabel, hYLabel], 'FontName', 'Arial', 'FontSize', 11)
set(hTitle, 'FontSize', 12, 'FontWeight' , 'bold')
% 背景颜色
set(gcf,'Color',[1 1 1])

%% 图片输出
exportgraphics(figureHandle,'test.png','Resolution',300)