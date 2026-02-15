clc;
clear;
close all;
%带填充纹理的柱状图
%% 数据准备
% 读取数据
load data.mat
% 初始化参数
X = x;
Y = dataset;

%% 颜色定义
map = colormap(nclCM(486));%color包里选颜色
C1 = map(1,1:3);
C2 = map(2,1:3);
C3 = map(3,1:3);
C4 = map(6,1:3);

%% 图片尺寸设置（单位：厘米）
close all;
figureUnits = 'centimeters';
figureWidth = 16;
figureHeight = 12;

%% 窗口设置
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);

%% 纹理填充柱状图绘制
GO = bar(X,Y,1,'EdgeColor','k','LineWidth',1);
hatchfill2(GO(1),'cross','HatchAngle',45,'HatchDensity',40,'HatchColor','k');
hatchfill2(GO(2),'single','HatchAngle',45,'HatchDensity',40,'HatchColor','k');
hatchfill2(GO(3),'single','HatchAngle',0,'HatchDensity',40,'HatchColor','k');
hatchfill2(GO(4),'single','HatchAngle',-45,'HatchDensity',40,'HatchColor','k');
hTitle = title('Texture filled bar chart');
hXLabel = xlabel('Samples');
hYLabel = ylabel('RMSE (m)');

%% 细节优化
% 赋色
GO(1).FaceColor = C1;
GO(2).FaceColor = C2;
GO(3).FaceColor = C3;
GO(4).FaceColor = C4;
% 坐标区基本属性调整
set(gca, 'Box', 'off', ...                                         % 边框
         'LineWidth', 1, ...                                       % 线宽
         'XGrid', 'off', 'YGrid', 'on', ...                        % 网格
         'TickDir', 'out', 'TickLength', [.01 .01], ...            % 刻度
         'XMinorTick', 'off', 'YMinorTick', 'off', ...             % 小刻度
         'XColor', [.1 .1 .1],  'YColor', [.1 .1 .1])              % 坐标轴颜色
% 坐标轴刻度调整
set(gca, 'YTick', 0:0.1:1,...
         'Ylim' , [0 0.6], ...
         'XTick', 1:4,...
         'Xticklabel',{'samp1' 'samp2' 'samp3' 'samp4'},...
         'Yticklabel',{0:0.1:1})
% legend
[hLegend, object_h, plot_h, text_str] = legendflex([GO(1),GO(2),GO(3),GO(4)],...
         {'A', 'B', 'C', 'D'},...
         'Location', 'NorthEast',...
         'FontName','Arial',...
         'FontSize', 10);
hatchfill2(object_h(5), 'cross', 'HatchAngle', 45, 'HatchDensity', 10, 'HatchColor', 'black');
hatchfill2(object_h(6), 'single', 'HatchAngle', 45, 'HatchDensity', 10, 'HatchColor', 'black');
hatchfill2(object_h(7), 'single', 'HatchAngle', 0, 'HatchDensity', 15, 'HatchColor', 'black');
hatchfill2(object_h(8), 'single', 'HatchAngle', -45, 'HatchDensity', 10, 'HatchColor', 'black');
% 字体和字号
set(gca, 'FontName', 'Arial', 'FontSize', 10)
set([hXLabel, hYLabel], 'FontName', 'Arial', 'FontSize', 11)
set(hTitle, 'FontSize', 12, 'FontWeight' , 'bold')
% 背景颜色
set(gcf,'Color',[1 1 1])

%% 图片输出
exportgraphics(figureHandle,'test1.png','Resolution',300)