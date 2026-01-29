clc;
clear;
close all;
%三维密度散点图

%% 数据准备
% 读取数据
load data.mat
% 初始化绘图参数
data = [x,y,z];
% 密度计算
radius = 1; % 定义半径
density_3D = density3D_KD(data(:,1:3),radius); % 3D密度

%% 颜色定义
map = colormap(nclCM(309));%color包里选颜色
% map = flipud(map);

%% 图片尺寸设置（单位：厘米）
close all;
figureUnits = 'centimeters';
figureWidth = 16;
figureHeight = 12;

%% 窗口设置
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);
hold on

%% 三维密度散点图绘制
scatter3(data(:,1), data(:,2), data(:,3), 5, density_3D, 'filled')
hTitle = title('Density Scatter3');
hXLabel = xlabel('XAxis');
hYLabel = ylabel('YAxis');
hZLabel = zlabel('ZAxis');

%% 细节优化
% 赋色
colormap(map)
colorbar
% 坐标轴美化
view(45,27)
% 俯视
% view(0,90)
% set(gca,'xlim',[-8.5 8.5],...
%         'ylim',[-8.5 8.5])

set(gca, 'Box', 'on', ...                                                     % 边框
         'XGrid', 'on', 'YGrid', 'on', 'ZGrid', 'on',...                      % 网格
         'TickDir', 'out', 'TickLength', [.005 .005], ...                     % 刻度
         'XColor', [.1 .1 .1],  'YColor', [.1 .1 .1],  'ZColor', [.1 .1 .1])  % 坐标轴颜色
% 字体和字号
set(gca, 'FontName', 'Arial', 'FontSize', 10)
set([hXLabel, hYLabel, hZLabel], 'FontSize', 11, 'FontName', 'Arial')
set(hTitle, 'FontSize', 12, 'FontWeight' , 'bold')
% 背景颜色
set(gcf,'Color',[1 1 1])

%% 图片输出
figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[0 0 figW figH]);
fileout = 'test';
print(figureHandle,[fileout,'.png'],'-r600','-dpng');