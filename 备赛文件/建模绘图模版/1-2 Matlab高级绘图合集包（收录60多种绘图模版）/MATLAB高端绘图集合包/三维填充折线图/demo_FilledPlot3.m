clc;
clear;
close all;
%三维填充折线图
%% 数据准备
% 读取数据
load data.mat
% 自变量
X = x;
% 因变量
Z = data(1:5,:)';

%% 颜色定义
map = colormap(nclCM(486));%color包里选颜色

%% 图片尺寸设置（单位：厘米）
close all;
figureUnits = 'centimeters';
figureWidth = 15;
figureHeight = 11;

%% 窗口设置
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);

%% 三维填充折线图绘制
p = FilledPlot3(X,Z,map,1,0.8);
hTitle = title('Extracted Spectra Subset');
hXLabel = xlabel('Mass/Charge (M/Z)');
hYLabel = ylabel('Samples');
hZLabel = zlabel('Ion Spectra');
view(45,45)

%% 细节优化
% 坐标区调整
% Y刻度标签定义
temp = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
ylbs = sprintfc('Samp %c',temp(1:5));
set(gca, 'Box', 'on', ...                                 % 边框
         'LineWidth',1,...                                % 线宽
         'XGrid', 'on', 'YGrid', 'on', 'ZGrid', 'on',...  % 网格
         'TickDir', 'out', 'TickLength', [.015 .015], ... % 刻度
         'XMinorTick', 'off', 'YMinorTick', 'off', ...    % 小刻度
         'XColor', [.1 .1 .1],  'YColor', [.1 .1 .1],...  % 坐标轴颜色
         'ytick',0:5,...
         'ylim',[-0.5 4.5],...
         'yticklabels',ylbs,...
         'zlim',[0 1.5])
% 字体和字号
set(gca, 'FontName', 'Arial', 'FontSize', 10)
set([hXLabel, hYLabel,hZLabel], 'FontSize', 11, 'FontName', 'Arial')
set(hTitle, 'FontSize', 12, 'FontWeight' , 'bold')
% 背景颜色
set(gcf,'Color',[1 1 1])

%% 图片输出
figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[0 0 figW figH]);
fileout = 'test';
print(figureHandle,[fileout,'.png'],'-r300','-dpng');