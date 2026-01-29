clc;
clear;
close all;
%条带热图
%% 数据准备
% 读取数据
load data.mat
% 初始化参数
x = X;
y = Y;
z = Z;
sw = 15; % 条带宽度

%% 图片尺寸设置（单位：厘米）
figureUnits = 'centimeters';
figureWidth = 16;
figureHeight = 12;

%% 窗口设置
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);

%% 条带热图绘制
stripheatmap(x,y,z,sw)
hTitle = title('Strip Heatmap Plot');
hXLabel = xlabel('K (w)');
hYLabel = ylabel('Samples');
view(0,90)

%% 细节优化
% 赋色
colormap(nclCM(309)) %color包里选颜色
colorbar
% 坐标区调整
% Y刻度标签定义
temp = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
ylbs = sprintfc('Samp %c',temp(1:20));
set(gca, 'Box', 'off', ...                                % 边框
         'LineWidth',1,...                                % 线宽
         'XGrid', 'off', 'YGrid', 'off', ...              % 网格
         'TickDir', 'out', 'TickLength', [.01 .01], ...   % 刻度
         'XMinorTick', 'off', 'YMinorTick', 'off', ...    % 小刻度
         'XColor', [.1 .1 .1],  'YColor', [.1 .1 .1],...  % 坐标轴颜色
         'xlim',[0 95],...                                % 坐标轴范围
         'ytick',1:15,...
         'ylim',[0 16],...
         'yticklabels',ylbs)
% 字体和字号
set(gca, 'FontName', 'Arial', 'FontSize', 10)
set([hXLabel, hYLabel], 'FontSize', 11, 'FontName', 'Arial')
set(hTitle, 'FontSize', 12, 'FontWeight' , 'bold')
% 背景颜色
set(gcf,'Color',[1 1 1])
% 添加上、右框线
xc = get(gca,'XColor');
yc = get(gca,'YColor');
unit = get(gca,'units');
ax = axes( 'Units', unit,...
           'Position',get(gca,'Position'),...
           'XAxisLocation','top',...
           'YAxisLocation','right',...
           'Color','none',...
           'XColor',xc,...
           'YColor',yc);
set(ax, 'linewidth',1,...
        'XTick', [],...
        'YTick', []);

%% 图片输出
figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[0 0 figW figH]);
fileout = 'test';
print(figureHandle,[fileout,'.png'],'-r300','-dpng');