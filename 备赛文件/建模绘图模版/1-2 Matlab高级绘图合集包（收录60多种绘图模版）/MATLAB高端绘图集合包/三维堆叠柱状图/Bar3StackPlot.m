clc;
clear;
close all;
%三维堆叠柱状图
%% 数据准备
% 读取数据
load data.mat
% 初始化
dataset = X;
s = 0.4; % 柱子宽度
n = size(dataset,3); % 堆叠组数

%% 图片尺寸设置（单位：厘米）
figureUnits = 'centimeters';
figureWidth = 18;
figureHeight = 16;

%% 窗口设置
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);

%% 三维堆叠柱状图绘制
% 赋色
map = colormap(nclCM(309)); %color包里选颜色
h = bar3stack(dataset,s,map);
hTitle = title('Bar3Stack Plot');
hXLabel = xlabel('Variable1');
hYLabel = ylabel('Variable2');
hZLabel = zlabel('Variable3');
view(134,25)


%% 细节优化
% 坐标区调整
set(gca, 'Box', 'on', ...                                                           % 边框
         'LineWidth', 1, 'GridLineStyle', '-',...                                   % 坐标轴线宽
         'XGrid', 'on', 'YGrid', 'on','ZGrid', 'on', ...                            % 网格
         'TickDir', 'out', 'TickLength', [.015 .015], ...                           % 刻度
         'XMinorTick', 'off', 'YMinorTick', 'off',  'ZMinorTick', 'off',...         % 小刻度
         'XColor', [.1 .1 .1],  'YColor', [.1 .1 .1], 'ZColor', [.1 .1 .1],...      % 坐标轴颜色
         'xtick',1:10,...                                                           % 坐标轴刻度
         'xticklabels',1:10,...
         'ytick',1:10,...
         'ylim',[0.5 10.5],...
         'yticklabels',1:10,...
         'ztick',0:10:60,...
         'zticklabels',0:10:60,...
         'zlim',[0 60])
% Legend设置    
hLegend = legend(h,...
                 'Samp1','Samp2','Samp3','Samp4','Samp5', ...
                 'Location', 'northwest',...
                 'Orientation','vertical');

% Legend位置微调 
P = hLegend.Position;
hLegend.Position = P + [0.05 -0.2 0 0];
% 字体和字号
set(gca, 'FontName', 'Arail', 'FontSize', 10)
set([hLegend,hXLabel, hYLabel,hZLabel], 'FontName', 'Arail', 'FontSize', 10)
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