clc;
clear;
close all;
%带类别标签的三维柱状图
%% 数据准备
% 读取数据
load data.mat
% 初始化
s = 0.4; % 柱子宽度
MA = dataset;
Labels = lbs;

%% 颜色定义
map = colormap(nclCM(486));%color包里选颜色

%% 图片尺寸设置（单位：厘米）
close all;
figureUnits = 'centimeters';
figureWidth = 18;
figureHeight = 16;

%% 窗口设置
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);

%% 带类别标签的三维柱状图绘制
h = bar3withlabel(MA,s,map,Labels);
hTitle = title('Bar3withLabels Plot');
hXLabel = xlabel('Variable1');
hYLabel = ylabel('Variable2');
hZLabel = zlabel('Variable3');
view(134,25)
% alpha(0.9) % 透明度

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
         'zlim',[0 20])
% Legend设置 
[u1,u2,u3] = unique(lbs(:),'stable');
idx = [4 5 2 1 3];
% idx = 1:length(u1); %标签默认顺序
u1 = num2str(u1);
hLegend = legend(h(u2(idx)),u1(idx),...
                 'Location', 'northwest',...
                 'Orientation','vertical');
% hLegend.ItemTokenSize = [5 5];
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