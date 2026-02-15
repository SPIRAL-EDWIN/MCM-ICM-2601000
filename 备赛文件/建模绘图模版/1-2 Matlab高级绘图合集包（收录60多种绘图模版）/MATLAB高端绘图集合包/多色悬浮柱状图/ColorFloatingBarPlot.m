clc;
clear;
close all;
%多色悬浮柱状图

%% 数据准备
% 读取数据
load data.mat
% 初始化绘图参数
X = x;
Y1 = data1;
Y2 = data2;
lbs = {'Jan.','Feb.','Mar.','Apr.','May.','Jun.','Jul.','Aug.','Sep.','Oct.','Nov.','Dec'};

%% 颜色定义
N = length(X);
map1 = colormap(nclCM(486));%color包里选颜色
map = map1(1:N,:);

%% 图片尺寸设置（单位：厘米）
close all;
figureUnits = 'centimeters';
figureWidth = 15;
figureHeight = 13;

%% 窗口设置
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);

%% 多色悬浮柱状图绘制
h = Floatingbar(X,Y1,Y2,map,0.7);
hTitle = title('Floating bar chart');
hXLabel = xlabel('Month');
hYLabel = ylabel('Temperature(\circC)');

%% 细节优化
% 目标属性调整
set(h,'EdgeColor','none')
% 坐标区调整
set(gca, 'Box', 'off', ...                                         % 边框
         'LineWidth', 1, 'GridLineStyle', '-',...                  % 坐标轴线宽
         'XGrid', 'off', 'YGrid', 'on', ...                        % 网格
         'TickDir', 'out', 'TickLength', [.01 .01], ...            % 刻度
         'XMinorTick', 'off', 'YMinorTick', 'off', ...             % 小刻度
         'XColor', [.1 .1 .1],  'YColor', [.1 .1 .1])              % 坐标轴颜色
set(gca, 'XTick', 1:12,...
         'Xlim' , [0.3 12.7], ... 
         'Xticklabel',lbs)
% 字体和字号
set(gca, 'FontName', 'Arial', 'FontSize', 9)
set([hXLabel,hYLabel], 'FontSize', 11, 'FontName', 'Arial')
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