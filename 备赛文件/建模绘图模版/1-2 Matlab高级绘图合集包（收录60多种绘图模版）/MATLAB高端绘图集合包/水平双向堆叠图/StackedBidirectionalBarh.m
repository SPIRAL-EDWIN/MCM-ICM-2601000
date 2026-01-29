clc;
clear;
close all;
%水平双向堆叠图绘制模板

%% 数据准备
% 读取数据
load data.mat
% 初始化绘图参数
x = X;
y1 = A1;
y2 = A2;

%% 颜色定义
C1 = colormap(nclCM(486));%color包里选颜色
C = C1(1:6,:);

%% 图片尺寸设置（单位：厘米）
close all;
figureUnits = 'centimeters';
figureWidth = 16;
figureHeight = 12;

%% 窗口设置
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]); % define the new figure dimensions
hold on

%% 水平双向堆叠图绘制
GO1 = barh(x,y1,0.9,'stacked','EdgeColor','none');
GO2 = barh(x,y2,0.9,'stacked','EdgeColor','none');
hTitle = title('Stacked Bidirectional barh chart');
hXLabel = xlabel('Xaxis');
hYLabel = ylabel('Yaxis');

%% 细节优化
% 赋色
for i = 1:6
    GO1(i).FaceColor = C(i,1:3);
    GO2(i).FaceColor = C(i,1:3);
end
% 坐标区调整
set(gca, 'Box', 'off', ...                                         % 边框
         'LineWidth', 1, 'GridLineStyle', '-',...                  % 坐标轴线宽
         'XGrid', 'on', 'YGrid', 'off', ...                        % 网格
         'TickDir', 'out', 'TickLength', [.01 .01], ...            % 刻度
         'XMinorTick', 'off', 'YMinorTick', 'off', ...             % 小刻度
         'XColor', [.1 .1 .1],  'YColor', [.1 .1 .1])              % 坐标轴颜色
set(gca, 'Xlim' , [-4 5],...
         'YTick', 1:13,...
         'Ylim' , [0.5 13.5], ... 
         'Yticklabel',{'A','B','C','D','E','F','G','H','I','J','K','L','M'})
% 基线调整
BL = get(GO2,'BaseLine');
BL{1,1}.LineStyle = 'none';
YL = get(gca,'ylim');
plot([0 0],YL,'color', 'k','linewidth',1,'LineStyle','--')
% Legend
hLegend = legend(GO1, ...
                 'Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', ...
                 'Location', 'northwest','Orientation','vertical');
hLegend.ItemTokenSize = [10 10];
% 字体和字号
set(gca, 'FontName', 'Arial', 'FontSize', 10)
set([hLegend,hXLabel,hYLabel], 'FontSize', 11, 'FontName', 'Arial')
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