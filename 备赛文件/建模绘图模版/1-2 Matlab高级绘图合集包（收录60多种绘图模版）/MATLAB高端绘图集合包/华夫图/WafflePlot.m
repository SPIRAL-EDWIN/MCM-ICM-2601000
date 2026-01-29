clc;
clear;
close all;
%华夫图
%% 数据准备
% 读取数据
A = readcell('dataset.xlsx');
data = A(2:8,2:6);

%% 颜色定义
C = colormap(nclCM(393));%color包里选颜色
C = C(1:9,1:3);

%% 图片尺寸设置（单位：厘米）
close all;
figureUnits = 'centimeters';
figureWidth = 13;
figureHeight = 15;

%% 窗口设置
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);

%% 华夫图绘制
[h_surf,u1,u2,u3] = waffle(data,2);
hXLabel = xlabel('Clinical');
hYLabel = ylabel('Patient');

%% 细节优化
% u1标签排序
idx = [2 1 3 5 4 6 7 8 9];
% u1标签默认顺序
% idx = 1:length(u1);
% 赋色
map = C(idx,1:3);
colormap(map)
% 坐标轴美化
axis tight equal
set(gca, 'xaxislocation','bottom',...
         'yaxislocation','left',...
         'YDir','reverse',...
         'xtick',1:5,...
         'ytick',1:7,...
         'Xticklabel',{'Grade Group' 'PSA' 'ICC' 'IDC' 'Stage'},...
         'Yticklabel',{'ICC1' 'ICC2' 'ICC3' 'ICC4' 'ICC5' 'ICC6' 'ICC7'})
% Legend
hLegend = legend(h_surf(u2(idx)),u1(idx),...
                'FontWeight','normal',...
                'FontSize',9,...
                'Box','off',...
                'Location', 'eastoutside',...
                'Orientation','vertical');
hLegend.ItemTokenSize = [10 10];
% 字体和字号
set(gca, 'FontSize', 9, 'FontName', 'Arail')
set([hXLabel, hYLabel], 'FontSize', 11, 'FontName', 'Arail')
% 背景颜色
set(gcf,'Color',[1 1 1])
% 删除白边
set(gca,'LooseInset',get(gca,'TightInset'))

%% 图片输出
figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[0 0 figW figH]);
fileout = 'test0';
print(figureHandle,[fileout,'.png'],'-r300','-dpng');