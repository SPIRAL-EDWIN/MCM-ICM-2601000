clc;
clear;
close all;
%双特征渲染三维散点图

%% 数据准备
% 读取数据
data = load('data.txt');
% 初始化绘图参数
idx1 = find(data(:,4)==1);
idx2 = find(data(:,4)==0);
% 散点数据1
x1 = data(idx1,1);
y1 = data(idx1,2);
z1 = data(idx1,3);
f1 = data(idx1,3);
% 散点数据2
x2 = data(idx2,1);
y2 = data(idx2,2);
z2 = data(idx2,3);
f2 = data(idx2,3);

%% 颜色定义
map1 = colormap(nclCM(484));%color包里选颜色
map2 = colormap(nclCM(487));%color包里选颜色

%% 图片尺寸设置（单位：厘米）
close all;
figureUnits = 'centimeters';
figureWidth = 10;
figureHeight = 11;

%% 窗口设置
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);

%% 双特征渲染三维散点图
ax = gca;
% 特征渲染三维散点图1绘制
scatter3(x1, y1, z1, 5, f1, 'filled')
caxis([min(f1(:)) max(f1(:))]);
colormap(map1)
freezeColors; 
hold on
% 特征渲染三维散点图2绘制
scatter3(x2, y2, z2, 5, f2, 'filled')
caxis([min(f2(:)) max(f2(:))]);
colormap(map2)
freezeColors; 
% 标题、标签、视角
hTitle = title('DoubleFeatureScatter Plot');
hXLabel = xlabel('x');
hYLabel = ylabel('y');
hZLabel = zlabel('z');
view(-37.5,30)

%% 细节优化
% 添加颜色条
colorbar_k2('right',f1,map1,f2,map2)
% 坐标区调整
axes(ax)
axis equal
set(gca, 'Box', 'off', ...                                                          % 边框
         'LineWidth', 1, 'GridLineStyle', '-',...                                   % 坐标轴线宽
         'XGrid', 'on', 'YGrid', 'on', 'ZGrid', 'on',...                            % 网格
         'TickDir', 'out', 'TickLength', [.01 .01], ...                             % 刻度
         'XColor', [.1 .1 .1],  'YColor', [.1 .1 .1],'ZColor', [.1 .1 .1])          % 坐标轴颜色
% 字体和字号
set(gca, 'FontName', 'Arial', 'FontSize', 11)
set([hXLabel,hYLabel,hZLabel], 'FontName',  'Arial', 'FontSize', 11)
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
