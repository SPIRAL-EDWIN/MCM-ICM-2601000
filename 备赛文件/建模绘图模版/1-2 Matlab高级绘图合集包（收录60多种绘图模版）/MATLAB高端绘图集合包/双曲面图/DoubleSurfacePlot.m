clc;
clear;
close all;
%双曲面图

%% 数据准备
% 读取数据
load data.mat
% 初始化绘图参数
% 曲面1
x1 = X;
y1 = Y;
z1 = Z1;
% 曲面2
x2 = X;
y2 = Y;
z2 = Z2;

%% 颜色定义
map1 = colormap(nclCM(196));%color包里选颜色
% map1 = flipud(map1);
map2 = colormap(nclCM(303));%color包里选颜色

%% 图片尺寸设置（单位：厘米）
close all;
figureUnits = 'centimeters';
figureWidth = 15;
figureHeight = 13;

%% 窗口设置
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);

%% 双曲面图绘制
ax = gca;
% 曲面1绘制
s = surf(X,Y,Z1,'EdgeColor','none');
caxis([min(Z1(:)) max(Z1(:))]);
colormap(map1)
freezeColors; 
hold on
% 曲面2绘制
s2 = surf(X,Y,Z2,'EdgeColor','none');
caxis([min(Z2(:)) max(Z2(:))]);
colormap(map2)
freezeColors; 
% 标题、标签、视角
hTitle = title('DoubleSurface Plot');
hXLabel = xlabel('x');
hYLabel = ylabel('y');
hZLabel = zlabel('z');
view(-35,30)

%% 细节优化
% 添加颜色条
colorbar_k2('right',Z1,map1,Z2,map2)
% 坐标区调整
axes(ax)
axis tight
set(gca, 'Box', 'off', ...                                                          % 边框
         'LineWidth', 1, 'GridLineStyle', '-',...                                   % 坐标轴线宽
         'XGrid', 'on', 'YGrid', 'on', 'ZGrid', 'on',...                            % 网格
         'TickDir', 'out', 'TickLength', [.01 .01], ...                             % 刻度
         'XColor', [.1 .1 .1],  'YColor', [.1 .1 .1],'ZColor', [.1 .1 .1],...       % 坐标轴颜色
         'zlim',[0 700])
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