clc;
clear;
close all;
%双网格曲面图

%% 数据准备
% 读取数据
load data.mat
% 预处理/初始化绘图参数
% 网格1抽稀
x1 = X(1:5:end,1:5:end);
y1 = Y(1:5:end,1:5:end);
z1 = Z1(1:5:end,1:5:end);
% 网格2抽稀
x2 = X(1:5:end,1:5:end);
y2 = Y(1:5:end,1:5:end);
z2 = Z2(1:5:end,1:5:end);

%% 颜色定义
map1 = colormap(nclCM(484));%color包里选颜色
map2 = colormap(nclCM(487));%color包里选颜色

%% 图片尺寸设置（单位：厘米）
close all;
figureUnits = 'centimeters';
figureWidth = 15;
figureHeight = 13;

%% 窗口设置
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);

%% 双网格曲面图绘制
ax = gca;
% 网格曲面1绘制
s = mesh(x1,y1,z1,'LineWidth',1);
caxis([min(z1(:)) max(z1(:))]);
colormap(map1)
freezeColors; 
hold on
% 网格曲面2绘制
s2 = mesh(x2,y2,z2,'LineWidth',1);
caxis([min(z2(:)) max(z2(:))]);
colormap(map2)
freezeColors; 
% 标题、标签、视角
hTitle = title('DoubleMesh Plot');
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