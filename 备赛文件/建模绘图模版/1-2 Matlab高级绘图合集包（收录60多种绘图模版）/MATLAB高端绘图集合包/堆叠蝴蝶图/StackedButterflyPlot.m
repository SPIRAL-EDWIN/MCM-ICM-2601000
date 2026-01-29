clc;
clear;
close all;
%堆叠蝴蝶图

%% 数据准备
% 读取数据
load data.mat
% 初始化绘图参数
x1 = X1;
x2 = X2;
Label={'Sample1','Sample2','Sample3','Sample4','Sample5','Sample6','Sample7','Sample8'};

%% 颜色定义
C = colormap(nclCM(486));%color包里选颜色
C1 = C(1,1:3);
C2 = C(2,1:3);
C3 = C(3,1:3);
C4 = C(4,1:3);
C5 = C(10,1:3);
C6 = C(9,1:3);
C7 = C(8,1:3);
C8 = C(7,1:3);

%% 图片尺寸设置（单位：厘米）
close all;
figureUnits = 'centimeters';
figureWidth = 16;
figureHeight = 12;

%% 窗口设置
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);

%% 堆叠蝴蝶图绘制
[ax1,ax2,b1,b2] = Butterfly(figureHandle,X1,X2,Label,'stacked');

%% 细节优化
% 左翼优化
% 赋色
b1(1).FaceColor = C1;
b1(2).FaceColor = C2;
b1(3).FaceColor = C3;
b1(4).FaceColor = C4;
% 坐标区调整
set(ax1, 'Box','off',...
         'LineWidth',1,...
         'TickLength',[0 0],...
         'XGrid','on','YGrid','off',...
         'XDir','reverse',...
         'YDir','reverse',...
         'YAxisLocation','right',...
         'YTick',[])
set(ax1, 'ylim', [0.5 8.5])
ax1.XRuler.Axle.LineStyle = 'none'; 
% 标签及Legend1设置
hLegend1 = legend(ax1, ...
                 'F1','F2','F3','F4', ...
                 'Location', 'northoutside',...
                 'Orientation','horizontal');
hLegend1.ItemTokenSize = [10 10];
hLegend1.Box = 'off';
% 字体字号
set([ax1,hLegend1], 'FontName', 'Arial', 'FontSize', 9)

% 右翼优化
% 赋色
b2(1).FaceColor = C5;
b2(2).FaceColor = C6;
b2(3).FaceColor = C7;
b2(4).FaceColor = C8;
set(ax2, 'Box','off',...
         'LineWidth',1,...
         'TickLength',[0 0],...
         'XGrid','on','YGrid','off',...
         'XDir','normal',...
         'YDir','reverse',...
         'YAxisLocation','left',...
         'YTick',[])
set(ax2, 'ylim', [0.5 8.5])
ax2.XRuler.Axle.LineStyle = 'none';  
% 标签及Legend2设置
hLegend2 = legend(ax2, ...
                 'F5','F6','F7','F8', ...
                 'Location', 'northoutside',...
                 'Orientation','horizontal');
hLegend2.ItemTokenSize = [10 10];
hLegend2.Box = 'off';
% 字体字号
set([ax2,hLegend2], 'FontName', 'Arial', 'FontSize', 9)

% 背景颜色
set(gcf,'Color',[1 1 1])

%% 图片输出
figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[0 0 figW figH]);
fileout = 'test';
print(figureHandle,[fileout,'.png'],'-r300','-dpng');