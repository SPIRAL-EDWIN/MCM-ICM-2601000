clc;
clear;
close all;
%双三角热图

%% 数据准备
% 读取数据
load data.mat
% 初始化绘图参(此处仅使用一种数据演示)
Z1 = data; 
Z2 = data;
% 标签
xlb = {'Carb','Wt','Hp','Cyl','Disp','Qsec','Vs','Mpg','Drat','Gear'};
ylb = {'Carb','Wt','Hp','Cyl','Disp','Qsec','Vs','Mpg','Drat','Gear'};

%% 颜色定义
map1 = colormap(nclCM(486));%color包里选颜色
% map1 = flipud(map1);
map2 = colormap(nclCM(48));%color包里选颜色
% map2 = flipud(map2);

%% 图片尺寸设置（单位：厘米）
close all;
figureUnits = 'centimeters';
figureWidth = 16;
figureHeight = 15;

%% 窗口设置
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);

%% 双三角热图绘制
dtriheatmap(Z1,Z2,map1,map2,xlb,ylb,'right')

%% 细节优化
% 背景颜色
set(gcf,'Color',[1 1 1])

%% 图片输出
exportgraphics(gcf,'test.png','Resolution',300)