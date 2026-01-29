clc;
clear;
close all;
%三角气泡热图
%% 数据准备
% 读取数据
load data.mat
% 数据矩阵
Z = data;
% 标签
xlb = {'Carb','Wt','Hp','Cyl','Disp','Qsec','Vs','Mpg','Drat','Gear'};
ylb = {'Carb','Wt','Hp','Cyl','Disp','Qsec','Vs','Mpg','Drat','Gear'};

%% 颜色定义
map = colormap(nclCM(486));%color包里选颜色

%% 图片尺寸设置（单位：厘米）
close all;
figureUnits = 'centimeters';
figureWidth = 15;
figureHeight = 15;

%% 窗口设置
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);

%% 气泡热图绘制
% tribubbleheatmap(Z,30,600,xlb,ylb,'trid') % 下三角
tribubbleheatmap(Z,30,600,xlb,ylb,'triu') % 上三角

%% 细节优化
% 赋色
colormap(map)
% 字体和字号
set(gca, 'FontName', 'Arial', 'FontSize', 10)
% 背景颜色
set(gcf,'Color',[1 1 1])

%% 图片输出
exportgraphics(gca,'test.png','Resolution',300)