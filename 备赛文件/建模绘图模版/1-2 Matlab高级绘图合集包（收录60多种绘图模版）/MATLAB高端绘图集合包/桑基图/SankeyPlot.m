clc;
clear;
close all;
%桑基图

%% 数据准备
% 读取数据
tbl = readtable('data.xlsx');
% 转换分类数组
tbl.Age = categorical(tbl.Age);
tbl.Treatment = categorical(tbl.Treatment);
tbl.Surgery = categorical(tbl.Surgery);
tbl.Result = categorical(tbl.Result);
% 初始化绘图参数
data = tbl;

%% 颜色定义
N = 16; % N = 层数 * max(每层子单元数量), 4*max([4 3 2 3])
map = colormap(nclCM(272));%color包里选颜色
map = map(1:16,:);

%% 桑基图绘制
close all;
% 参数设置
options.figureWidth = 15;         % 图片宽度(cm)
options.figureHeight = 14;        % 图片高度(cm)
options.color_map = map;          % 配色
options.flow_transparency = 0.7;  % 流动路径线透明度
options.show_perc = false;        % 是否显示节点块占比
options.text_color = [0 0 0];     % 文字颜色
options.show_layer_labels = true; % 是否坐标底部显示层名称
options.show_cat_labels = true;   % 是否在节点块显示分类名称
options.show_legend = false;      % 是否显示图例
% 绘图
SankeyChart(data,options);

%% 图片输出
exportgraphics(gcf,'test.png','Resolution',300)