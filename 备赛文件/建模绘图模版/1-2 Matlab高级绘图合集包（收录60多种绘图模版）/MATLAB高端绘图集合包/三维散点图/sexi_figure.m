%% 生成色系代码
clc;
clear;
color1 = [67 5 87]./255; % 第一种颜色
color2 = [248 230 32]./255;% 第二种颜色
n=60;% 60个颜色数据点
R =(linspace(color1(1),color2(1),n))';
G =(linspace(color1(2),color2(2),n))';
B =(linspace(color1(3),color2(3),n))';
mycolor = [R,G,B];
colormap(mycolor);
colorbar();
