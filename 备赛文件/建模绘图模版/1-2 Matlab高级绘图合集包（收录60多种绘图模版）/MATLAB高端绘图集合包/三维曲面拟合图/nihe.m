%% matlab绘制三维曲面图
%获取颜色柱数据与实验数据
clc;
clear;
load mycolor1.mat
load mycolor2.mat
[data,str,all] = xlsread('XYZ2.xlsx');%三维数据文件 把你的数据复制到XYZ文件替换数据就行
x=data(:,1);
y=data(:,2);
z=data(:,3);
c=data(:,4);%用来表示颜色特征
method_fit = 'cubic';
%插值生成网格化数据
[XX,YY,Z]=griddata(x,y,z,linspace(min(x),max(x))',linspace(min(y),max(y)),method_fit);
[X,Y,C]=griddata(x,y,c,linspace(min(x),max(x))',linspace(min(y),max(y)),method_fit);
figure('color',[1 1 1]);
mesh(XX,YY,Z,C);%三维曲面
xlabel('X轴');
ylabel('Y轴');
zlabel('Z轴');
title('三维数值拟合曲线');
colormap(autumn);
colorbar;
h = colorbar;%右侧颜色栏
set(get(h,'label'),'string','z值');%给右侧颜色栏命名
grid on;
%用来调整三维视角角度
view(157,11);
shading interp
