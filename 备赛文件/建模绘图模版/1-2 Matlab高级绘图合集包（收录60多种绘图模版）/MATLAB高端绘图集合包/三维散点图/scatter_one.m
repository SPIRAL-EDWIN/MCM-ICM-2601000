%% 绘制三维散点（气泡）图
clc;
clear;
close all;
%% 随机生成X,Y,Z数据
X = 1:64;
Y = randperm(64);
Z = randperm(64);
ZF = randperm(64)*30;% 散点特征：散点值大小/颜色深浅
%% 导入所有颜色矩阵
% 颜色文件有8中颜色map分别为 mycolor1---mycolor8
load mycolor.mat;
%% 开始绘制三维散点图
mycolor_value= mycolor1;
figure('color',[1 1 1]);
scatter3(X,Y,Z,ZF,mycolor_value,'.');
colormap(mycolor_value);
colorbar()
%% 完善图例与坐标区
xlabel('X','Fontname','微软雅黑');
ylabel('Y','Fontname','微软雅黑');
zlabel('Z','Fontname','微软雅黑');
title('三维散点','Fontname','微软雅黑');
set(gca,'Box','on');
ax = gca;
ax.BoxStyle = 'full';
set(gca,'Xgrid','off','Ygrid','off','Zgrid','off');
set(gca,'Linewidth',1.2);
%% 保存图片
print(gcf,'图1','-dpng','-r600');





