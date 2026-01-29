%% 绘制三维气泡图
clc;
clear;
close all;
%% 随机生成数据
X = 1:64;
Y = randperm(64);
Z = randperm(64);
ZF = randperm(64)*30;
%% 导入所有颜色矩阵
load mycolor.mat;
%% 开始绘制散点图
figure('color',[1 1 1]);
ax1 = subplot(221);
X = 1:64;
Y = randperm(64);
Z = randperm(64);
scatter3(X,Y,Z,ZF,mycolor1,'.');
colormap(ax1,mycolor1);
set(gca,'Linewidth',0.9);
xlabel('X','Fontname','微软雅黑');
ylabel('Y','Fontname','微软雅黑');
zlabel('Z','Fontname','微软雅黑');
title('三维散点','Fontname','微软雅黑');

%% 绘制子图2
X = 1:64;
Y = randperm(64);
Z = randperm(64);
ax2 = subplot(222);
scatter3(X,Y,Z,ZF,mycolor1,'.');
colormap(ax2,mycolor1);
set(gca,'Linewidth',0.9);
xlabel('X','Fontname','微软雅黑');
ylabel('Y','Fontname','微软雅黑');
zlabel('Z','Fontname','微软雅黑');
title('三维散点','Fontname','微软雅黑');
%% 绘制子图3
ax3 = subplot(223);
X = 1:64;
Y = randperm(64);
Z = randperm(64);
scatter3(X,Y,Z,ZF,mycolor1,'.');
colormap(ax3,mycolor1);
set(gca,'Linewidth',0.9);
xlabel('X','Fontname','微软雅黑');
ylabel('Y','Fontname','微软雅黑');
zlabel('Z','Fontname','微软雅黑');
title('三维散点','Fontname','微软雅黑');
%% 绘制子图4
X = 1:64;
Y = randperm(64);
Z = randperm(64);
ax4 = subplot(224);
scatter3(X,Y,Z,ZF,mycolor1,'.');
colormap(ax4,mycolor1);
set(gca,'Linewidth',0.9);
xlabel('X','Fontname','微软雅黑');
ylabel('Y','Fontname','微软雅黑');
zlabel('Z','Fontname','微软雅黑');
title('三维散点','Fontname','微软雅黑');
%% 设置颜色柱的位置信息\保存图片
colorbar('position',[0.95,0.11,0.015,0.8]);
print(gcf,'成图.jpg','-djpeg','-r900');
