%% 二维散点图教程绘制
%% 数据准备
clc
clear;
%生成500个0-1之间的数据
x = 100:600;
a= 0;
b=1;
R = a + (b-a).*rand(500,1);
m =R;
%% 开始绘图

figure('color',[1 1 1]);
s = scatter(1:length(m),m,'filled');
s.LineWidth = 0.6;
kk = rand(1,500)';
s.AlphaData = kk;
s.MarkerFaceAlpha = 'flat';
s.MarkerEdgeColor = 'k';
c1 = [246 214 3]/255;
c2 = [9 12 19]/255;
c3 = [254 114 141]/255;
c4 = [128 159 186]/255;
s.MarkerFaceColor = c1;
s.MarkerFaceColor = c2;
s.MarkerFaceColor = c3;
s.MarkerFaceColor = c4;
set(gca,'Xlim',[-50 550]);
set(gca,'Ylim',[0,1.1]);
set(gca,'Linewidth',1);
grid on;


