%% 进阶绘图第14期-对数坐标图

clc;
clear;
close all;



X = 1:30;
Y = X.^3+X.^2;
Y2 = 10*X.^2;

figure('color',[1 1 1]);

subplot(2,2,1);
plot(X,Y,'Linewidth',1.5);
hold on;
plot(X,Y2,'Linewidth',1.5);
set(gca,'Linewidth',1.5);
grid on;
title('原直角坐标');

subplot(2,2,2);
plot(X,Y,'Linewidth',1.5);
hold on;
plot(X,Y2,'Linewidth',1.5);
set(gca,'XScale','log');
set(gca,'Linewidth',1.5);
grid on;
title('X对数');

subplot(2,2,3);
plot(X,Y,'Linewidth',1.5);
hold on;
plot(X,Y2,'Linewidth',1.5);
set(gca,'YScale','log');
set(gca,'Linewidth',1.5);
grid on;
title('Y对数');

subplot(2,2,4);

plot(X,Y,'Linewidth',1.5);
hold on;
plot(X,Y2,'Linewidth',1.5);
set(gca,'XScale','log');
set(gca,'YScale','log');

set(gca,'Linewidth',1.5);
grid on;
title('X-Y对数');



