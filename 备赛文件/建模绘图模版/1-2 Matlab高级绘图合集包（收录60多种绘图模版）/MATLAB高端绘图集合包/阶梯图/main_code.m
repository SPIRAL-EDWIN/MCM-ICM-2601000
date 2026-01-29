%% Matlab绘制阶梯图
clc;
clear;
close all;
%第一个例子  使用正余弦函数作为案例
X = 0:0.1:2*pi;
Y1 = sin(X);
Y2 = cos(X);
Y3 = 0.2*cos(X)+0.2*sin(X);
[x1,y1] = stairs(Y1);
[x2,y2] = stairs(Y2);
[x3,y3] = stairs(Y3);
figure('color',[1 1 1]);
stairs(x1,y1,'-.','color','m','Linewidth',1.5);
hold on;
stairs(x2,y2,'-.','color','r','Linewidth',1.5);
hold on;
stairs(x3,y3,'-.','color','g','Linewidth',1.5);
grid on;
xlabel('time','Fontname','Times new Roman','Fontweight','bold');
ylabel('Amplitude','Fontname','Times new Roman','Fontweight','bold');

set(gca,'Linewidth',1);
set(gca,'TickDir', 'in', 'TickLength', [.005 .005]);    
set(gca,'Xlim',[0 64]);





%% 第二个例子
clc;
x_new = 1:20;
Y = [1,2,3,5,2,4,7,1,3,2,7,2,7,4,7.5,5,2,1,4,5];
rand_num = randperm(20);
Y = Y(rand_num);
figure('color',[1 1 1]);
stairs(x_new,Y,'Linewidth',1.5,'color','r');
grid on;
xlabel('time','Fontname','Times new Roman','Fontweight','bold');
ylabel('Amplitude','Fontname','Times new Roman','Fontweight','bold');
set(gca,'Linewidth',1);
set(gca,'TickDir', 'in', 'TickLength', [.005 .005]);    


