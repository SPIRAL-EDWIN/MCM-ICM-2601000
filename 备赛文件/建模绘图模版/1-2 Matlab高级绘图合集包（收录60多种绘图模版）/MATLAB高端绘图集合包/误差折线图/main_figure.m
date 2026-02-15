%% 画误差折线图
clc;
clear;
data = [0.6,0.8,0.4,0.5,0.55,0.6,0.8,0.9,0.93,0.82,0.75];
X = 1:11;
Y = data;
err_std = data/9;
figure('color',[1 1 1]);
color = [0.294117647058824,0,0.509803921568627];
e = errorbar(X,Y,err_std,'-s','MarkerSize',12,...
    'MarkerEdgeColor',color,'MarkerFaceColor',color);
e.Color = color;
e.CapSize = 10;
e.Marker = '*';
e.LineWidth = 1.2;
xlabel('横坐标','Fontsize',13,'Fontname','楷体','Fontweight','bold');
ylabel('实验结果','Fontsize',13,'Fontname','楷体','Fontweight','bold');
title('误差折线图','Fontsize',13,'Fontname','楷体','Fontweight','bold');
set(gca,'Linewidth',1.1);
set(gca,'TickDir','in','TickLength',[0.007,0.007]);
X={'','Tink','Malx','KNC','BXY','CNN','ALL','TXTP','WNET','SUY','MAG','CTI'};
set(gca,'Xlim',[0,12],'Xtick',[0:1:12],'Xticklabel',X);
grid on;
