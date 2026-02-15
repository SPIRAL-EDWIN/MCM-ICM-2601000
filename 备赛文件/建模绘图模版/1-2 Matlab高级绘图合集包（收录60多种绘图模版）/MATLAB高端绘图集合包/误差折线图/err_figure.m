clc;clear all;
x=1:11;
y=2:2:22;
err=0.3*ones(1,length(x));
errorbar(x,y,err)

xlabel('横坐标','Fontsize',13,'Fontname','楷体','Fontweight','bold');
ylabel('实验结果','Fontsize',13,'Fontname','楷体','Fontweight','bold');
title('误差折线图','Fontsize',13,'Fontname','楷体','Fontweight','bold');
set(gca,'Linewidth',1.1);