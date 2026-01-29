%% 数据准备
clc;
clear;
close all;
x = 1:5;
% dataset为4*2的矩阵，4代表4大组、2代表每组2个柱子
dataset = [0.241,0.33;0.219,0.254;0.238,0.262;0.19,0.329];
% 误差矩阵大小也是5*3
Mean = dataset; % 下方长度
Std = dataset/9; % 上方长度
%% 颜色定义 自己可以修改喜欢的颜色
C1 = [193 192 250]./255;
C2 =[254 253 163]./255;
% C1 = [0 191 196]./255;
% C2 =[248 118 109]./255;
%% 绘图
% 绘制初始柱状图
figure('color',[1 1 1]);
GO = bar(dataset,1,'EdgeColor','k');
GO(1).FaceColor = C1;
GO(2).FaceColor = C2;
% 添加误差棒
hold on;
errorbar([1+0.14  2+0.14  3+0.14  4+0.14 ],Mean(:,2),Std(:,2),'k','Linestyle','None','LineWidth', 1.2);
hold on;
errorbar([0.775+0.09 1.775+0.09  2.775+0.09  3.775+0.09 ],Mean(:,1),Std(:,1),'k','Linestyle','None','LineWidth', 1.2);
%% 参数调整
hold on;
ylabel('Mean Accuracy','Fontname','Times New Roman','FontSize',12,'FontWeight','bold');
% title('柱状图带误差棒','Fontname','微软雅黑');
% 坐标区调整
set(gca,'box','off');
% 重置box
set(gca,'XGrid', 'off', 'YGrid', 'on');
set(gca,'TickDir', 'out', 'TickLength', [.01 .01], 'XMinorTick', 'off', 'YMinorTick', 'off');          
% 设置X轴属性
set(gca,'Xticklabel',{'Mascu' 'Agency' 'Feme' 'Comuhjh'});
% 设置Y轴属性
set(gca,'YTick', 0:0.1:1,'Ylim',[0 0.5]);
set(gca,'Linewidth',1.1);
set(gca,'XGrid','off','YGrid','off');
set(gca,'Fontname','Times New Roman','FontSize',12,'FontWeight','bold');
plot([0.5,4.3],[0.2,0.2],'LineStyle','--','Color','k','Linewidth',1);
% Legend 设置   
hLegend = legend([GO(1),GO(2)],'Odd Numbers', 'Even Numbers','Location', 'northeast');
set(hLegend,'box','off');
set(hLegend,'FontName','Times New Roman','FontSize',12,'FontWeight','bold');
