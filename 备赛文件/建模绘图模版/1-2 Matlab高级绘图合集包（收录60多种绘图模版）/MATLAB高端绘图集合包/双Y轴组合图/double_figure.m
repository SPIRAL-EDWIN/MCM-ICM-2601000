%% 柱状图与误差折线图数据准备
clc;
clear;
X = 1:14;
% 数据准备
data_zhu = [0,1000,200,150,160,4500,1800,6200,10000,0,4000,0,400,3000];% 柱状图数据
data_zhe = [0.32,0.27,0.31,0.29,0.68,0.65,0.29,0.42,0.6,0.45,0.51,0.32,0.31,0.51];
err_data =     [0.1,  0.13,0.11,0.14,0.2, 0.3,   0.23, 0.15,0.2,0.12, 0.15,0.1, 0.1, 0.2];
err_data_max = [0.15, 0.2, 0.18,0.21,0.25, 0.37, 0.29, 0.2, 0.27,0.18,0.19,0.2, 0.22, 0.26];
err_data_min = [0.06, 0.1,0.09, 0.11,0.14, 0.26, 0.2,  0.11,0.12,0.09,0.11,0.08, 0.07, 0.14];
err_data_up = err_data_max-err_data;
err_data_down = err_data-err_data_min;
%% 开始绘图（左轴为堆叠柱状图）
hFig = figure('color',[1 1 1]);
set(hFig, 'Position', [300 100 1000 500]);
yyaxis left;
set(gca,'ycolor',[0 0 0]);
%提取柱状图的颜色
color_zhu = [176,190,175]/255;
bar_figure = bar(X,data_zhu,0.6,'FaceAlpha',0.2);%用来修改透明度
set(bar_figure,'Facecolor',color_zhu);
hold on;
set(gca,'Ylim',[0 12000]);
ylabel('Number of need','Fontsize',13);
%% 开始绘制误差图
yyaxis right;
Y = data_zhe;
err_std = err_data;
set(gca,'ycolor',[0 0 0]);
e = errorbar(X,Y,err_data_up,err_data_down,'o','MarkerSize',6,'MarkerEdgeColor','r','MarkerFaceColor','r');
e.Color = 'k';%修改误差线的颜色
e.CapSize = 6; %修改误差线的宽度
e.LineWidth = 1.1;
ylabel('R_{T}_{C}','Fontsize',13);
%% 美化图像
set(gca,'XGrid', 'off', 'YGrid', 'off','TickDir', 'in', 'TickLength', [.003 .003]);          
% 对X轴显示范围与横坐标显示设置
XX2 = {' ' ,'sadhiusaduias','sadsafasag','egregreg','fgdgfdg3','wrttertrete','werewrtg','dfdsfdsvfdg','dr34r43rf',...
    'werewtret','454fgfgfg','dsd3rfrf','rt56yhgrtfb','fret54rgtvfd','fwr33defdsf',' '};
set(gca,'Xlim',[0 15],'Xtick', [0:1:15],'Xticklabel',XX2,'XTickLabelRotation',-45);
h1 = legend('Number','RC' );
set(h1,'box','off','Location','Northwest');
set(gca,'Linewidth',1.2);
set(gca,'looseInset',[0 0 0 0]) %去掉图窗的多余白边