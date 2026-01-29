%% 顶刊论文图表复现第1期——复现热图
%% 生成对角为1的随机系数矩阵
clc;
clear;
close all;
%X为0-1矩阵 
X = rand(8);
for i =1:8
    for j = 1:8
        if X(i,j)>0.4&&X(i,j)<0.6
            X(i,j) = X(i,j)-0.4;
        elseif X(i,j)>0.2&&X(i,j)<0.4
             X(i,j) = X(i,j)-0.2; 
        end
    end
end
%% 定义X轴名字 定义Y轴名字

load colormap2;
b = bar3(X,1);
for i = 1:length(b)
    zdata = b(i).ZData;
    b(i).CData = zdata;
    b(i).FaceColor = 'interp';
end
colormap(mycolor);
view(0,90);
c = colorbar;
axis tight;
set(gcf,'color',[1 1 1]);
hold on;
plot3([0.5 8.5 ],[0.5 0.5],[0 0],'k','Linewidth',3);
hold on;
%% 调整坐标区参数
Xlabel_name = {'NOTCH1','NOTCH2','NOTCH3','NOTCH4','DLL1','DLL4','JAG1','JAG2'};
Ylabel_name = {'SMC','EC-ven','EC-art','PC','EC-cap','Mesen','EC','EC-In'};
set(gca,'ygrid','on','color',[250 227 225]/255,'Gridalpha',1);
set(gca,'xgrid','on','color',[250 227 225]/255,'Gridalpha',0.4);
set(gca,'zgrid','off');
set(gca,'Box','off');
set(gca,'TickDir','out','Ticklength',[0.01,0.01]);
set(gca,'Linewidth',3);
set(gca,'Xticklabel',Xlabel_name,'XTickLabelRotation',90,'fontsize',17);
set(gca,'Yticklabel',Ylabel_name,'YTickLabelRotation',0,'fontsize',17);
title('cell-states','Fontsize',25,'position',[-2,4.45],'Rotation',90);
%% 设置colorbar的刻度与标签显示参数
caxis([0,1]);
set(c,'tickdir','out');
set(c,'Ytick',0.0:0.2:1.2);
set(c,'YtickLabel',{'0.0','0.2','0.4','0.6','0.8','1.0'});
set(c,'Linewidth',1.5);
set(c,'FontSize',20);
set(c,'Box','on');
%% 修改白边
% set(gca, 'LooseInset', [0.1,0 0,0]);
% set (gcf,'Position',[0,0,590,520]);
% axis normal;
