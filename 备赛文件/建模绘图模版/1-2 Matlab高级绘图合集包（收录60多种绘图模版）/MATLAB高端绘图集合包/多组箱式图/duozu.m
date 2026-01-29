%% 导入数据
clc;
clear;
[all,abc,str] = xlsread('data2.xlsx');

%% edge color
data1=all(:,1:5);
data2=all(:,7:11);
edgecolor1=[0,0,0]; % black color
edgecolor2=[0,0,0]; % black color
fillcolor1=[206, 85, 30]/255; % fillcolors = rand(24, 3);
fillcolor2=[46, 114, 188]/255;
fillcolors=[repmat(fillcolor1,5,1);repmat(fillcolor2,5,1)];
position_1 = [0.8:2:8.8];  % define position for first group boxplot
position_2 = [1.2:2:9.2];  % define position for second group boxplot 
box_1 = boxplot(data1,'positions',position_1,'colors',edgecolor1,'width',0.2,'symbol','r+','outliersize',5);
hold on;
box_2 = boxplot(data2,'positions',position_2,'colors',edgecolor2,'width',0.2,'symbol','r+','outliersize',5);
boxobj = findobj(gca,'Tag','Box');
for j=1:length(boxobj)
    patch(get(boxobj(j),'XData'),get(boxobj(j),'YData'),fillcolors(j,:),'FaceAlpha',0.5);
end
set(gca,'XTick', [1  3  5  7 9],'Xlim',[0 10]);
set(gca,'YTick',1:1:7.5,...
    'Ylim',[1 7.5]);

boxchi = get(gca, 'Children');

xticks([1 3 5 7 9]);
xticklabels({'Nan','MT','Lowdo','Midean','Highest'});
hold on;
x13 = 4.8;
y13 = 5;
x15 = 8.8;
y15 = 6.5;
plot([x13,x13],[y13,7],'b','Linewidth',0.9);
hold on;
plot([x13,x15],[7,7],'b','Linewidth',0.9);
hold on;
plot([x15,x15],[y15,7],'b','Linewidth',0.9);
text((x15+x13)/2,7.1,'*','fontsize',20);

x23 = 5.2;
y23 = 6;
x25 = 9.2;
y25 = 6;
plot([x23,x23],[y23,6.2],'r','Linewidth',0.9);
hold on;
plot([x23,x25],[6.2,6.2],'r','Linewidth',0.9);
hold on;
plot([x25,x25],[y25,6.2],'r','Linewidth',0.9);
hold on;
text((x25+x23)/2,6.3,'*','fontsize',20);


%% 设置边框
% 坐标轴美化
set(gca, 'Box', 'on', ...                                       % 边框
        'LineWidth', 1,...                                      % 线宽
        'XGrid', 'off', 'YGrid', 'off', ...                     % 网格
        'TickDir', 'in', 'TickLength', [.015 .015], ...         % 刻度
        'XMinorTick', 'off', 'YMinorTick', 'off')
% 字体和字号
set(gca, 'FontSize', 10)
% 背景颜色
set(gcf,'Color',[1 1 1])
% 设置图例
xlabel('花键');
ylabel('Kenan');
title('多组别箱式图');
%图例设置
% legend([boxchi(1),boxchi(6)], ["Fuzzing", "Uniform"] );
%% 保存图片
savefig(gcf,'boxplot_position_fillcolor.fig');
print(gcf,'result','-dpng','-r600');