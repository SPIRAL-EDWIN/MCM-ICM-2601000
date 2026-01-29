%% 数据准备
data = [1.5,2,0.8;...
        5,1.2,1.2;...
        10,3,1;...
        5,1.2,1.2;...
        1.5,2,0.8];
error_data = [1,1,2;...
             2,0.3,1.8;...
              5,1,1;...
              2,0.3,1.8;...
              1,1,2];
%% 开始绘图
figure('color',[1 1 1]);
bar_figure = bar(data,0.6,'stacked');
hold on;
errorbar([],[cumsum(data')]',[],error_data,'LineStyle','none',...
    'Color',[0 0 0],'Linewidth',1.1,'CapSize',17);

%% 设置颜色
C1 = [244 206 125]./255;
C2= [123 160 176]./255;
color1 = C1;
color2  =C2;
color3 = C1;
Color = [color1;color2 ;color3];
for i = 1:size(Color,1)
    bar_figure(i).FaceColor = Color(i,:);
end
%% 设置坐标区参数
xlabel_str = {'value1','value2','value3','value4','value5'};
ylabel('实验值','Fontsize',13,'Fontname','楷体');
title('堆叠柱状图','Fontsize',13,'Fontname','楷体');
set(gca,'YTick',0:2:16,'Ylim',[0 16]);
set(gca,'Xticklabel',xlabel_str);
set(gca,'box','off');
h = legend('zre','mfe','dfj');
set(h,'box','off');

set(gca,'Linewidth',1);


