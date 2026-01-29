%%  绘制三维折线图并且进行填充设置
clc;
clear;
% 首先说一下数据格式的组成
%  X：    数据X是自变量  就是你的时间序列的自变量时间 一个行向量
%  Y:     数据Y也是一个自变量 有几个线就有几行，列向量是数据点个数 假如一根线 第一行全是1 两根线就是第一行全是1 第二行全是2
% 三行的话数据第三行就全是3,  这个例子是5根线
%  Z:     Z数据就是你的时间序列对应的值，行数也是线的数量。这个例子是5根线 
num_line = 7;
X = 0:0.01:pi;
y = ones(1,size(X,2));
Y = [];
for i = 1:num_line
    Y(i,:) = y.*i;
end
Z = [];
for i = 1:num_line
    Z(i,:) = i.*sin(X);
end

% 随机设置颜色
colorall=rand(num_line,3);

% 这是5根线的颜色

% 这下面是绘图代码  先画线 后填充

% 绘制折线图
for i=1:size(Z,1)
    plot3(X,Y(i,:),Z(i,:),'LineWidth',1,'Color',colorall(i,:));
    hold on;
    fill3(X,Y(i,:),Z(i,:),colorall(i,:),'FaceAlpha',0.5,'EdgeColor',colorall(i,:))
end

view(53,51);

% 下面是修改图窗属性
set(gcf,'color','w');
set(gca,'Box','on');
set(gca,'Xgrid','on','Ygrid','on','Zgrid','on');
xlabel('弧度角-0-2pi','Fontname','微软雅黑');
ylabel('折线数量','Fontname','微软雅黑');
zlabel('值','Fontname','微软雅黑');
title('三维折线图','Fontname','微软雅黑');
% 这是修改Y轴的标签 每个线的名字
cell_str = {};
for j = 1:num_line
    cell_str{j} = ['line',num2str(j)];
end

set(gca,'Ylim',[0 num_line+1],'Ytick', [1:1:num_line], 'Yticklabel',cell_str);
print(gcf,'图','-dpng','-r300');
    


