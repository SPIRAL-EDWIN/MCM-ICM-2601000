function [ax1,ax2,b1,b2]=Butterfly(figureHandle,X1,X2,Label,type)
% figureHandle  -  Figure对象
% X1            -  左翼数据，如果X1是m×n矩阵，则创建每组包含n个条形的m个组。
% X2            -  右翼数据，如果X2是m×n矩阵，则创建每组包含n个条形的m个组。
% Label         -  中部标签注释，1×n元胞数组
% type          -  'stacked'绘制堆叠样式，'normal'绘制正常样式
% By 阿昆的科研日常

%% 数据初始化
Y = 1:size(X1,2);


switch type
    case 'stacked'
        %% 左翼绘制
        % 创建坐标区
        ax1 = axes('Parent',figureHandle,'Position',[0.05,0.08,0.5-0.1,0.9]);
        % 绘制水平柱状图
        b1 = barh(ax1,Y,X1,0.6,'stacked');
        %% 右翼绘制
        % 创建坐标区
        ax2 = axes('Parent',figureHandle,'Position',[0.55,0.08,0.5-0.1,0.9]);
        % 绘制水平柱状图
        b2 = barh(ax2,Y,X2,0.6,'stacked');

    case 'normal'
        %% 左翼绘制
        % 创建坐标区
        ax1 = axes('Parent',figureHandle,'Position',[0.05,0.08,0.5-0.1,0.9]);
        % 绘制水平柱状图
        b1 = barh(ax1,Y,X1,0.6);
        %% 右翼绘制
        % 创建坐标区
        ax2 = axes('Parent',figureHandle,'Position',[0.55,0.08,0.5-0.1,0.9]);
        % 绘制水平柱状图
        b2 = barh(ax2,Y,X2,0.6);

end

%% 中部注释
lim = get(ax2,'XLim');
xx = lim(2)/0.4*(-0.05);
for i = 1:length(Y)
    text(xx, i, Label{i}, ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','middle', ...
        'FontSize',9, ...
        'FontName','Arial', ...
        'color','k')
end

end