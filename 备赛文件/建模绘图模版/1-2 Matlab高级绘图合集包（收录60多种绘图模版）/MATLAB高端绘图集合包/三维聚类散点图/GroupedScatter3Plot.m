clc;
clear;
close all;
%三维聚类散点图
%% 数据准备
% 读取数据
A = load('data.txt');
% 初始化参数
X = A(:,1);
Y = A(:,2);
Z = A(:,3);
L = A(:,7);
lgs = {'Powerline','Low vegetation','Impervious surfaces','Car',...
       'Fence/Hedge','Roof','Facade','Shrub','Tree'};

%% 颜色定义
C = colormap(nclCM(486));%color包里选颜色

%% 图片尺寸设置（单位：厘米）
close all;
figureUnits = 'centimeters';
figureWidth = 16;
figureHeight = 12;

%% 窗口设置
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);

%% 三维聚类散点图绘制
groupedscatter3(X,Y,Z,L,C)
view(-27,27)

%% 细节优化
% 删除坐标刻度
set(gca,'xtick',[])
set(gca,'ytick',[])
set(gca,'ztick',[])
% 坐标区显示调整
axis off equal
% Legend
[hL1,hL2]= legend(lgs,...
                'FontWeight','bold',...
                'Box','on',...
                'Location', 'eastoutside',...
                'Orientation','vertical',...
                'Position',[0.016851481816408,0.643277084236093,0.299896380526801,0.335097009332513]);
set(hL1, 'FontName', 'Arial', 'FontSize', 10)
% Legend图例尺寸
for n=1:length(hL2)
    if sum(strcmp(properties(hL2(n)),'MarkerSize'))
        hL2(n).MarkerSize=8;
    elseif sum(strcmp(properties(hL2(n).Children),'MarkerSize'))
        hL2(n).Children.MarkerSize=8;
    end
end
% 删除白边
set(gca,'LooseInset',get(gca,'TightInset'))
% 背景颜色
set(gcf,'Color',[1 1 1])

%% 图片输出
figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[0 0 figW figH]);
fileout = 'test';
print(figureHandle,[fileout,'.png'],'-r300','-dpng');