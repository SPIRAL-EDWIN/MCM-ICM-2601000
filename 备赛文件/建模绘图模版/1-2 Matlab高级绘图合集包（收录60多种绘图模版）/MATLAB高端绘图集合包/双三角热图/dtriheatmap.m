function dtriheatmap(data1,data2,map1,map2,xlb,ylb,mode)
% data1,data2 - m*n matrix
% map1,map2  - colormap matrix
% xlb     - xticklabels
% ylb     - yticklabels
% mode   - set position of colorbar, 'right' or 'bottom'

% 构造位置数据
x = 1:size(data1,2);
y = 1:size(data1,1);
[X,Y] = meshgrid(x,y);
Y = flipud(Y);
% 构造网格数据
w = x(2)-x(1);
Xt = (x(1)-w/2):w:(x(end)+w/2);
Yt = (y(1)-y/2):w:(y(end)+y/2);
[Xmesh,Ymesh] = meshgrid(Xt,Yt);
Ymesh = flipud(Ymesh);
Zmesh = zeros(size(Xmesh));

%% 下三角
% 三角热图数据构造
temp = triu(ones(size(data1)),0);
idx = find(temp==1);
data1(idx) = NaN;
Z1 = data1;
xp = [X(:)-w/2,X(:)+w/2,X(:)+w/2,X(:)-w/2];
yp = [Y(:)+w/2,Y(:)+w/2,Y(:)-w/2,Y(:)-w/2];
% 颜色构造
minc = min(data1(:));
maxc = max(data1(:));
num = size(map1,1);
x = linspace(minc,maxc,num);
cmap1 = interp1(x,map1,data1(:),'linear'); %...插值
% 三角热图绘制
for i = 1:length(data1(:))
    if isnan(data1(i))
        
    else
        patch(xp(i,1:4),yp(i,1:4),cmap1(i,1:3),'edgecolor','none')
    end
end

%% 上三角
% 三角热图数据构造
temp = triu(ones(size(data2)),1);
idx = find(temp==0);
data2(idx) = NaN;
Z2 = data2;
xp = [X(:)-w/2,X(:)+w/2,X(:)+w/2,X(:)-w/2];
yp = [Y(:)+w/2,Y(:)+w/2,Y(:)-w/2,Y(:)-w/2];
% 颜色构造
minc = min(data2(:));
maxc = max(data2(:));
num = size(map2,1);
x = linspace(minc,maxc,num);
cmap2 = interp1(x,map2,data2(:),'linear'); %...插值
% 三角热图绘制
for i = 1:length(data2(:))
    if isnan(data2(i))
        
    else
        patch(xp(i,1:4),yp(i,1:4),cmap2(i,1:3),'edgecolor','none')
    end
end

%% 绘制网格
hold on
mesh(Xmesh,Ymesh,Zmesh,'facecolor','none','EdgeColor',[0.2 0.2 0.2],'LineWidth',1)
view(0,90)
axis equal tight off

%% 添加标签
set(gca,'yticklabels',[])
for i = 1:size(Z1,1)
    text(X(1,i),Y(1,1)+0.6,xlb{i},'HorizontalAlignment','center','VerticalAlignment','bottom','FontSize',11,'FontName','arial','color','k')
    text(X(1,1)-0.6,Y(i,1),ylb{i},'HorizontalAlignment','right','VerticalAlignment','middle','FontSize',11,'FontName','arial','color','k')
end

%% 绘制颜色条
colorbar_k2dt(mode,Z1,map1,Z2,map2)

end