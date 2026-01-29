function bubbleheatmap(data,minsz,maxsz)
% data - m*n matrix
% minsz   - size of the min bubble
% maxsz   - size of the max bubble

% 构造气泡位置数据
x = 1:size(data,2);
y = 1:size(data,1);
[X,Y] = meshgrid(x,y);
Z = data;

% 构造网格数据
w = x(2)-x(1);
Xt = (x(1)-w/2):w:(x(end)+w/2);
Yt = (y(1)-y/2):w:(y(end)+y/2);
[Xmesh,Ymesh] = meshgrid(Xt,Yt);
Zmesh = zeros(size(Xmesh));

% 构造方块尺寸数据
minz = min(Z(:));
maxz = max(Z(:));
S = minsz+(Z(:)-minz)./(maxz-minz).*(maxsz-minsz);

% 绘制气泡热图
mesh(Xmesh,Ymesh,Zmesh,'EdgeColor','k','LineWidth',1)
hold on
scatter(X(:),Y(:),S,Z(:),'filled')
view(0,90)
end