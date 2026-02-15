function Cobubbleheatmap(data,minsz,maxsz)
% data - m*n matrix, all the values in the matrix are at [-1 1]
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

% 构造气泡尺寸数据
% 负
minznega = 0;
maxznega = 1;
negadataidx = find(data<0);
Snega = minsz+(abs(Z(negadataidx))-minznega)./(maxznega-minznega).*(maxsz-minsz);
% 正
minzposi = 0;
maxzposi = 1;
posidataidx = find(data>=0);
Sposi = minsz+(Z(posidataidx)-minzposi)./(maxzposi-minzposi).*(maxsz-minsz);


% 绘制相关性气泡热图
mesh(Xmesh,Ymesh,Zmesh,'EdgeColor','k','LineWidth',1)
hold on
scatter(X(negadataidx),Y(negadataidx),Snega,Z(negadataidx),'filled')
scatter(X(posidataidx),Y(posidataidx),Sposi,Z(posidataidx),'filled')
view(0,90)

end