function stripheatmap(X,Y,Z,lw)
% X,Y,Z  - M*N matrix, contain N strips
% lw     - line width
N = size(X,2);
X(end+1,1:N) = NaN(1,N);
Y(end+1,1:N) = NaN(1,N);
Z(end+1,1:N) = NaN(1,N);
data = [X(:),Y(:),Z(:)];
patch(data(:,1),data(:,2),data(:,3),data(:,3),'edgecolor','interp','LineWidth',lw)
end