function linearheatmap(X,Y,Z,ms,mt)
% ms    - maker size
% mt   - marker tpye, eg: 's'

data = [X(:),Y(:),Z(:)];
if nargin < 5
    scatter3(data(:,1),data(:,2),data(:,3),ms,data(:,3),'filled')
else
    scatter3(data(:,1),data(:,2),data(:,3),ms,data(:,3),mt,'filled')
end
end