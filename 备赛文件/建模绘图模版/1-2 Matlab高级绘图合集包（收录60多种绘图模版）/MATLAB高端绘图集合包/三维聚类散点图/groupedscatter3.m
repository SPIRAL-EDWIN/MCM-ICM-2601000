function groupedscatter3(X,Y,Z,L,C)
% X, Y, Z  -  M*1 matrix
% L        -  M*1 label matrix
% C        -  nCluster*3 color matrix

Cluster = unique(L);
nCluster = length(Cluster);

for i = 1:nCluster
    id = find(L==Cluster(i));  
    scatter3(X(id,1),Y(id,1),Z(id,1),8,...
        'Marker','o', ...
        'MarkerEdgeColor',C(i,1:3), ...
        'MarkerFaceColor',C(i,1:3));
    hold on
end

end