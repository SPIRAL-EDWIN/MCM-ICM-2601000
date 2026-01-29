function [h_surf,u1,u2,u3] = waffle(data,lw)
% data - m*n matrix
% lw   - line width of the grid

[u1,u2,u3] = unique(data,'stable');
A = reshape(u3,size(data));
x = 1:size(data,2);
y = 1:size(data,1);
[xx,yy] = meshgrid(x,y);
data1 = [xx(:),yy(:),A(:)];
h_surf = gridplot2(data1,lw);

end

function h_surf = gridplot2(h, lw)


gd = h;
h = gca;

% Sort the input data
gd = sortrows(gd, 1);

% Assign the corresponding x, y, and z values
x = gd(:, 1);
y = gd(:, 2);
z = gd(:, 3);

% Find the increment in x and y directions
if x(1) == x(2)
    ny = diff(find(y == y(1), 2));
    nx = numel(x) / ny;
elseif y(1) == y(2)
    nx = diff(find(x == x(1), 2));
    ny = numel(y) / nx;
end

dx = (x(end) - x(1)) / (nx - 1);
dy = (y(end) - y(1)) / (ny - 1);

% Make coordinates of the verices of the patches
x_vert = [x - dx / 2, x + dx / 2, x + dx / 2, x - dx / 2];
y_vert = [y - dy / 2, y - dy / 2, y + dy / 2, y + dy / 2];

% Plot the patches
for i = 1:size(x_vert,1)
    h_surf(i) = patch(x_vert(i,:), y_vert(i,:), z(i),...
                      'edgecolor','w',...
                      'linewidth',lw,...
                      'Parent', h);
end
end