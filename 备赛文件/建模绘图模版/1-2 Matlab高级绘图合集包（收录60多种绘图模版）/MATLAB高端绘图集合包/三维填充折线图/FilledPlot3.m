function p = FilledPlot3(X,data,map,LineWidth,FaceAlpha)
% X  -  x vector with m numbers
% data - m*n matrix, n is the number of lines
% map  - n*3 color matrix
% LineWidth  - LineWidth of the line
% FaceAlpha  - FaceAlpha of the patch
% Author:AKun
% 公众号：阿昆的科研日常

% Format x as a column vector
XX = X(:);

% Extract some important parameters
mini = min(XX);
maxi = max(XX);
[m,n] = size(data);

% Get the position of each dataset
offset = 1;
ypos = 0:offset:offset*(n-1);

% Prepare the data for patch plotting
X = repmat([mini;XX;maxi],1,n);
Y = repmat(ypos,m+2,1);
Z = fliplr([min(data,[],1);data;min(data,[],1)]);
h = gca;
hold on
for i = 1:n
    p(i) = patch(X(:,i),Y(:,i),Z(:,i),map(i,:), ...
        'FaceAlpha',FaceAlpha,...
        'EdgeColor','none',...
        'Parent', h);
    plot3(X(:,i),Y(:,i),Z(:,i),'linewidth',LineWidth,'Color',map(i,:),'Parent', h)
end
view(3)
% set(gca,'Projection','Perspective');
end