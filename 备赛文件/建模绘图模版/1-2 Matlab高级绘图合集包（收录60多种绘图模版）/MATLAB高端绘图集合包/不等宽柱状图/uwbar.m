function h = uwbar(x,y,lw,color)
% x     - x vector including N numbers
% y     - y vector including N numbers
% lw    - line width
% color - N*3 FaceColor Matrix
% 公众号：阿昆的科研日常

% assign default values for undeclared parameters
x = x(:);  
y = y(:);
nx = length(x)-1;

% plot the bars
for i=1:nx
    
    x1=x(i); x2=x(i+1); y1=y(i);
    % set vertices
    verts=[x1 0
           x1 y1
           x2 y1
           x2 0];
    % set faces
    faces=[1 2 3 4];
    
    patchinfo.Vertices = verts;
    patchinfo.Faces = faces;
    
    patchinfo.FaceColor = color(i,1:3);
    h(i)=patch(patchinfo,'LineWidth',lw);
    hold on;
end
end