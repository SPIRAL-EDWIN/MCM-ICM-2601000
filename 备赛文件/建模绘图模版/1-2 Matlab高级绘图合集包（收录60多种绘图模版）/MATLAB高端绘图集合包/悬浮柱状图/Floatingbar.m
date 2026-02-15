function h = Floatingbar(x,zl,zu,color,scale)
% x     - x vector including N numbers
% zl    - y1 vector including N numbers
% zu    - y2 vector including N numbers
% color - N*3 FaceColor Matrix
% scale - bar width [0,1]
% 公众号：阿昆的科研日常

% assign default values for undeclared parameters
width = 1*scale;
x = x(:); 
zl = zl(:);  
zu = zu(:);
nx = length(x);

sx = width/2;

% plot the bars
for i=1:nx
    
    x1=x(i); zl1=zl(i); zu1=zu(i);
    % set vertices
    verts=[x1-sx zl1
           x1-sx zu1
           x1+sx zl1
           x1+sx zu1];
    % set faces
    faces=[1 2 4 3];
    
    patchinfo.Vertices = verts;
    patchinfo.Faces = faces;
    
    patchinfo.FaceColor = color(i,1:3);
    h(i)=patch(patchinfo);
    hold on;
end
end