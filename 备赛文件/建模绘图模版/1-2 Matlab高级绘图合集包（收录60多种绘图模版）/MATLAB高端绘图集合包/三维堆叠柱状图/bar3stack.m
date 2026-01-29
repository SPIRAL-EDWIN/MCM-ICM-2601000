function h = bar3stack(z,s,c)
% 公众号：阿昆的科研日常
% z   - m*n*l matrix
% s   - width of the bars (0 to 1)
% c   - l*3 color scheme matrix

x = 1:size(z,2); 
y = 1:size(z,1); 
ngroups = size(z,3); 

% get bin centers
[x1,y1] = meshgrid(x,y);
% initial zldata
z1 = zeros(size(z,1),size(z,2));

for i = 1:ngroups
    z2 = z1;
    z1 = z1+squeeze(z(:,:,i));
    h(i) = bar3level(x1, y1, z2, z1, c(i,1:3), s);
    hold on
end
% axis tight
set(gca,'ydir','reverse')
set(gca,'xlim',[1-s/2 max(x)+s/2])
dx = diff(get(gca,'xlim'));
dy = size(z,1)+1;
dz = (sqrt(5)-1)/2*dy;
set(gca,'PlotBoxAspectRatio',[dx dy dz])
view(gca, 3);
grid on
box on
end

function h = bar3level(x,y,zl,zu,color,scale)

% assign default values for undeclared parameters
width = 1*scale;
x = x(:); 
y = y(:); 
zl = zl(:);  
zu = zu(:);
nx = length(x);

sx = width/2;
sy = width/2;

% plot the bars
for i=1:nx
    
    x1=x(i); y1=y(i); zl1=zl(i); zu1=zu(i);
    % set vertices
    verts=[x1-sx y1-sy zl1
           x1-sx y1+sy zl1
           x1+sx y1-sy zl1
           x1+sx y1+sy zl1
           x1-sx y1-sy zu1
           x1-sx y1+sy zu1
           x1+sx y1-sy zu1
           x1+sx y1+sy zu1];
    % set faces
    faces=[1 2 6 5;
           3 1 5 7;
           4 3 7 8;
           2 4 8 6;
           5 6 8 7;
           1 2 4 3];
    
    patchinfo.Vertices = verts;
    patchinfo.Faces = faces;
    
    patchinfo.FaceColor = color;
    h=patch(patchinfo,'CDataMapping','scaled','Edgecolor','k');
    hold on;
end
end