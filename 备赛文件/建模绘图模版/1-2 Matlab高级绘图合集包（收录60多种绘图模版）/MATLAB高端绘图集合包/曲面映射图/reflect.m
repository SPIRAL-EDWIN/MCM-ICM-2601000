%% matlab绘制三维曲面图
%获取数据
clc;
clear;
for i = 1:100
    x(i) = rand*(10-20)+20;
    y(i) = rand*(15-25)+25;
    z(i) = rand*(30-40)+40;
end


load color.mat
method_fit = 'v4';
%插值生成网格化数据
[XX,YY,Z]=griddata(x,y,z,linspace(min(x),max(x))',linspace(min(y),max(y)),method_fit);
figure('color',[1 1 1]);
mesh(XX,YY,Z,'edgecolor','none','facecolor','interp');%三维曲面
hold on;
mesh(XX,YY,20+0*Z,Z)
hold on;
plot3(XX,YY,Z,'o','MarkerFaceColor','k','MarkerSize',1.5);
xlabel('X轴');
ylabel('Y轴');
zlabel('Z轴');
colormap(color);
colorbar;
h = colorbar;%右侧颜色栏
set(get(h,'label'),'string','z值');%给右侧颜色栏命名
grid on;
shading interp
warning('off');



