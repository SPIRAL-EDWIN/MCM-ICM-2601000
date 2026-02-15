%% 生成色系代码
clc;
clear;
color1 = [255 255 255]./255; % 第一种颜色
color2 = [237 109 160]./255;% 第二种颜色
n=64;% 60个颜色数据点
R1 =(linspace(color1(1),color2(1),n))';
G1 =(linspace(color1(2),color2(2),n))';
B1 =(linspace(color1(3),color2(3),n))';
mycolor1 = [R1,G1,B1];
color1 = [237 109 160]./255; % 第一种颜色
color2 = [68 39 103]./255;% 第二种颜色
n=64;% 60个颜色数据点
R1 =(linspace(color1(1),color2(1),n))';
G1 =(linspace(color1(2),color2(2),n))';
B1 =(linspace(color1(3),color2(3),n))';
mycolor2 = [R1,G1,B1];
mycolor = [mycolor1;mycolor2];
colormap(mycolor);
c= colorbar;

%% 设置colorbar的刻度与标签显示参数
set(c,'tickdir','out');
set(c,'Ytick',0.0:0.2:1.0);
set(c,'YtickLabel',{'0.0','0.2','0.4','0.6','0.8','1.0'});
set(c,'Linewidth',1.1);
set(c,'FontSize',13);


