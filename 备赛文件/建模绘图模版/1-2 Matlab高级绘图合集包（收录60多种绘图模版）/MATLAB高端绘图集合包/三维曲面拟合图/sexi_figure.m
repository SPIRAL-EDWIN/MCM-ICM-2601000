%% 生成色系代码
%% 第一种颜色柱
clc;
clear;
color1 = [0 0 0]./255; % 第一种颜色
color2 = [255 255 255]./255;% 第二种颜色
color3 = [234,23,12]./255;
n=64;% 60个颜色数据点
R1 = [color1(1),color2(1),color3(1)]';
G1 = [color1(2),color2(2),color3(2)]';
B1 = [color1(3),color2(3),color3(3)]';
mycolor_heibai = [R1,G1,B1];
colormap(mycolor_heibai);
c= colorbar;
save mycolor_heibai
%% 生成颜色柱
clc;
clear;
figure('color',[1 1 1]);
color1 = [0 176 240]./255; % 第一种颜色
color2 = [226 240 217]./255;% 第二种颜色
n=64;% 颜色数据点
R1 =(linspace(color1(1),color2(1),n))';
G1 =(linspace(color1(2),color2(2),n))';
B1 =(linspace(color1(3),color2(3),n))';
mycolor2 = [R1,G1,B1];
colormap(mycolor2);
c= colorbar;
save mycolor2
%% 设置colorbar的刻度与标签显示参数
set(c,'tickdir','out');
set(c,'Ytick',0.0:0.2:1.0);
set(c,'YtickLabel',{'0.0','0.2','0.4','0.6','0.8','1.0'});
set(c,'Linewidth',1.1);
set(c,'FontSize',13);


