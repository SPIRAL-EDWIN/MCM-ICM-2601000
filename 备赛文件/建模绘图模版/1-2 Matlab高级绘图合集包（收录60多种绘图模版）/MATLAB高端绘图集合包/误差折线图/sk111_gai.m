clear all ;
clc;
figure('color',[1 1 1]);
Ce=[0.2288 0.6293 0.9823 1.449];
qe=[0.0632 0.1072 0.1373 0.2];
x=Ce;%x=Ce
y=Ce./qe;%y=Ce/qe
p=polyfit(x,y,1);
xi=linspace(x(1),x(end),20);
yi=polyval(p,xi);
plot(x,y,'*')
hold on
yfit=polyval(p,x);
plot(x,yfit,'ko')
xlabel('Ce'),ylabel('Ce/qe') 
hold on
err=0.3*ones(1,length(x));

color=[0.294117647058824,0,0.509803921568627];

%我修改为了x  和  yfit  
e = errorbar(x,yfit,err,'-s','MarkerSize',6,...
    'MarkerEdgeColor',color,'MarkerFaceColor',color);



e.Color=color;
e.CapSize=15;
e.LineWidth=1.2;
title('Langmuir')
legend({'测量值','拟合曲线','拟合值误差'},'Location','NorthWest')
Qmax=1/p(1)
 KL=1/(Qmax*p(2)) 
 NRMSE=goodnessOfFit(y',yfit','NRMSE')
NMSE=goodnessOfFit(y',yfit','NMSE');


%% 这是加了去掉右边和上面的刻度的代码
box off;
ax2 = axes('Position',get(gca,'Position'),...
    'XAxisLocation','top',...
    'Color','none',...
    'Xcolor','k','Ycolor','k');
set(ax2,'YTick',[]);
set(ax2,'XTick',[]);
box on;


