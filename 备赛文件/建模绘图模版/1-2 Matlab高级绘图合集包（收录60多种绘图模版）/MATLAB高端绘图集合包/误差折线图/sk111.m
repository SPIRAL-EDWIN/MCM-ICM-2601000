clear all ;clc
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
plot(x,yfit,'-ko')
xlabel('Ce'),ylabel('Ce/qe') 
hold on
err=0.3*ones(1,length(x));

color=[0.294117647058824,0,0.509803921568627];
errorbar(x,yfit,err);

