function GO = Fbarstacked(y,bw,C,FaceAlpha)
% y  - m*n matrix
% bw - bar width
% C  - m*3 color matrix

% 绘制堆叠图
x = 1:size(y,2);
GO = bar(x,y',bw,'stacked','EdgeColor','k');
% 绘制连接区块
conr = zeros(size(y,1)+1,size(y,2));
conr(2:end,:) = reshape([GO.YEndPoints]',size(y,2),size(y,1))';
for i = 1:length(GO)
    % 赋色
    GO(i).FaceColor = C(i,1:3);
    % 绘制填充
    for j = 1:size(y,2)-1
        L1 = min(conr(i,j),conr(i+1,j));
        L2 = max(conr(i,j),conr(i+1,j));
        R1 = min(conr(i,j+1),conr(i+1,j+1));
        R2 = max(conr(i,j+1),conr(i+1,j+1));
        fill([j+0.5*bw,j+1-0.5*bw,j+1-0.5*bw,j+0.5*bw],[L1,R1,R2,L2],...
            GO(i).FaceColor, ...
            'FaceAlpha',FaceAlpha, ...
            'EdgeColor','none');
    end
end

end