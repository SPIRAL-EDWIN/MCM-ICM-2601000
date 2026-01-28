function plot_ans(route, points)
% PLOT_ANS 按给定路径绘制折线，并首尾闭合
%   route : 点索引向量
%   points: N×2 坐标矩阵

    xy = points(route, :);
    xy(end+1, :) = points(route(1), :);  % 闭合

    plot(xy(:,1), xy(:,2), '-o', 'LineWidth', 1.5);
    axis equal; grid on;
    xlabel('x'); ylabel('y'); title('Route');
end
