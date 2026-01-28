function L = calc(route, points)
% CALC 计算给定路径(route)的总长度（包含首尾闭合的一条边）
%   route : 点索引向量（1-based）
%   points: N×2 坐标矩阵，每行 [x y]
%   返回 L ：总长度

    N = numel(route);
    L = 0;

    % 相邻段长度
    for i = 1:N-1
        a = points(route(i),   :);
        b = points(route(i+1), :);
        L = L + hypot(b(1)-a(1), b(2)-a(2));
    end

    % 闭合段（最后一个到第一个）
    a = points(route(end), :);
    b = points(route(1),  :);
    L = L + hypot(b(1)-a(1), b(2)-a(2));
end
