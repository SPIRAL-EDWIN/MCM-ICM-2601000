function route = greedy(start_idx, points)
% GREEDY 最近邻贪心构造路径（不回到起点，仅返回访问顺序）
%   start_idx : 起点索引（1-based）
%   points    : N×2 坐标矩阵
%   返回 route: 访问顺序（长度 N）

    N = size(points, 1);
    if start_idx < 1 || start_idx > N
        error('start_idx 超出范围。');
    end

    visited = false(N, 1);
    route   = zeros(N, 1);
    route(1) = start_idx;
    visited(start_idx) = true;

    for i = 1:N-1
        last = route(i);
        nxt = 0; 
        min_len = inf;

        for j = 1:N
            if ~visited(j)
                d = hypot(points(last,1)-points(j,1), points(last,2)-points(j,2));
                if d < min_len
                    min_len = d;
                    nxt = j;
                end
            end
        end

        route(i+1) = nxt;
        visited(nxt) = true;
    end
end
