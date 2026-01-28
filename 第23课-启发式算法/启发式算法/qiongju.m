% tsp_bruteforce.m — 穷举所有排列求最短 TSP 回路
% 依赖：calc.m, plot_ans.m；数据文件：points.json

clc; clear; close all;

%% 读取点集（points.json: [[x1,y1],[x2,y2],...])
raw = jsondecode(fileread('points.json'));
if iscell(raw)
    points = vertcat(raw{:});   % 若为元胞，拼成 N×2 数组
else
    points = raw;               % 已是数值矩阵
end

N = size(points, 1);
if N < 2
    error('点数不足。');
end

%% 穷举排列：固定起点为 1，其余点 2..N 全排列
idx = 2:N;                  % 对应 Python 的 nums = range(1,N)
allPerms = perms(idx);      % MATLAB 的全排列（每行一个排列）
numPerms = size(allPerms, 1);

fprintf('将枚举 %d 个排列（N=%d）。\n', numPerms, N);

min_len = inf;
best_route = [];

tic;
for r = 1:numPerms
    route = [1, allPerms(r, :)];              % 对应 Python 的 route.insert(0, 0)
    L = calc(route, points);                  % 回路长度（含首尾闭合）
    if L < min_len
        min_len = L;
        best_route = route;
    end
end
elapsed = toc;

%% 输出结果
fprintf('最短回路长度：%.10f\n', min_len);
fprintf('耗时：%.6f 秒\n', elapsed);
disp('最优访问顺序（1-based 索引）：');
disp(best_route);

%% 画最优路径
figure;
plot_ans(best_route, points);
title('最短回路（穷举）');
