clc; clear; close all;

% 读取点集
raw = jsondecode(fileread('points.json'));
if iscell(raw)
    points = vertcat(raw{:});   % 若为元胞，拼成 N×2 数组
else
    points = raw;               % 已是数值矩阵
end
points = double(points);

% Python 的 greedy(2, ...) ⇒ MATLAB 起点索引 = 2 + 1 = 3
start_idx = 3;

% 路径与长度
ans_route = greedy(start_idx, points);     % 访问顺序（不回到起点）
tour_len  = calc(ans_route, points);       % 回路总长度（含首尾闭合）

% 打印结果
disp('访问顺序（1-based 索引）:');
disp(ans_route.');
fprintf('回路总长度 = %.6f\n', tour_len);

% 画路径
figure;
plot_ans(ans_route, points);
title('Greedy 路径（起点=第3个点）');
