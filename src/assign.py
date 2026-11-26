# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""N×M 相似度矩阵的最优一一分配。

验证码只有 2~6 个提示字：规模小就穷举（保证全局最优），超过 6 个退化贪心。
"""
from __future__ import annotations

import itertools

import numpy as np

MAX_EXACT = 6  # 6! = 720，再大穷举代价不可接受


def _mean_margin(sim: np.ndarray, perm: list[int]) -> float:
    """每个提示字 top1-top2 余弦差的均值，可作低置信拒识依据。"""
    margins = []
    for i, j in enumerate(perm):
        if j < 0:
            continue
        row = np.sort(sim[i])[::-1]
        if row.size >= 2:
            margins.append(float(row[0] - row[1]))
    return float(np.mean(margins)) if margins else 0.0


def greedy_assignment(sim: np.ndarray) -> tuple[list[int], float, float]:
    """贪心：每次取全局最大，然后屏蔽该行该列。"""
    n, m = sim.shape
    perm = [-1] * n
    if n == 0 or m == 0:
        return perm, 0.0, 0.0
    work = sim.astype(np.float64).copy()
    for _ in range(min(n, m)):
        i, j = divmod(int(np.argmax(work)), m)
        perm[i] = j
        work[i, :] = -np.inf
        work[:, j] = -np.inf
    score = sum(float(sim[i, perm[i]]) for i in range(n) if perm[i] >= 0)
    return perm, score, _mean_margin(sim, perm)


def exact_assignment(sim: np.ndarray) -> tuple[list[int], float, float]:
    """穷举最优分配，返回 (perm, score, margin)；perm[i] 是第 i 行匹配到的列，未匹配为 -1。"""
    n, m = sim.shape
    if n == 0 or m == 0:
        return [-1] * n, 0.0, 0.0
    if max(n, m) > MAX_EXACT:
        return greedy_assignment(sim)

    if n <= m:
        # 提示字 <= 场景字：枚举 n 列的注入（mPn 种）
        perm = list(
            max(
                itertools.permutations(range(m), n),
                key=lambda p: sum(float(sim[i, p[i]]) for i in range(n)),
            )
        )
    else:
        # 场景字不够：枚举 n 行里挑 m 行（nPm 种），其余记为未匹配
        rows = max(
            itertools.permutations(range(n), m),
            key=lambda p: sum(float(sim[p[j], j]) for j in range(m)),
        )
        perm = [-1] * n
        for j, i in enumerate(rows):
            perm[i] = j

    score = sum(float(sim[i, perm[i]]) for i in range(n) if perm[i] >= 0)
    return perm, score, _mean_margin(sim, perm)
