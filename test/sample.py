"""
方針
* 配置のみからいい感じにAを生成
* ダイクストラで最短経路を決定
* 経路上の次点以降の点をできる限り含むようにBを更新
"""

import sys
import heapq
from collections import deque
import random

# 関数
def dist(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def dijkstra(graph, start, target):
    # 初期化
    n = len(graph)
    visited = [False] * n
    distance = [sys.maxsize] * n
    parent = [-1] * n  # 親ノードを記録
    distance[start] = 0
    pq = [(0, start)]

    # ダイクストラ法
    while pq:
        # 未処理の中で最小の距離を持つ頂点を取り出す
        dist, u = heapq.heappop(pq)
        if visited[u]:
            continue

        # 目標ノードに到達したら経路を復元して返す
        if u == target:
            return distance[u], get_path(parent, target)

        # 訪問済みにする
        visited[u] = True

        # uから到達可能な頂点の距離を更新する
        for v, weight in graph[u]:
            if not visited[v]:
                new_distance = distance[u] + weight
                if new_distance < distance[v]:
                    distance[v] = new_distance
                    parent[v] = u  # 親ノードを記録
                    heapq.heappush(pq, (new_distance, v))

    # ターゲットに到達できなかった場合
    return sys.maxsize, []

def get_path(parent, target):
    # 終点から親ノードを辿っていく
    path = []
    while target != -1:
        path.append(target)
        target = parent[target]
    path.reverse()
    return path


def bfs(graph, start):
    visited = [False] * len(graph)
    queue = deque([start])
    bfs_order = []

    while queue:
        node = queue.popleft()
        if not visited[node]:
            visited[node] = True
            bfs_order.append(node)
            for neighbor, _ in graph[node]:
                if not visited[neighbor]:
                    queue.append(neighbor)
    
    return bfs_order

def find_and_print_subarray(A, subarray, L_B):
    """
    長さL_BのAの部分配列にsubarrayが含まれていたら、その部分配列を出力する関数
    """
    len_A = len(A)
    len_subarray = len(subarray)

    # Aの中で長さL_Bのスライドウィンドウを作成し、比較する
    for i in range(len_A - L_B + 1):
        window = A[i:i + L_B]
        if set(subarray) <= set(window):
            return window, i
    return False, False



# get input
N, M, T, L_A, L_B = map(int, input().split())

graph = [[] for _ in range(N)]

for _ in range(M):
    u, v = map(int, input().split())
    graph[u].append((v, 1))
    graph[v].append((u, 1))

t = [0] + list(map(int, input().split()))

# ダイクストラ法により最短経路算出

route = [0]

for i in range(T-1):
    start = t[i]
    target = t[i+1]

    _, path = dijkstra(graph, start, target)
    route.extend(path[1:])

P = []
for _ in range(N):
    x, y = map(int, input().split())
    P.append((x, y))

# 配置に合わせてAを生成
## 今回は，0から幅優先探索していった結果を順次格納する
## 余った部分には0〜599の離散一様乱数を入れる

A = []
bfs_result = bfs(graph, 0)
A[:len(bfs_result)] = bfs_result

for i in range(N, L_A):
    A.append(random.randint(0, T-1))



# j = L_B〜1について，t[i:i+j]を含むR_Aを探す→見つかり次第，それをBとする

while len(route): # 通過したノードを経路から削除していく
    tmp_node = route[0]

    # R_Aの決定
    for i in range(L_B+1, 0):
        print(i)
        subarray = route[1:i]
        res, idx_R_A = find_and_print_subarray(A, subarray, L_B)

        if res:
            break
    
    # 信号操作を行う
    print('s', L_B, idx_R_A, 0)

    # 移動を行う
    for s_i in subarray:
        print('m', s_i)
    
    # 通ったノードを削除
    route = route[1+len(subarray):]
    break
    


sys.exit()


pos_from = 0
for pos_to in t:

    # determine the path by DFS
    path = []
    visited = [False] * N

    def dfs(cur, prev):
        if visited[cur]:
            return False

        if cur != pos_from:
            path.append(cur)

        visited[cur] = True
        if cur == pos_to:
            return True

        # visit next city in ascending order of Euclidean distance to the target city
        for v in sorted(graph[cur], key=lambda x: dist(P[x], P[pos_to])):
            if v == prev:
                continue
            if dfs(v, cur):
                return True
        path.pop()
        return False

    dfs(pos_from, -1)

    # for every step in the path, control the signal and move
    for u in path:
        print('s', 1, u, 0)
        print('m', u)

    pos_from = pos_to
