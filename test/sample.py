"""
方針
* 配置のみからいい感じにAを生成
* ダイクストラで最短経路を決定
* 経路上の次点以降の点をできる限り含むようにBを更新
"""

import sys
import heapq

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



# get input
N, M, T, L_A, L_B = map(int, input().split())

graph = [[] for _ in range(N)]

for _ in range(M):
    u, v = map(int, input().split())
    graph[u].append((v, 1))
    graph[v].append((u, 1))

t = [0] + list(map(int, input().split()))

# ダイクストラ法により計算ルート

route = [0]

for i in range(T-1):
    start = t[i]
    target = t[i+1]

    _, path = dijkstra(graph, start, target)
    route.extend(path[1:])


print(len(route))

sys.exit()
P = []
for _ in range(N):
    x, y = map(int, input().split())
    P.append((x, y))

# construct and output the array A
A = [0] * L_A
for i in range(L_A):
    if i < N:
        A[i] = i
    else:
        A[i] = 0
print(*A)

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
        for v in sorted(G[cur], key=lambda x: dist(P[x], P[pos_to])):
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
