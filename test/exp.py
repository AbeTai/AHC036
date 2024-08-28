import sys
import heapq

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

graph = [
    [(1, 2), (2, 4)],
    [(0, 2), (2, 1), (3, 4)],
    [(0, 4), (1, 1), (3, 1), (4, 3)],
    [(1, 4), (2, 1), (4, 1)],
    [(2, 3), (3, 1)]
]

print(dijkstra(graph, 0,1))