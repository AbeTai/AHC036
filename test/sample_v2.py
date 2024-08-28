def dist(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

import sys
import heapq
import numpy as np

def dijkstra_distance(graph, start, target):
    # 初期化
    n = len(graph)
    visited = [False] * n
    distance = [sys.maxsize] * n
    distance[start] = 0
    pq = [(0, start)]

    # ダイクストラ法
    while pq:
        # 未処理の中で最小の距離を持つ頂点を取り出す
        dist, u = heapq.heappop(pq)
        if visited[u]:
            continue

        # 目標ノードに到達したら距離を返す
        if u == target:
            return distance[u]

        # 訪問済みにする
        visited[u] = True

        # uから到達可能な頂点の距離を更新する
        for v, weight in graph[u]:
            if not visited[v]:
                new_distance = distance[u] + weight
                if new_distance < distance[v]:
                    distance[v] = new_distance
                    heapq.heappush(pq, (new_distance, v))

    # ターゲットに到達できなかった場合
    return sys.maxsize


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

# 現在地のノードからいけるノードを生成
def get_reachable_nodes(graph, start_node):
    visited = set()
    stack = [start_node]
    reachable_nodes = []

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            reachable_nodes.append(node)
            for neighbor, _ in graph[node]:
                if neighbor not in visited:
                    stack.append(neighbor)
    
    return reachable_nodes



# get input
N, M, T, L_A, L_B = map(int, input().split())

graph = [[] for _ in range(N)]

for _ in range(M):
    u, v = map(int, input().split())
    graph[u].append((v, 1))
    graph[v].append((u, 1))

t = [0] + list(map(int, input().split()))

# construct and output the array A
A = [0] * L_A
for i in range(L_A):
    if i < N:
        A[i] = i
    else:
        A[i] = 0

now_node = t[0]
idx_t = 0
flg_comp = 0

while flg_comp != 600:
    # 現在の都市と繋がってる都市を抽出
    neighbors = [tmp[0] for tmp in graph[now_node]]
    
    # neighborsを含む長さl_BのAの部分集合R_Aの候補を作成
    subarrays = []
    for n in neighbors:
        idx_n = A.index(n)
        for i in range(L_B):
            start = max(0, idx_n - i)
            end = min(idx_n + L_B - i, L_A)
            subarrays.append(A[start:end])

    end_node_cand = []

    for s in subarrays:
        # R_Bそれぞれについて，現在地からいける都市の候補を生成
        ## subarraysに存在するノードのグラフだけにする
        ### 一旦，sに存在するノードのインデックス以外のリストの中身を削除
        graph_tmp = [[] for _ in range(N)]
        s_set = set([0] + s)
        for idx_s in s_set:
            for neighbor, weight in graph[idx_s]:
                if neighbor in s_set:
                    graph_tmp[idx_s].append((neighbor, weight))
        print(graph_tmp)
        
        ### 到着先のインデックスも追加
        for u in range(len(graph_tmp)):
            for v, weight in graph_tmp[u]:
                if u not in [n for n, _ in graph_tmp[v]]:
                    graph_tmp[v].append((u, weight))
        #print(graph_tmp)

        # 信号変更なしで行きうるノードを算出
        reachable_nodes = get_reachable_nodes(graph_tmp, now_node)
        reachable_nodes.remove(now_node)

        # 各ノードと次のノードとの距離を計算
        distance_reachable_nodes_to_taeget = []
        for r in reachable_nodes:
            
            distance_reachable_nodes_to_taeget.append(dijkstra_distance(graph, r, t[idx_t+1]))

        # 一気に移動するノードを決定
        end_node_per_s = reachable_nodes[np.argmin(distance_reachable_nodes_to_taeget)]
        end_node_cand.append([end_node_per_s, min(distance_reachable_nodes_to_taeget)])

    # 今回の信号変更で動くノードを決定
    min_index = np.argmin([item[1] for item in end_node_cand])
    result = end_node_cand[min_index][0]
    print(end_node_cand)
    print(result)
    
    # 現在地からノードまでの経路を取得
    print(now_node, result)
    print(graph_tmp)
    _, move_list = dijkstra(graph_tmp, now_node, result) # 到達できないend_nodeを指定してる→行きうるコード算出に問題あり？多分graph_tmpの部分
    print(move_list)

    # 情報を更新
    now_node = result
    
        
    sys.exit()

    

sys.exit()

# TODO: 信号更新タイミングへの対応
# TODO: graph_tmpが最後のsubarrayになってしまう問題→採用する信号パターンの場合のgraph_tmpになっていない




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
