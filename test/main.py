"""
方針
* 配置のみからいい感じにAを生成
* ダイクストラで最短経路を決定
* 経路上の次点以降の点をできる限り含むようにBを更新
"""

import sys
import heapq
from collections import deque, OrderedDict

# 関数
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
    # Aの中で長さL_Bのスライドウィンドウを作成し、比較する
    for i in range(len_A - L_B + 1):
        window = A[i:i + L_B]
        if set(subarray) <= set(window):
            return True, window, i
    return False, False, False

# get input
N, M, T, L_A, L_B = map(int, input().split())

graph = [[] for _ in range(N)]
for _ in range(M):
    u, v = map(int, input().split())
    graph[u].append((v, 1))
    graph[v].append((u, 1))

t = [0] + list(map(int, input().split()))

"""
P = []
for _ in range(N):
    x, y = map(int, input().split())
    P.append((x, y))
"""

# ダイクストラ法により最短経路算出
route = []
for i in range(T):
    start = t[i]
    target = t[i+1]
    _, path = dijkstra(graph, start, target)
    route.extend(path[1:])

# 配置に合わせてAを生成
## 今回は，0から幅優先探索していった結果を順次格納する

A = []
bfs_result = bfs(graph, 0)
A[:len(bfs_result)] = bfs_result

valid_list = [x for x in range(0, 600)]
if len(set(valid_list) - set(A)) != 0:
    A.extend(list(set(valid_list) - set(A)))

#A.extend([0]*(L_A-len(A)))

## 余っている部分に，ゴールから幅優先した結果を，余っている分格納する
remaining_len = L_A - len(A)
if remaining_len > 0:
    bfs_result_reversed = bfs(graph, t[-1])
    A.extend(bfs_result_reversed[:remaining_len])

print(*A)

# j = L_B〜1について，t[i:i+j]を含むR_Aを探す→見つかり次第，それをBとする
A_use = A

# TLE対策のため，jのサイズを制限する（maxはL_B）
j = min(10, L_B)

while len(route) > 0: # 通過したノードを経路から削除していく
    # R_Aの決定
    ## 次のノードを含むA_useの中の長さL_Bの部分配列を抽出→これにだけ探索をかける
    idx_next_list = [i for i, x in enumerate(A_use) if x == route[0]]  # 全ての位置を取得
    #print(idx_next_list)
    R_A_cand = []
    idx_R_A_cand = []
    for idx_next in idx_next_list: # 次ノードである各インデックスに対するループ
        for b in range(L_B): # 長さL_Bの候補リストの生成
            start = max(0, idx_next-L_B+b+1)
            end = min(start+L_B, len(A_use))
            if start + L_B > len(A_use):
                start = (len(A_use) - L_B) + 1
                end = len(A_use)
            #print(start, end)
            #print(end-start)
            R_A_cand.append(A_use[start:end])
            idx_R_A_cand.append(start)
    #print(R_A_cand)
    #print(idx_R_A_cand)

    #R_A_cand = list(OrderedDict.fromkeys([tuple(x) for x in R_A_cand]))
    #idx_R_A_cand = list(OrderedDict.fromkeys(idx_R_A_cand))

    #print(R_A_cand)
    #print(idx_R_A_cand)

    #R_A_cand = list(OrderedDict.fromkeys([tuple(x) for x in R_A_cand]))
    #idx_R_A_cand = list(set(idx_R_A_cand))
    if j >= len(route): # routeの残り数に合わせて，探索範囲を制限
        j = min(len(route), L_B)
    
    for i in range(j, 0, -1):
        if i >1:
            subarray = route[0:i]
            
            for idx_R_A_cand_tmp, R_A_cand_tmp in zip(idx_R_A_cand, R_A_cand):
                #print(idx_R_A_cand_tmp, R_A_cand_tmp)
                if set(subarray) <= set(R_A_cand_tmp):
                    idx_R_A = idx_R_A_cand_tmp
                    break
            if set(subarray) <= set(R_A_cand_tmp):
                break

        else:
            subarray = [route[0]]
            idx_R_A = A_use.index(subarray[0])
    
    # idx_R_AがL_A - lを超えないようにする
    if idx_R_A > L_A - L_B:
        idx_R_A = idx_R_A - (idx_R_A + L_B - L_A) - 1
    
    #print(subarray)
    #print(idx_R_A)

    # 信号操作を行う
    print('s', L_B, idx_R_A, 0)

    # 移動を行う
    for s_i in subarray:
        print('m', s_i)
    
    # 通ったノードを削除
    route = route[len(subarray):]

    #sys.exit()