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
            return window
    return False

A = [0, 1, 2, 3, 4, 5]
subarray = [1,2]
L_B = 4

print(find_and_print_subarray(A, subarray, L_B))

# %%
with open('./tools/out/0015.txt', 'r') as file:
    lines = file.readlines()
    first_line = lines[0].strip().split()
    num = 731
    L_B = 5
    print(first_line[num:num+L_B])

# %%