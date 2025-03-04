# グローバル変数
SIZE = 4  # 盤面のサイズ

# 隣接セルを取得する関数
def get_neighbors(cell):
    row, col = divmod(cell, SIZE)  # 座標取得
    directions = [
        (-1, 0),  # 上
        (1, 0),   # 下
        (0, -1),  # 左
        (0, 1),   # 右
        (-1, 1),  # 右上
        (1, -1)   # 左下
    ]
    neighbors = []
    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        if 0 <= nr < SIZE and 0 <= nc < SIZE:
            neighbors.append(nr * SIZE + nc)
    return neighbors

# 勝利条件を判定する関数
# def check_winner(board, player):
#     value = 1 if player == 1 else -1
#     visited = set()
#     stack = []

#     # プレイヤー1のスタートとゴール
#     if player == 1:
#         start = [i for i in range(SIZE) if board[0, i] == value]
#         goal = [i for i in range(SIZE * (SIZE - 1), SIZE * SIZE) if board[-1, i % SIZE] == value]
#     # プレイヤー2のスタートとゴール
#     else:
#         start = [i for i in range(0, SIZE * SIZE, SIZE) if board[i // SIZE, 0] == value]
#         goal = [i for i in range(SIZE - 1, SIZE * SIZE, SIZE) if board[i // SIZE, -1] == value]

#     stack.extend(start)

#     while stack:
#         current = stack.pop()
#         if current in goal:
#             return True

#         if current not in visited:
#             visited.add(current)
#             neighbors = get_neighbors(current)
#             for neighbor in neighbors:
#                 row, col = divmod(neighbor, SIZE)
#                 if board[row, col] == value:
#                     stack.append(neighbor)

#     return False

# MCTSのための勝利条件判定関数
def check_winner(board, player):
    import numpy as np  # 必要に応じてモジュールをインポート
    value = 1 if player == 1 else -1
    visited = set()
    stack = []

    # プレイヤー1のスタートとゴール
    if player == 1:
        start = [i for i in range(SIZE) if board[0][i] == value]
        goal = [i for i in range(SIZE * (SIZE - 1), SIZE * SIZE) if board[-1][i % SIZE] == value]
    # プレイヤー2のスタートとゴール
    else:
        start = [i for i in range(0, SIZE * SIZE, SIZE) if board[i // SIZE][0] == value]
        goal = [i for i in range(SIZE - 1, SIZE * SIZE, SIZE) if board[i // SIZE][-1] == value]

    stack.extend(start)

    while stack:
        current = stack.pop()
        if current in goal:
            return True

        if current not in visited:
            visited.add(current)
            neighbors = get_neighbors(current)
            for neighbor in neighbors:
                row, col = divmod(neighbor, SIZE)
                if board[row][col] == value:
                    stack.append(neighbor)

    return False


# 盤面の初期化
def initialize_board(size=SIZE):
    return [[0] * size for _ in range(size)]

# 行動が有効かを判定
def is_valid_action(board, action):
    row, col = divmod(action, SIZE)
    return board[row][col] == 0

# 行動を盤面に適用
def apply_action(board, action, player):
    row, col = divmod(action, SIZE)
    if not is_valid_action(board, action):
        raise ValueError(f"Action {action} is invalid!")
    board[row][col] = player

# 盤面を可視化
def display_board(board):
    symbols = {0: ".", 1: "O", -1: "X"}
    for row in board:
        print(" ".join(symbols[cell] for cell in row))
    print()

# セル番号を座標に変換
def convert_to_coordinates(action):
    return divmod(action, SIZE)

# 座標をセル番号に変換
def convert_to_action(row, col):
    return row * SIZE + col
