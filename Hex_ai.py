import sys
import math
import random
import numpy as np
from Hex_utils import (SIZE, get_neighbors, initialize_board, display_board,
                       apply_action, check_winner, is_valid_action)

# =========================
#  ZOBRISTハッシュの準備
# =========================
# ZOBRIST_RANDOMS[cell_id][0] => cell_idにplayer=1が置かれている場合に使う乱数
# ZOBRIST_RANDOMS[cell_id][1] => cell_idにplayer=-1が置かれている場合に使う乱数
ZOBRIST_RANDOMS = None

def init_zobrist():
    global ZOBRIST_RANDOMS
    random.seed(2023)  # 再現性を持たせたい場合は固定seed
    ZOBRIST_RANDOMS = [
        (random.getrandbits(64), random.getrandbits(64))
        for _ in range(SIZE * SIZE)
    ]

def compute_zobrist_hash(board):
    """
    現在の盤面からZobristハッシュ値(64bit)を計算して返す。
    毎回盤面を走査する方式。4x4程度ならこれでも十分高速。
    """
    h = 0
    for cell_id in range(SIZE * SIZE):
        r, c = divmod(cell_id, SIZE)
        val = board[r][c]
        if val == 1:
            # player=1
            h ^= ZOBRIST_RANDOMS[cell_id][0]
        elif val == -1:
            # player=-1
            h ^= ZOBRIST_RANDOMS[cell_id][1]
    return h

# =========================
#  最短パス距離による評価
# =========================
def shortest_path_length(board, player):
    """
    board: 2次元リスト[[0, 1, ...], [...], ...]
    player: 1 or -1
    戻り値: 最短の手数 (存在しなければ None)
    """
    from collections import deque
    target_value = 1 if player == 1 else -1

    # スタート地点のリスト、ゴール条件も定義
    if player == 1:
        starts = [c for c in range(SIZE) if board[0][c] == target_value]
        is_goal = lambda cell: (cell // SIZE) == (SIZE - 1)  # 最下段に到達
    else:
        starts = [r*SIZE for r in range(SIZE) if board[r][0] == target_value]
        is_goal = lambda cell: (cell % SIZE) == (SIZE - 1)  # 右端に到達

    visited = set()
    queue = deque()
    for s in starts:
        queue.append((s, 0))  # (セル番号, 距離)
        visited.add(s)

    while queue:
        current, dist = queue.popleft()
        if is_goal(current):
            return dist  # 最短距離を返す

        for nxt in get_neighbors(current):
            r, c = divmod(nxt, SIZE)
            if board[r][c] == target_value and nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, dist + 1))

    return None  # つながらない場合

def evaluate_board(board, player):
    """
    board が勝敗未決の場合に使われるヒューリスティック。
    - 自分の最短接続長が短いほど有利
    - 相手の最短接続長が長いほど有利
    という観点でスコアを計算し、返す。
    大きいほど "playerにとって" 有利。
    """
    sp_self = shortest_path_length(board, player)
    sp_opp = shortest_path_length(board, -player)

    # どちらも経路なし
    if sp_self is None and sp_opp is None:
        return 0
    # 自分だけ繋げない
    if sp_self is None:
        return -9999
    # 相手だけ繋げない
    if sp_opp is None:
        return 9999

    # (相手の最短 - 自分の最短) が大きいほど自分に有利
    return (sp_opp - sp_self)

# =========================
#  Move Ordering
# =========================
def order_moves(board, valid_actions, player):
    """
    有効手を「evaluate_boardによる簡易評価」でソートする。
    評価値の高い順に返す → αβ探索で先に有望手を調べ、枝刈り効率UP。
    """
    scored_actions = []
    for action in valid_actions:
        r, c = divmod(action, SIZE)
        board[r][c] = player
        score = evaluate_board(board, player)
        board[r][c] = 0
        scored_actions.append((action, score))
    # scoreの降順にソート（高い＝有望）
    scored_actions.sort(key=lambda x: x[1], reverse=True)
    return [a for (a, s) in scored_actions]

# =========================
#  Negamax(α-β) with Zobrist
# =========================
def negamax(board, depth, alpha, beta, player, cache):
    """
    board: 現在の盤面 (2次元リスト)
    depth: 残り探索深度
    alpha, beta: α-β探索用ウィンドウ
    player: 手番 (1 or -1)
    cache: トランスポジションテーブル (dict)
    """
    # 勝利判定: 直前に打ったのは -player
    if check_winner(board, -player):
        # -playerが勝っている => 現在player視点では大きな負
        return (-999999 + depth, None)

    # 空きセル取得
    valid_actions = []
    for cell_id in range(SIZE * SIZE):
        if board[cell_id // SIZE][cell_id % SIZE] == 0:
            valid_actions.append(cell_id)
    if not valid_actions:
        # 空きがない => 実質終局
        return (0, None)

    # 深度0 => 評価値を返す
    if depth == 0:
        return (evaluate_board(board, player), None)

    # Zobristハッシュ + (player, depth) をキーにキャッシュ
    board_hash = compute_zobrist_hash(board)
    cache_key = (board_hash, player, depth)
    if cache_key in cache:
        return cache[cache_key]

    best_score = -math.inf
    best_move = None

    # Move Ordering で手をソート
    ordered_actions = order_moves(board, valid_actions, player)

    for action in ordered_actions:
        r, c = divmod(action, SIZE)
        board[r][c] = player
        score, _ = negamax(board, depth - 1, -beta, -alpha, -player, cache)
        board[r][c] = 0

        # negamaxなので符号反転
        score = -score

        # best_score更新
        if score > best_score:
            best_score = score
            best_move = action

        # α値更新
        alpha = max(alpha, score)
        if alpha >= beta:
            # βカット
            break

    # キャッシュ格納
    cache[cache_key] = (best_score, best_move)
    return (best_score, best_move)

def find_best_move(board, player, max_depth=6, cache=None):
    """
    現在の盤面boardと手番playerに対し、深さmax_depthで探索して最善手を返す。
    """
    if cache is None:
        cache = {}
    score, action = negamax(board, max_depth, -math.inf, math.inf, player, cache)
    return action

def play_hex_with_alphabeta(ai_players=(1, -1), max_depth=10):
    """
    AI同士またはAIと人間でHexをプレイする関数。
    
    ai_players: AIが担当するプレイヤーのリスト（1または-1を指定）
    max_depth: AIの探索深さ
    """
    # Zobrist初期化
    init_zobrist()

    board = initialize_board(SIZE)
    print("=== Game Start (4x4 Hex) ===")
    display_board(board)

    current_player = 1  # ゲーム開始時はPlayer 1
    move_count = 0

    while True:
        if current_player in ai_players:  # AIのターン
            print(f"AI (Player {current_player}) thinking ...")
            best_move = find_best_move(board, player=current_player, max_depth=max_depth)
            print(f"AI chooses action: {best_move}")
            apply_action(board, best_move, current_player)
        else:  # 人間のターン
            print(f"Your turn (Player {current_player}).")
            valid_actions = [
                i for i in range(SIZE * SIZE)
                if board[i // SIZE][i % SIZE] == 0
            ]
            print(f"Valid actions: {valid_actions}")
            action = int(input("Enter your move: "))
            while action not in valid_actions:
                print("Invalid move. Try again.")
                action = int(input("Enter your move: "))
            apply_action(board, action, current_player)

        display_board(board)

        # 勝利判定
        if check_winner(board, current_player):
            if current_player in ai_players:
                print(f"AI (Player {current_player}) wins!")
            else:
                print(f"You (Player {current_player}) win!")
            break

        current_player = -current_player
        move_count += 1

        # 念のため上限手数をチェック
        if move_count >= SIZE * SIZE:
            print("All cells are filled. But in Hex, one side must have won - check logic.")
            break

if __name__ == "__main__":
    """
    実行例:
      python main.py
      => "Enter AI players" が求められるので "1, -1" と入力すればAI同士の対戦
      => "1" ならPlayer1がAI、"-(1) -1" ならPlayer-1がAI
    """
    print("Enter AI players. For example:")
    print("  - '1, -1' for both players as AI.")
    print("  - '1' for only Player 1 as AI (default).")
    print("  - '-1' for only Player -1 as AI.")
    ai_input = input("Enter AI players (comma-separated): ").strip()
    if ai_input:
        ai_players = tuple(map(int, ai_input.split(",")))
    else:
        ai_players = (1,)  # default: Player1のみAI

    play_hex_with_alphabeta(ai_players=ai_players, max_depth=10)
