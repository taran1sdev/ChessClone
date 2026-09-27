import io
import os
import sys
import berserk 
import chess
import chess.pgn
import duckdb
import numpy as np
import torch
from torch.utils.data import Dataset

# Connect to lichess API and convert PGNs to 3D arrays for use with tensor

PIECE_TYPES = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING
]

# Encode chess position to 18 x 8 x 8 binary matrix
# Planes 0-5:   Friendly pieces
# Planes 6-11:  Enemy pieces
# Plane 12:     Side to move
# Planes 13-16: Castling rights
# Plane 17:     Attacked squares
def board_to_tensor(board: chess.Board) -> np.ndarray:
    tensor = np.zeros((18, 8, 8), dtype=np.float32)
     
    us = board.turn
    them = not us

    # Piece positions relative to active turn
    for plane_idx, piece_type in enumerate(PIECE_TYPES):
        for square in board.pieces(piece_type, us):
            rank, file = divmod(square, 8)
            tensor[plane_idx, rank, file] = 1.0

        for square in board.pieces(piece_type, them):
            rank, file = divmod(square, 8)
            tensor[plane_idx + 6, rank, file] = 1.0
    
    # Side to move
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    # Castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0

    # Attacked pieces
    for square in chess.SQUARES:
        if board.is_attacked_by(us, square):
            rank, file = divmod(square, 8)
            tensor[17, rank, file] = 1.0

    return tensor

# Encode a move into an integer between 0 and 4095
def move_to_index(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square

# Class to store the chess games retrieved from lichess API
class ChessGameDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.Y = torch.tensor(np.array(Y), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def fetch_and_prepare_dataset(username: str) -> ChessGameDataset:
    token = os.getenv("LICHESS_API_TOKEN")
    
    if not token:
        print("No Lichess API token found")
        sys.exit(1)

    session = berserk.TokenSession(token) 
    client = berserk.Client(session=session)
    clean_username = username.strip().lower()

    print(f"Fetching all games for {clean_username}")

    X, Y = [], []
    move_tree = {}

    try:
        export_kwargs = {
                "as_pgn": True
        }

        games_pgn_data = client.games.export_by_player(
                clean_username,
                **export_kwargs
        )
        
        if isinstance(games_pgn_data, str):
            full_pgn_text = games_pgn_data
        else:
            full_pgn_text = "".join(list(games_pgn_data))
        
        pgn_stream = io.StringIO(full_pgn_text)
        
        game_count = 0
        while True: 
            game = chess.pgn.read_game(pgn_stream)
            if game is None:
                break

            game_count += 1
            
            white = game.headers.get("White", "").strip().lower()
            black = game.headers.get("Black", "").strip().lower()

            if white == clean_username:
                playing_as = chess.WHITE
            elif black == clean_username:
                playing_as = chess.BLACK
            else:
                continue

            board = game.board()
            for move in game.mainline_moves():
                if board.turn == playing_as:
                    X.append(board_to_tensor(board))
                    Y.append(move_to_index(move))
                
                # Here we add moves / frequencies to our move tree
                state_key = board.epd()
                move_uci = move.uci()
                if state_key not in move_tree:
                    move_tree[state_key] = {}

                move_tree[state_key][move_uci] = (
                        move_tree[state_key].get(move_uci, 0) + 1
                )

                board.push(move)


        print(f"Finished. Total games processed: {game_count}")
        return ChessGameDataset(X,Y), move_tree

    except Exception as e:
        print(f"Error fetching data: {e}")
        return ChessGameDataset([], []), {}


# Retrieve the rapid elo / rating deviation for player for now
# TODO: Later we should specify time controls for training data
# and fallback elo..
def get_player_rating_data(username: str) -> tuple[int,int]:
    token = os.getenv("LICHESS_API_TOKEN")
    session = berserk.TokenSession(token) if token else None
    client =  berserk.Client(session=session)
    clean_username = username.strip().lower()

    try:
        user_data = client.users.get_public_data(clean_username)
        
        rapid_perfs = user_data.get("perfs", {}).get("rapid", {})

        rating = int(rapid_perfs.get("rating", 1500))
        rd = int(rapid_perfs.get("rd", 50))

        return rating, rd 

    except Exception as e:
        print(f"Could not retrieve rating: {e}")
        return 1500, 50

# Get games for pre-training within elo bracket
def fetch_elo_bracket_dataset(
        elo: int, 
        rd: int, 
        total_games: int = 1000):
    
    parquet_url = "https://huggingface.co/datasets/Lichess/standard-chess-games/resolve/main/data/year=2025/month=01/train-00000-of-00072.parquet"

    min_elo = min(elo - rd, 400)
    max_elo = max(elo + rd, 3000)

    query = f"""
        SELECT MoveText, WhiteElo, BlackElo
        FROM read_parquet('{parquet_url}')
        WHERE (WhiteElo BETWEEN {min_elo} AND {max_elo})
           OR (BLACKElo BETWEEN {min_elo} AND {max_elo})
        LIMIT {total_games}
    """

    try:
        conn = duckdb.connect()
        conn.execute("INSTALL httpfs; LOAD httpfs;")
        results = conn.execute(query).fetchall()
    except Exception as e:
        print(f"DuckDB query failed: {e}")
        sys.exit(1)

    X_base, Y_base = [], []
    games_collected = 0

    for row in results:
        movetext, white_elo, black_elo = row[0], row[1], row[2]
        if not movetext:
            continue


        game = chess.pgn.read_game(io.StringIO(movetext))
        if game is None:
            continue


        board = game.board()
        for move in game.mainline_moves():
            X_base.append(board_to_tensor(board))
            Y_base.append(move_to_index(move))

            board.push(move)

        games_collected += 1

    print("[Pre-Training] Collected pre-training games successfully")
    print(f"Total games collected: {games_collected}")
    return ChessGameDataset(X_base, Y_base)
