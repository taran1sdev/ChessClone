import io
import os
import sys
import berserk 
import chess
import chess.pgn
import numpy as np
import torch
from torch.utils.data import Dataset

# Connect to lichess API and convert PGNs to 3D arrays for use with tensor

PIECE_TO_PLANE = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
}

LICHESS_LAUNCH_EPOCH_MS = 1276992001000

# Encode chess position to (8, 8, 12) binary array
def board_to_tensor(board: chess.Board) -> np.ndarray:
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        plane = PIECE_TO_PLANE[piece.piece_type]
        
        if piece.color == chess.BLACK:
            plane += 6
        tensor[plane, row, col] = 1.0
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
                board.push(move)

            game_count += 1
            if game_count % 100 == 0:
                print(f"Processed {game_count} games...")

            print(f"Finished. Total games processed: {game_count}")
            return ChessGameDataset(X,Y)

    except Exception as e:
        print(f"Error fetching data: {e}")
        return ChessGameDataset([], [])


# Retrieve the rapid elo for player for now
# TODO: Later we should specify time controls for training data
# and fallback elo..
def get_player_rapid_elo(username: str) -> int:
    token = os.getenv("LICHESS_API_TOKEN")
    session = berserk.TokenSession(token) if token else None
    client =  berserk.Client(session=session)
    clean_username = username.strip().lower()

    try:
        user_data = client.users.get_public_data(clean_username)
        
        rapid_perfs = user_data.get("perfs", {}).get("rapid", {})

        rating = rapid_perfs.get("rating", 1500)
        return int(rating)

    except Exception as e:
        print(f"Could not retrieve rating: {e}")
        return 1500
