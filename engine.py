import chess
import chess.engine
import torch
import numpy as np
from data_loader import board_to_tensor, move_to_index

class AIPlayerClone:
    def __init__(self, 
                 model, 
                 fallback_elo, 
                 stockfish_path: str = "stockfish"):
        self.model = model
        self.model.eval()
        self.fallback_elo = fallback_elo
        self.stockfish_path = stockfish_path

    def get_move(self, board: chess.Board) -> chess.Move:
        tensor = torch.tensor(board_to_tensor(board)).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(tensor).squeeze(0)

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        best_move = None
        best_score = float('-inf')

        # Compare model to legal moves in position
        for move in legal_moves:
            idx = move_to_index(move)
            score = logits[idx].item()
            if score > best_score:
                best_score = score
                best_move = move

        # Choose AI move if it is valid
        if best_move is not None:
            print(f"AI move: {best_move}. Confidence score: {best_score}")
            return best_move

        print(f"Falling back to stockfish")
        return self._stockfish_fallback(board)

    def _stockfish_fallback(self, board: chess.Board) -> chess.Move:
        try:
            with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
                engine.configure({
                    "UCI_LimitStrength": True,
                    "UCI_Elo": self.fallback_elo
                })

                result = engine.play(board, chess.engine.Limit(time=0.5))
                return result.move
        except Exception as e:
            print(f"Stockfish execution failed: {e}")
            # Just default to first legal move in this case
            return list(board.legal_moves)[0]

