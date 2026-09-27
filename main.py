import chess
from data_loader import (
        fetch_and_prepare_dataset, 
        get_player_rating_data,
        fetch_elo_bracket_dataset 
)

from trainer import train_clone_model
from engine import AIPlayerClone
from torch.utils.data import ConcatDataset

def main():
    username = input("Enter lichess username: ").strip()
    
    elo, rd = get_player_rating_data(username) 
    
    base_dataset = fetch_elo_bracket_dataset(elo, rd)
    user_dataset, move_tree = fetch_and_prepare_dataset(username)
    
    combined_dataset = ConcatDataset([base_dataset, user_dataset])

    model = train_clone_model(combined_dataset, epochs=5)

    clone_engine = AIPlayerClone(
            model, 
            move_tree=move_tree, 
            fallback_elo=elo)

    board = chess.Board()
    user_color = chess.WHITE

    print("\n--- Game Started! You are White ---")
    while not board.is_game_over():
        print("\n" + str(board))
        if board.turn == user_color:
            move_uci = input("\nYour move (e.g. e2e4): ").strip()
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Illegal move! Try again.")
            except ValueError:
                print("Invalid UCI format.")
        else:
            print("\nAI Clone thinking...")
            ai_move = clone_engine.get_move(board)
            board.push(ai_move)
            print(f"AI Clone played: {ai_move}")

    print("\nGame Over! Result: " + board.result())

if __name__ == "__main__":
    main()
