import chess
from data_loader import fetch_and_prepare_dataset, get_player_rapid_elo
from trainer import train_clone_model
from engine import AIPlayerClone

def main():
    username = input("Enter lichess username: ").strip()
    
    rapid_elo = get_player_rapid_elo(username) 
    
    dataset = fetch_and_prepare_dataset(username)
    
    model = train_clone_model(dataset, epochs=5)

    clone_engine = AIPlayerClone(model, rapid_elo)

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
