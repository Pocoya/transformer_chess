# MCTS-Guided Policy-Value Transformer (Chess Engine)

A high-performance chess engine based on the **Transformer-Base** architecture, designed as a **Policy-Value Network (PV-Net)** to guide Monte Carlo Tree Search (MCTS).
## Key Features
- **Dual-Head Brain**: A single Transformer handling two jobs at once: a **Policy Head** to pick the best move and a **Value Head** to evaluate who is winning.
- **Chess-Specific Tokenization**: Custom character-level encoding built specifically to read board positions (FEN) and move sequences (UCI) perfectly.
- **Hybrid Training**: Optimized using Multi-Task Loss, combining Cross-Entropy (for move accuracy) and Mean Squared Error (for board evaluation accuracy).


## Project Structure
- `data.py`: Character-level tokenizer for FEN strings and UCI moves.
- `config.yaml`: Configuration for hyperparameters and training.
- `extract_data.py`: Extracting stockfish evaluation data.
- `model.py`: MCTS-Guided Policy-Value Transformer.
- `train.py`: Training logic and eval.
