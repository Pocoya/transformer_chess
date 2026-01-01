import zstandard as zstd
import json
import io
import math
import os

def cp_to_value(cp):
    """
    Standard squashing function: 
    - Turns +300 (1 pawn up) into ~0.76
    - Turns 0 (Equal) into 0.0
    - Turns -300 into -0.76
    """
    return math.tanh(cp / 400.0)

def extract_stockfish_data(input_file, output_file, num_samples=5000000):
    count = 0
    print(f"File: {input_file}...")

    with open(input_file, 'rb') as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')

            with open(output_file, 'w') as out:
                for line in text_stream:
                    try:
                        data = json.loads(line)
                        is_white_to_move = data['fen'].split()[1] == 'w'

                        best_eval_entry = max(data['evals'], key=lambda x: x.get('depth', 0))
                        pv = best_eval_entry['pvs'][0]
                        best_move = pv['line'].split()[0]
                        
                        if 'mate' in pv:
                            # If my turn and I have a Mate in X, it's +1.0
                            mate_val = pv['mate']
                            if is_white_to_move:
                                value = 1.0 if mate_val > 0 else -1.0
                            else:
                                value = 1.0 if mate_val < 0 else -1.0
                        else:
                            # Flip the score if it is Black's turn
                            cp_score = pv['cp']
                            relative_score = cp_score if is_white_to_move else -cp_score
                            value = cp_to_value(relative_score)

                        out.write(json.dumps({
                            'src': data['fen'], 
                            'tgt': best_move, 
                            'val': value
                        }) + '\n')
                        
                        count += 1
                        if count % 100000 == 0:
                            print(f"Extracted {count:,} samples...")
                        if count >= num_samples:
                            break
                            
                    except (KeyError, IndexError, ValueError):
                        continue
                    
    print(f"Saved {count:,} positions with scores to {output_file}")

extract_stockfish_data("data/lichess_db_eval.jsonl.zst", "chess_train_data.jsonl")