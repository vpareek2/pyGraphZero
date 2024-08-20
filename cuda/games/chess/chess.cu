#include "chess.cuh"

// Initialize the chess game state
__host__ __device__ void chess_init(IGame* self) {

    if (self == NULL) {
    // Handle error: self or board is NULL
        return;
    }
    // Initialize pieces
    for (int i = 0; i < CHESS_BOARD_SIZE; i++) {
        board->pieces[i] = EMPTY;
    }

    // Set up initial piece positions
    board->pieces[0] = board->pieces[7] = ROOK * WHITE;
    board->pieces[1] = board->pieces[6] = KNIGHT * WHITE;
    board->pieces[2] = board->pieces[5] = BISHOP * WHITE;
    board->pieces[3] = QUEEN * WHITE;
    board->pieces[4] = KING * WHITE;

    for (int i = 8; i < 16; i++) {
        board->pieces[i] = PAWN * WHITE;
    }

    for (int i = 48; i < 56; i++) {
        board->pieces[i] = PAWN * BLACK;
    }

    board->pieces[56] = board->pieces[63] = ROOK * BLACK;
    board->pieces[57] = board->pieces[62] = KNIGHT * BLACK;
    board->pieces[58] = board->pieces[61] = BISHOP * BLACK;
    board->pieces[59] = QUEEN * BLACK;
    board->pieces[60] = KING * BLACK;

    // Initialize other game state variables
    board->player = WHITE;
    board->castling_rights[0][0] = board->castling_rights[0][1] = true;
    board->castling_rights[1][0] = board->castling_rights[1][1] = true;
    board->en_passant_target = -1;
    board->halfmove_clock = 0;
    board->fullmove_number = 1;
}

// Set up the initial chess board configuration
__host__ __device__ void chess_get_init_board(const IGame* self, int* board) {
    if (self == NULL || board == NULL) {
        // Handle error: self or board is NULL
        return;
    }

    const ChessGame* chess = (const ChessGame*)self;
    if (chess == NULL) {
        // Handle error: invalid cast
        return;
    }

    for (int i = 0; i < CHESS_BOARD_SIZE; i++) {
        board[i] = chess->board.pieces[i];
    }
}

// Return the dimensions of the chess board (8x8)
__host__ __device__ void chess_get_board_size(const IGame* self, int* rows, int* cols) {
    if (rows == NULL || cols == NULL) {
        // Handle error: rows or cols is NULL
        return;
    }

    *rows = 8;
    *cols = 8;
}

// Return the total number of possible moves in chess (e.g., 64 * 73 for all possible start and end squares including promotions)
__host__ __device__ int chess_get_action_size(const IGame* self) {
    // In chess, there are 64 possible starting squares and 73 possible target squares
    // (including the 9 possible underpromotion moves for pawns)
    return 64 * 73;
}

__host__ __device__ void chess_get_next_state(const IGame* self, const int* board, int player, int action, int* next_board, int* next_player) {
    if (self == NULL || board == NULL || next_board == NULL || next_player == NULL) {
        // Handle error: invalid input
        return;
    }
    
    const ChessGame* chess = (const ChessGame*)self;
    if (chess == NULL) {
        // Handle error: invalid cast
        return;
    }
    
    // Copy the current board state to the next board
    for (int i = 0; i < CHESS_BOARD_SIZE; i++) {
        next_board[i] = board[i];
    }
    
    // Create a temporary ChessBoard to work with
    ChessBoard temp_board = chess->board;
    for (int i = 0; i < CHESS_BOARD_SIZE; i++) {
        temp_board.pieces[i] = board[i];
    }
    temp_board.player = player;
    
    // Decode the action into start and end positions
    int start = action / 73;
    int end = action % 73;
    int promotion = QUEEN;  // Default promotion piece

    // Handle promotion moves
    if (end >= 64) {
        promotion = end - 64 + KNIGHT;  // KNIGHT, BISHOP, ROOK, QUEEN
        end = start / 8 == 1 ? start + 8 : start - 8;  // Move to the last rank
    }
    
    // Get the moving piece and capture piece (if any)
    int moving_piece = temp_board.pieces[start];
    int capture_piece = temp_board.pieces[end];
    
    // Apply the basic move
    temp_board.pieces[start] = EMPTY;
    temp_board.pieces[end] = moving_piece;
    
    // Handle special moves
    int piece_type = abs(moving_piece);
    int rank_start = start / 8;
    int file_start = start % 8;
    int rank_end = end / 8;
    int file_end = end % 8;
    
    // Pawn promotion
    if (piece_type == PAWN && (rank_end == 0 || rank_end == 7)) {
        temp_board.pieces[end] = player * promotion;
    }
    
    // En passant capture
    if (piece_type == PAWN && end == temp_board.en_passant_target) {
        int captured_pawn_pos = end + (player == WHITE ? -8 : 8);
        temp_board.pieces[captured_pawn_pos] = EMPTY;
    }
    
    // Update en passant target
    if (piece_type == PAWN && abs(rank_end - rank_start) == 2) {
        temp_board.en_passant_target = (start + end) / 2;
    } else {
        temp_board.en_passant_target = -1; // Reset en passant target
    }
    
    // Castling
    if (piece_type == KING && abs(file_end - file_start) == 2) {
        int rook_start, rook_end;
        if (file_end > file_start) { // Kingside castling
            rook_start = start + 3;
            rook_end = start + 1;
        } else { // Queenside castling
            rook_start = start - 4;
            rook_end = start - 1;
        }
        temp_board.pieces[rook_end] = temp_board.pieces[rook_start];
        temp_board.pieces[rook_start] = EMPTY;
    }
    
    // Update castling rights
    int player_index = player == WHITE ? 0 : 1;
    if (piece_type == KING) {
        temp_board.castling_rights[player_index][0] = false;
        temp_board.castling_rights[player_index][1] = false;
    } else if (piece_type == ROOK) {
        if (start == 0 || start == 56) // Queenside rook
            temp_board.castling_rights[player_index][1] = false;
        else if (start == 7 || start == 63) // Kingside rook
            temp_board.castling_rights[player_index][0] = false;
    }
    
    // Update halfmove clock
    if (piece_type == PAWN || capture_piece != EMPTY)
        temp_board.halfmove_clock = 0;
    else
        temp_board.halfmove_clock++;
    
    // Update fullmove number
    if (player == BLACK)
        temp_board.fullmove_number++;
    
    // Update the player
    temp_board.player = -player;
    *next_player = -player;
    
    // Copy the updated chess board back to the next_board array
    for (int i = 0; i < CHESS_BOARD_SIZE; i++) {
        next_board[i] = temp_board.pieces[i];
    }
}

__host__ __device__ void chess_get_valid_moves(const IGame* self, const int* board, int player, bool* valid_moves) {
    const ChessGame* chess_game = (const ChessGame*)self;
    ChessBoard temp_board;

    // Copy the board state to a temporary ChessBoard
    for (int i = 0; i < CHESS_BOARD_SIZE; i++) {
        temp_board.pieces[i] = board[i];
    }
    temp_board.player = player;
    temp_board.en_passant_target = chess_game->board.en_passant_target;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            temp_board.castling_rights[i][j] = chess_game->board.castling_rights[i][j];
        }
    }

    // Initialize all moves as invalid
    for (int i = 0; i < CHESS_BOARD_SIZE * 73; i++) {
        valid_moves[i] = false;
    }

    // Iterate through all pieces of the current player
    for (int start = 0; start < CHESS_BOARD_SIZE; start++) {
        int piece = temp_board.pieces[start];
        if (piece * player <= 0) continue; // Skip empty squares and opponent's pieces

        // Generate legal moves for the current piece
        for (int end = 0; end < CHESS_BOARD_SIZE; end++) {
            if (is_legal_move(&temp_board, start, end)) {
                // Make the move
                ChessBoard next_board = temp_board;
                make_move(&next_board, start, end);

                // Check if the move leaves the player in check
                if (!is_check(&next_board, player)) {
                    valid_moves[start * 73 + end] = true;

                    // Handle pawn promotion
                    if (abs(piece) == PAWN && (end / 8 == 0 || end / 8 == 7)) {
                        valid_moves[start * 73 + end] = false; // Disable default move
                        for (int promotion = KNIGHT; promotion <= QUEEN; promotion++) {
                            valid_moves[start * 73 + (64 + promotion - KNIGHT)] = true;
                        }
                    }
                }
            }
        }

        // Handle castling
        if (abs(piece) == KING) {
            int rank = start / 8;
            if (can_castle_kingside(&temp_board, player)) {
                valid_moves[start * 73 + (start + 2)] = true;
            }
            if (can_castle_queenside(&temp_board, player)) {
                valid_moves[start * 73 + (start - 2)] = true;
            }
        }
    }
}

__host__ __device__ int chess_get_game_ended(const IGame* self, const int* board, int player) {
    const ChessGame* chess_game = (const ChessGame*)self;
    ChessBoard temp_board;

    // Copy the input board to the temporary board
    for (int i = 0; i < CHESS_BOARD_SIZE; i++) {
        temp_board.pieces[i] = board[i];
    }
    temp_board.player = player;
    temp_board.en_passant_target = chess_game->board.en_passant_target;
    temp_board.halfmove_clock = chess_game->board.halfmove_clock;
    temp_board.fullmove_number = chess_game->board.fullmove_number;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            temp_board.castling_rights[i][j] = chess_game->board.castling_rights[i][j];
        }
    }

    // Check for checkmate
    if (is_checkmate(&temp_board, player)) {
        return -player; // Return the opposite of the current player (winner)
    }

    // Check for stalemate
    if (is_stalemate(&temp_board, player)) {
        return 1e-4; // Draw
    }

    // Check for insufficient material
    if (is_insufficient_material(&temp_board)) {
        return 1e-4; // Draw
    }

    // Check for fifty-move rule
    if (is_fifty_move_rule(&temp_board)) {
        return 1e-4; // Draw
    }

    // Game hasn't ended
    return 0;
}
__host__ __device__ void chess_get_canonical_form(const IGame* self, const int* board, int player, int* canonical_board) {
    if (player == WHITE) {
        // If player is White, simply copy the board
        for (int i = 0; i < CHESS_BOARD_SIZE; i++) {
            canonical_board[i] = board[i];
        }
    } else {
        // If player is Black, flip the board vertically and negate the pieces
        for (int i = 0; i < CHESS_BOARD_SIZE; i++) {
            int flipped_index = CHESS_BOARD_SIZE - 1 - i;
            canonical_board[i] = -board[flipped_index];
        }
    }
}

// Heuristic evaluation of the board state (optional, can return 0 if not implemented)
__host__ __device__ float chess_evaluate(const IGame* self, const int* board, int player) {
    float score = 0;
    for (int i = 0; i < CHESS_BOARD_SIZE; i++) {
        int piece = board[i];
        switch (abs(piece)) {
            case PAWN:   score += 1.0f * (piece > 0 ? 1 : -1); break;
            case KNIGHT: score += 3.0f * (piece > 0 ? 1 : -1); break;
            case BISHOP: score += 3.0f * (piece > 0 ? 1 : -1); break;
            case ROOK:   score += 5.0f * (piece > 0 ? 1 : -1); break;
            case QUEEN:  score += 9.0f * (piece > 0 ? 1 : -1); break;
        }
    }
    return score * player;
}
// Generate all symmetries of the board (rotations and reflections)
void chess_get_symmetries(const IGame* self, const int* board, const float* pi, int (*symmetries)[CHESS_BOARD_SIZE], float (*symmetries_pi)[CHESS_BOARD_SIZE * 73], int* num_symmetries) {
    for (int i = 0; i < CHESS_BOARD_SIZE; i++) {
        symmetries[0][i] = board[i];
    }
    for (int i = 0; i < CHESS_BOARD_SIZE * 73; i++) {
        symmetries_pi[0][i] = pi[i];
    }
    *num_symmetries = 1;
}

// Convert the board state to a string representation
void chess_string_representation(const IGame* self, const int* board, char* str, int str_size) {
    const char* pieces = " PNBRQK pnbrqk";
    int idx = 0;
    for (int rank = 7; rank >= 0; rank--) {
        idx += snprintf(str + idx, str_size - idx, "%d ", rank + 1);
        for (int file = 0; file < 8; file++) {
            int piece = board[rank * 8 + file];
            str[idx++] = pieces[piece < 0 ? (6 - piece) : piece];
            if (idx >= str_size - 1) goto end;
            if (file < 7) str[idx++] = ' ';
            if (idx >= str_size - 1) goto end;
        }
        if (rank > 0) {
            str[idx++] = '\n';
            if (idx >= str_size - 1) goto end;
        }
    }
    idx += snprintf(str + idx, str_size - idx, "\n  a b c d e f g h");
end:
    str[idx] = '\0';
}

// Display the current board state (e.g., print to console)
void chess_display(const IGame* self, const int* board) {
    char str[256];
    chess_string_representation(self, board, str, sizeof(str));
    printf("%s\n", str);
}

// Create a new chess game instance
ChessGame* create_chess_game() {
    ChessGame* game = (ChessGame*)malloc(sizeof(ChessGame));
    if (!game) {
        fprintf(stderr, "Failed to allocate memory for ChessGame\n");
        return NULL;
    }
    game->base.init = chess_init;
    game->base.get_init_board = chess_get_init_board;
    game->base.get_board_size = chess_get_board_size;
    game->base.get_action_size = chess_get_action_size;
    game->base.get_next_state = chess_get_next_state;
    game->base.get_valid_moves = chess_get_valid_moves;
    game->base.get_game_ended = chess_get_game_ended;
    game->base.get_canonical_form = chess_get_canonical_form;
    game->base.evaluate = chess_evaluate;
    chess_init(&game->base);
    return game;
}

// Free resources associated with a chess game instance
void destroy_chess_game(ChessGame* game) {
    if (game) {
        free(game);
    }
}

/***************************************************
 * HELPERS
 **************************************************/

/__host__ __device__ bool is_check(const ChessBoard* board, int player) {
    U64 king_bb = 0, occ = 0, attacks = 0;
    int king_square = -1;

    // Set up bitboards
    for (int i = 0; i < CHESS_BOARD_SIZE; i++) {
        if (board->pieces[i] != EMPTY) {
            occ |= 1ULL << i;
            if (board->pieces[i] == player * KING) {
                king_bb = 1ULL << i;
                king_square = i;
            }
        }
    }

    if (king_square == -1) return false;  // No king found, shouldn't happen in a valid position

    // Check for pawn attacks
    attacks |= get_pawn_attacks(king_square, player);

    // Check for knight attacks
    attacks |= get_knight_attacks(king_square);

    // Check for sliding piece attacks (bishop, rook, queen)
    attacks |= get_sliding_attacks(BISHOP, king_square, occ);
    attacks |= get_sliding_attacks(ROOK, king_square, occ);

    // Check if any enemy piece is on an attacking square
    while (attacks) {
        int sq = LSB(attacks);
        if (board->pieces[sq] * player < 0) {  // Enemy piece
            int piece_type = abs(board->pieces[sq]);
            if (piece_type == PAWN || piece_type == KNIGHT ||
                (piece_type == BISHOP && (get_sliding_attacks(BISHOP, sq, occ) & king_bb)) ||
                (piece_type == ROOK && (get_sliding_attacks(ROOK, sq, occ) & king_bb)) ||
                (piece_type == QUEEN && ((get_sliding_attacks(BISHOP, sq, occ) | get_sliding_attacks(ROOK, sq, occ)) & king_bb))) {
                return true;
            }
        }
        POP_LSB(attacks);
    }

    return false;
}

__host__ __device__ bool is_checkmate(const ChessBoard* board, int player) {
    if (!is_check(board, player)) return false;

    // Try all possible moves for the player
    for (int from = 0; from < CHESS_BOARD_SIZE; from++) {
        if (board->pieces[from] * player <= 0) continue;  // Not player's piece

        U64 moves = get_attacks(board->pieces[from], from, 0);  // Pseudo-legal moves
        while (moves) {
            int to = LSB(moves);
            
            // Make the move
            ChessBoard new_board = *board;
            new_board.pieces[to] = new_board.pieces[from];
            new_board.pieces[from] = EMPTY;

            // If this move gets out of check, it's not checkmate
            if (!is_check(&new_board, player)) return false;

            POP_LSB(moves);
        }
    }

    return true;  // No legal moves found
}

__host__ __device__ bool is_stalemate(const ChessBoard* board, int player) {
    if (is_check(board, player)) return false;

    // Try all possible moves for the player
    for (int from = 0; from < CHESS_BOARD_SIZE; from++) {
        if (board->pieces[from] * player <= 0) continue;  // Not player's piece

        U64 moves = get_attacks(board->pieces[from], from, 0);  // Pseudo-legal moves
        while (moves) {
            int to = LSB(moves);
            
            // Make the move
            ChessBoard new_board = *board;
            new_board.pieces[to] = new_board.pieces[from];
            new_board.pieces[from] = EMPTY;

            // If this move doesn't result in check, it's a legal move
            if (!is_check(&new_board, player)) return false;

            POP_LSB(moves);
        }
    }

    return true;  // No legal moves found
}

// Check if there's insufficient material for checkmate
__host__ __device__ bool is_insufficient_material(const ChessBoard* board);

// Check for threefold repetition (may need additional state tracking)
__host__ __device__ bool is_threefold_repetition(const ChessBoard* board);

// Check if the fifty-move rule applies
__host__ __device__ bool is_fifty_move_rule(const ChessBoard* board);

__host__ __device__ bool is_legal_move(const ChessBoard* board, int start, int end);
__host__ __device__ void make_move(ChessBoard* board, int start, int end);
__host__ __device__ bool can_castle_kingside(const ChessBoard* board, int player);
__host__ __device__ bool can_castle_queenside(const ChessBoard* board, int player);
