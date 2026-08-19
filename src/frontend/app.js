// CONFIGS:
const API_BASE = "http://localhost:8000";
const BOARD_BOUNDS = { rows: 5, cols: 4 };

const CLASS_NUMBERS = {
    king: 5,
    queen: 4,
    rook: 3,
    bishop: 2,
    knight: 1,
    pawn: 0
};

// TEMPORARY DATA HOLDERS (BACKEND WILL PASS THIS):
let is_white_on_turn = null;
let is_from_white_player = null;
let is_board_disabled = false;
let pawn_to_promote = null;
let is_playing_against_agent = null;
let is_current_game_finished = null;


// CLASSES:
class Square {

    constructor(y, x, is_white) {
        this.y = y;
        this.x = x;
        this.is_white = is_white;
    }
}

class Piece {

    constructor(uid, is_white, row, column, had_first_move, class_number) {
        this.uid = uid;
        this.is_white = is_white === "true";
        this.row = Number(row);
        this.column = Number(column);
        this.had_first_move = Boolean(had_first_move);
        this.class_number = Number(class_number);
    }
}

class PieceDTO {

    constructor(uid, is_white, position, had_first_move, class_number) {
        this.uid = uid;
        this.is_white = is_white;
        this.position = position;
        this.had_first_move = had_first_move;
        this.class_number = class_number;
    }
}


// CONSTANT HTML ELEMENTS:
const restart_pvp_button = document.getElementById('restart_pvp');
const restart_pve_button = document.getElementById('restart_pve');
const board = document.getElementById('board');
const board_wrapper = document.getElementById('board_wrapper');
const status = document.getElementById('status');
const promotionModalWrapper = document.getElementById('promotion_modal_wrapper');
const promotionSquaresDivs = document.querySelectorAll('.promotion_square');

const squares = [];
for (let i = BOARD_BOUNDS.rows - 1; i >= 0; --i) {
    for (let j = 0; j < BOARD_BOUNDS.cols; ++j) {
        squares.push(new Square(i, j, (i + j) % 2 === 0));
    }
}

squares.forEach(square => {

    const div = document.createElement('div');
    div.className = `square ${square.is_white ? 'white' : 'black'}`;
    div.dataset.row = square.y;
    div.dataset.column = square.x;
    board.append(div);
});

const squaresDivs = document.querySelectorAll(".square");


/* FUNCTIONALITIES: START */
// RENDER BOARD:
async function render_board(pieces) {

    squaresDivs.forEach(square => square.innerHTML = '');

    pieces.forEach(piece => {
        const [y, x] = piece.position;
        const targetSquare = document.querySelector(`.square[data-row="${y}"][data-column="${x}"]`);

        if (targetSquare) {
            const div = document.createElement("div");
            div.className = "piece";

            div.dataset.uid = piece.uid;
            div.dataset.is_white = piece.is_white;
            div.dataset.row = piece.position[0];
            div.dataset.column = piece.position[1];
            div.dataset.had_first_move = piece.had_first_move;
            div.dataset.class_number = piece.class_number;

            const color = piece.is_white ? "white_" : "black_";
            div.style.backgroundImage = `url("figures/${color}${piece.class_number}.png")`;
            targetSquare.append(div);
        }
    });
}

// SQUARE CLICK:
let firstSquareDiv = null;
let current_valid_move_positions = [];

// REGISTRY CLICK:
async function square_click(event) {

    const squareDiv = event.currentTarget;

    if (firstSquareDiv === null) {
        return handle_first_click(squareDiv);
    }
    return handle_second_click(squareDiv);
}

// FIRST SQUARE:
async function handle_first_click(squareDiv) {

    if (squareDiv.children.length === 0) {
        return;
    }

    const pieceDiv = squareDiv.firstChild;
    const piece_is_white = pieceDiv.dataset.is_white === "true";

    if (piece_is_white !== is_white_on_turn) {
        return;
    }

    firstSquareDiv = squareDiv;
    firstSquareDiv.classList.add("first_click");

    const uid = pieceDiv.dataset.uid;
    current_valid_move_positions = await fetch_valid_moves_for_piece(uid);
    highlight_squares(current_valid_move_positions, "valid_move");
}

// SECOND SQUARE:
async function handle_second_click(squareDiv) {

    if (squareDiv === firstSquareDiv) {
        deselect();
        return;
    }

    const hasPiece = squareDiv.children.length > 0;
    const firstIsWhite = firstSquareDiv.firstChild.dataset.is_white === "true";
    const clickedIsWhite = hasPiece ? squareDiv.firstChild.dataset.is_white === "true" : null;

    if (hasPiece && clickedIsWhite === firstIsWhite) {
        deselect();
        return handle_first_click(squareDiv);
    }

    return attempt_move(firstSquareDiv, squareDiv);
}

function enable_board() {

    squaresDivs.forEach(squareDiv => {
        squareDiv.addEventListener("click", square_click);
    })

    board_wrapper.classList.remove("disabled");
}

function disable_board() {

    squaresDivs.forEach(squareDiv => {
        squareDiv.removeEventListener("click", square_click);
    })
    board_wrapper.classList.add("disabled");
}

function render_promotion_div(is_white) {

    disable_board()

    promotionSquaresDivs.forEach(promotionSquareDiv => {
        promotionSquareDiv.innerHTML = "";
    });

    const color = is_white ? "white_" : "black_";

    promotionSquaresDivs.forEach(promotionSquareDiv => {

        const div = document.createElement('div');
        div.className = "promotion_piece";
        const figureNumber = CLASS_NUMBERS[promotionSquareDiv.dataset.figure];
        div.style.backgroundImage = `url("figures/${color}${figureNumber}.png")`;
        promotionSquareDiv.appendChild(div);
    });
}

function show_promotion_modal() {
    promotionModalWrapper.style.display = "block";
    disable_board();
    render_promotion_div(is_white_on_turn);
}

function hide_promotion_modal() {
    promotionModalWrapper.style.display = "none";
    enable_board();
    pawn_to_promote = null;
}

function end_game_if_finished(is_game_finished, is_draw, is_white_winner) {

    if (!is_game_finished) {
        return;
    }

    status.innerText = (is_draw ? "DRAW" : (is_white_winner ? "WHITE" : "BLACK"));

    disable_board();
    is_board_disabled = true;
    is_current_game_finished = true;
}

function restart_game_registry(event) {

    const btn = event.currentTarget;

    if (btn === restart_pvp_button) {
        is_playing_against_agent = false;
    } else {
        is_playing_against_agent = true;
    }

    return restart_game();
}
/* FUNCTIONALITIES: END */


/* API CALLS: START */
// RESTARTING GAME:
async function restart_game() {

    is_current_game_finished = false
    status.innerText = "";
    if (!is_board_disabled) {
        disable_board();
    }
    enable_board();
    hide_promotion_modal();
    is_board_disabled = false;
    firstSquareDiv = null;
    current_valid_move_positions = [];
    is_white_on_turn = true;
    is_from_white_player = true;
    is_board_disabled = false;

    try {
        const response = await fetch(`${API_BASE}/api/get_initial_board`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data = await response.json();
        await render_board(data.pieces);

        is_white_on_turn = data.is_white_on_turn;
        is_from_white_player = data.is_white_on_turn;

    } catch (error) {
        console.error("restart_game failed:", error);
    }
}

// FETCH VALID MOVES FOR A PIECE (first click):
async function fetch_valid_moves_for_piece(uid) {

    const pieces = get_pieces();
    const piecesDto = make_dto_from_pieces(pieces);

    try {
        const response = await fetch(`${API_BASE}/api/get_piece_valid_moves`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                pieces: piecesDto,
                uid: uid,
                is_from_white_player: is_white_on_turn,
                is_white_on_turn: is_white_on_turn,
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            console.warn("get_piece_valid_moves exception:", errorData.detail);
            return [];
        }

        const data = await response.json();

        return data.valid_moves || [];

    } catch (error) {
        console.error("fetch_valid_moves_for_piece failed:", error);
        return [];
    }
}

// MAKING MOVE:
async function attempt_move(firstSquareDivArg, secondSquareDiv) {

    const pieceDiv = firstSquareDivArg.firstChild;
    const pieceDataset = pieceDiv.dataset;
    const uid = pieceDataset.uid;

    const y = Number(secondSquareDiv.dataset.row);
    const x = Number(secondSquareDiv.dataset.column);
    const position = [y, x];

    const pieces = get_pieces();
    const piecesDto = make_dto_from_pieces(pieces);

    clear_valid_move_highlights();

    try {
        const response = await fetch(`${API_BASE}/api/play_turn`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                pieces: piecesDto,
                uid: uid,
                position: position,
                is_white_on_turn: is_white_on_turn,
                is_from_white_player: is_white_on_turn,
                is_current_move_promotion: false,
                promote_to: null
            })
        });

        const originSquare = firstSquareDiv;
        firstSquareDiv.classList.remove("first_click");
        firstSquareDiv = null;

        if (!response.ok) {

            const errorData = await response.json();
            console.error("Validation Error Details:", errorData.detail);

            if (response.status === 409) {

                if (errorData.detail === "PiecePinnedException") {

                    const squares = [find_king_square(is_white_on_turn), originSquare];
                    flash_red(squares);

                } else if (errorData.detail === "NonExistentValidMoveException") {

                    triggerShake();
                }
            }

            return;
        }

        const data = await response.json();

        await render_board(data.pieces);

        if (data.is_enemy_in_check) {

            const enemyKingSquare = find_king_square(!is_white_on_turn);
            flash_red([enemyKingSquare]);
        }

        end_game_if_finished(data.is_game_finished, data.is_draw, data.is_white_winner);

        if (data.is_next_move_promotion === true) {
            pawn_to_promote = secondSquareDiv;
            return show_promotion_modal();
        }

        is_white_on_turn = !is_white_on_turn;
        is_from_white_player = !is_from_white_player;

        if (is_playing_against_agent) {
            setTimeout(() => {
                return call_agent_for_move();
            }, 500);
        }

    } catch (error) {
        console.error("attempt_move failed:", error);
    }
}

async function promote_pawn(squareDivPromotionPieceEvent) {

    if (!pawn_to_promote) {
        return
    }

    const squareDivPromotionPiece = squareDivPromotionPieceEvent.currentTarget;

    const pieceDiv = pawn_to_promote.firstChild;
    const pieceDataset = pieceDiv.dataset;
    const uid = pieceDataset.uid;

    const y = Number(pawn_to_promote.dataset.row);
    const x = Number(pawn_to_promote.dataset.column);
    const position = [y, x];

    const pieces = get_pieces();
    const piecesDto = make_dto_from_pieces(pieces);

    const promote_to = CLASS_NUMBERS[squareDivPromotionPiece.dataset.figure];

    try {
        const response = await fetch(`${API_BASE}/api/play_turn`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                pieces: piecesDto,
                uid: uid,
                position: position,
                is_white_on_turn: is_white_on_turn,
                is_from_white_player: is_white_on_turn,
                is_current_move_promotion: true,
                promote_to: promote_to,
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            console.error("Validation Error Details:", errorData.detail);
            return;
        }

        const data = await response.json();

        await render_board(data.pieces);

        if (data.is_enemy_in_check) {

            const enemyKingSquare = find_king_square(!is_white_on_turn);
            flash_red([enemyKingSquare]);
        }

        hide_promotion_modal();

        end_game_if_finished(data.is_game_finished, data.is_draw, data.is_white_winner);

        is_white_on_turn = !is_white_on_turn;
        is_from_white_player = !is_from_white_player;

        if (is_playing_against_agent) {
            setTimeout(() => {
                return call_agent_for_move();
            }, 500);
        }

    } catch (error) {
        console.error("promote_pawn failed:", error);
    }
}

async function call_agent_for_move() {

    if (is_current_game_finished) {
        return;
    }

    const pieces = get_pieces();
    const piecesDto = make_dto_from_pieces(pieces);

    try {
        const response = await fetch(`${API_BASE}/api/agent_move`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                pieces: piecesDto,
                is_for_white: is_white_on_turn,
                is_white_on_turn: is_white_on_turn,
                is_current_move_promotion: false
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            console.error("Validation Error Details:", errorData.detail);
            return;
        }

        const data = await response.json();

        await render_board(data.pieces);

        if (data.is_enemy_in_check) {

            const enemyKingSquare = find_king_square(!is_white_on_turn);
            flash_red([enemyKingSquare]);
        }

        if (data.is_next_move_promotion === true) {
            setTimeout(() => {
                return call_agent_for_promotion();
            }, 500);
        }

        end_game_if_finished(data.is_game_finished, data.is_draw, data.is_white_winner);

        is_white_on_turn = !is_white_on_turn;
        is_from_white_player = !is_from_white_player;

    } catch (error) {
        console.error("call_agent_for_move failed:", error);
    }
}

async function call_agent_for_promotion() {

    const pieces = get_pieces();
    const piecesDto = make_dto_from_pieces(pieces);

    try {
        const response = await fetch(`${API_BASE}/api/agent_move`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                pieces: piecesDto,
                is_for_white: is_white_on_turn,
                is_white_on_turn: is_white_on_turn,
                is_current_move_promotion: true
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            console.error("Validation Error Details:", errorData.detail);
            return;
        }

        const data = await response.json();

        await render_board(data.pieces);

        if (data.is_enemy_in_check) {

            const enemyKingSquare = find_king_square(!is_white_on_turn);
            flash_red([enemyKingSquare]);
        }

        end_game_if_finished(data.is_game_finished, data.is_draw, data.is_white_winner);

        is_white_on_turn = !is_white_on_turn;
        is_from_white_player = !is_from_white_player;

    } catch (error) {
        console.error("call_agent_for_move failed:", error);
    }
}
/* API CALLS: END */


// ADDING FUNCTIONS TO HTML ELEMENTS:
restart_pvp_button.addEventListener("click", restart_game_registry);
restart_pve_button.addEventListener("click", restart_game_registry);

promotionSquaresDivs.forEach((promotionSquareDiv) => {
    promotionSquareDiv.addEventListener("click", promote_pawn);
});


/* HIGHLIGHTING: START */
function highlight_squares(positions, className) {
    positions.forEach(([y, x]) => {
        const squareDiv = document.querySelector(`.square[data-row="${y}"][data-column="${x}"]`);
        if (squareDiv) squareDiv.classList.add(className);
    });
}

function clear_valid_move_highlights() {
    squaresDivs.forEach(s => s.classList.remove("valid_move"));
}

function deselect() {
    clear_valid_move_highlights();
    firstSquareDiv.classList.remove("first_click");
    firstSquareDiv = null;
    current_valid_move_positions = [];
}

function flash_red(squareDivs) {
    squareDivs.forEach(s => s && s.classList.add("red_highlight"));
    setTimeout(() => {
        squareDivs.forEach(s => s && s.classList.remove("red_highlight"));
    }, 1500);
}

function triggerShake() {
    board_wrapper.classList.add('shake-active');
    setTimeout(() => {
        board_wrapper.classList.remove('shake-active');
    }, 500);
}
/* HIGHLIGHTING: END */


// HELPERS:
function get_pieces() {

    const piecesObjs = []
    document.querySelectorAll(".piece").forEach(piece => {
        piecesObjs.push(new Piece(piece.dataset.uid, piece.dataset.is_white, piece.dataset.row,
            piece.dataset.column, piece.dataset.had_first_move, piece.dataset.class_number));
    })

    return piecesObjs;
}

function make_dto_from_pieces(pieces) {

    const piecesDto = []
    for (const piece of pieces) {
        piecesDto.push(new PieceDTO(
            piece.uid, piece.is_white, [piece.row, piece.column],
            piece.had_first_move, piece.class_number
        ));
    }

    return piecesDto;
}

function find_king_square(is_white) {

    const kingDiv = document.querySelector(
        `.piece[data-class_number="${CLASS_NUMBERS.king}"][data-is_white="${is_white}"]`
    );

    return kingDiv ? kingDiv.parentElement : null;
}