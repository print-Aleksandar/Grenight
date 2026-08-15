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
        this.is_white = is_white === "true" ? true : false;
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
const restart_game_button = document.getElementById('restart_game_button');
const board = document.getElementById('board');

const squares = [];
for (let i = 4; i >= 0; --i) {
    for (let j = 0; j < 4; ++j) {
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


/* FRONTEND FUNCTIONALITIES */
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

// RESTART GAME:
async function restart_game() {

    try {
        const response = await fetch("http://localhost:8000/api/get_initial_board");
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data = await response.json();
        return render_board(data.pieces);

    } catch (error) {
        console.error("restart_game failed:", error);
    }
}

// SQUARE CLICKS:
let first_square_clicked = null;

async function first_square_click(event) {

    const squareDiv = event.currentTarget;
    first_square_clicked = squareDiv;
    squareDiv.classList.add("first_click")
}

async function make_move(firstSquareDiv, secondSquareDiv) {

    const pieceDataset = firstSquareDiv.firstChild.dataset;

    const y = Number(secondSquareDiv.dataset.row);
    const x = Number(secondSquareDiv.dataset.column);
    const position = [y, x]
    const uid = pieceDataset.uid;

    const pieces = get_pieces();
    const piecesDto = make_dto_from_pieces(pieces);

    const url = "http://localhost:8000/api/play_turn"
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                pieces: piecesDto,
                uid: uid,
                position: position,
                is_white_on_turn: true,
                is_from_white_player: true,
                is_current_move_promotion: false,
                promote_to: null
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            console.log("Validation Error Details:", errorData.detail);
            return;
        }

        const data = await response.json();
        console.log(data.pieces);
        return render_board(data.pieces);

    } catch (error) {
        console.error("make_move failed:", error);
    }
}

async function inspect_first_click(event) {

    const secondSquareDiv = event.currentTarget;

    secondSquareDiv.classList.remove("first_click")

    if (secondSquareDiv !== first_square_clicked) {
        return make_move(first_square_clicked, secondSquareDiv);
    }

    first_square_clicked.classList.remove("first_click")
    first_square_clicked = null;
}

async function route(event) {

    if (first_square_clicked === null) {
        return first_square_click(event);
    } else {
        return inspect_first_click(event);
    }
}


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


// ADDING FUNCTIONS TO HTML ELEMENTS:
squaresDivs.forEach(squareDiv => {
    squareDiv.addEventListener("click", route);
});

restart_game_button.addEventListener("click", restart_game);
