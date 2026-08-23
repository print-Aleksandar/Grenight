import random
from domain.pieces import Knight, Bishop, Queen, Rook, PIECES_NUMBERS
from domain.requests import AgentMoveRequest, MoveRequest
from domain.responses import MoveResponse
from application.game_service import gather_valid_moves_player, make_move


def play_random_valid_move(request: AgentMoveRequest) -> MoveResponse:

    if request.is_current_move_promotion:
        for piece in request.pieces:
            if piece.can_implement_pawn_moves and piece.is_current_move_pawn_promotion:
                random_cls = random.choice([Knight, Bishop, Rook, Queen])
                move_request = MoveRequest(pieces=request.pieces,
                                           uid=piece.uid,
                                           position=piece.position,
                                           is_white_on_turn=request.is_for_white_turn,
                                           is_from_white_player=request.is_for_white,
                                           is_current_move_promotion=True,
                                           promote_to=PIECES_NUMBERS[random_cls])

                return make_move(move_request)


    uids_with_valid_moves = gather_valid_moves_player(request)

    list_with_tuples_uid_valid_move = []
    [list_with_tuples_uid_valid_move.extend([(uid, valid_move) for valid_move in valid_moves])
    for uid, valid_moves in uids_with_valid_moves.items()]

    random_move_choice = random.choice(list_with_tuples_uid_valid_move)
    uid, position = random_move_choice

    move_request = MoveRequest(pieces=request.pieces,
                               uid=uid,
                               position=position,
                               is_white_on_turn=request.is_for_white_turn,
                               is_from_white_player=request.is_for_white,
                               is_current_move_promotion=False,
                               promote_to=None)

    return make_move(move_request)
