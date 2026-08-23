import numpy as np
from domain.configs import MAX_ALLOWED_STEPS_WITHOUT_PAWN_MOVE_OR_CAPTURING
from domain.pieces import Piece, PIECES_NUMBERS
from domain.requests import MoveRequest, AgentMoveRequest
from domain.exceptions import GrenightException
from domain.board_initialization import create_initial_board
from application.game_service import make_move, gather_valid_moves_player
from application.board_getter import get_piece_by_position, get_piece_by_uid
from environment.action_encoder import ActionEncoder
from environment.piece_plane_encoder import PiecePlaneEncoder


class GrenightEnvironment:

    PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5

    def __init__(self):

        self.state_encoder = PiecePlaneEncoder()
        self.action_encoder = ActionEncoder()

        self.pieces: list[Piece] = []
        self.is_white_on_turn = True
        self.done = False

        self.current_repetition_count = 0
        self.steps_without_pawn_move_or_capture = 0

        self.position_counts: dict[tuple, int] = {}
        self.is_draw_by_rule = False
        self.draw_reason: str | None = None

    def reset(self) -> np.ndarray:

        self.pieces = create_initial_board()
        self.is_white_on_turn = True
        self.done = False
        self.steps_without_pawn_move_or_capture = 0
        self.position_counts = {}
        self.current_repetition_count = 0
        self.is_draw_by_rule = False
        self.draw_reason = None

        return self.get_state()

    def get_state(self) -> np.ndarray:

        return self.state_encoder.encode_planes(
            pieces=self.pieces,
            current_player_is_white=self.is_white_on_turn,
            steps_without_progress=self.steps_without_pawn_move_or_capture,
            max_steps_without_progress=MAX_ALLOWED_STEPS_WITHOUT_PAWN_MOVE_OR_CAPTURING,
            repetition_count=self.current_repetition_count,
            repetition_limit=3,
        )

    def legal_actions(self) -> list[int]:

        if self.done:
            return []

        request = AgentMoveRequest(
            pieces=self.pieces,
            is_for_white=self.is_white_on_turn,
            is_for_white_turn=self.is_white_on_turn,
            is_current_move_promotion=False,
        )

        uid_to_positions = gather_valid_moves_player(request)
        actions = []

        for uid, positions in uid_to_positions.items():
            piece = get_piece_by_uid(self.pieces, uid)
            if piece is None:
                continue

            is_pawn_about_to_promote = (
                piece.can_implement_pawn_moves and piece.is_next_move_pawn_promotion
            )

            for to_position in positions:
                if is_pawn_about_to_promote:
                    for promote_to in (3, 4): # TODO: put when scaling for knight and bishop: 1, 2, 3, 4, easy to forget place
                        action = self.action_encoder.encode_promotion(
                            from_position=piece.position,
                            to_position=to_position,
                            promote_to=promote_to,
                            current_player_is_white=self.is_white_on_turn,
                        )
                        actions.append(action)
                else:
                    action = self.action_encoder.encode(
                        from_position=piece.position,
                        to_position=to_position,
                    )
                    actions.append(action)

        return actions

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_encoder.NUM_ACTIONS, dtype=bool)
        actions = self.legal_actions()
        if actions:
            mask[actions] = True
        return mask

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:

        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset() first.")

        self.steps_without_pawn_move_or_capture += 1

        legal_acts = self.legal_actions()
        if action not in legal_acts:
            raise ValueError(f"Illegal action {action}")

        acting_player_is_white = self.is_white_on_turn

        if self.action_encoder.is_promotion_action(action):
            from_position, to_position, promote_to = self.action_encoder.decode_promotion(
                action,
                current_player_is_white=acting_player_is_white,
            )
            is_promotion = True

        else:
            from_position, to_position = self.action_encoder.decode(action)
            promote_to = None
            is_promotion = False

        piece = get_piece_by_position(self.pieces, from_position)
        if piece is None:
            raise ValueError(f"Illegal action {action}: no piece at {from_position}")

        if piece.is_white != acting_player_is_white:
            raise ValueError(f"Illegal action {action}: piece belongs to the other player")

        if piece.can_implement_pawn_moves:
            self.steps_without_pawn_move_or_capture = 0

        request = MoveRequest(
            pieces=self.pieces,
            uid=piece.uid,
            position=to_position,
            is_white_on_turn=self.is_white_on_turn,
            is_from_white_player=acting_player_is_white,
            is_current_move_promotion=False,
            promote_to=None,
        )

        try:
            response = make_move(request)
        except GrenightException as e:
            raise ValueError(f"Action {action} rejected by make_move: {type(e).__name__}")

        if is_promotion:
            request = MoveRequest(
                pieces=response.pieces,
                uid=piece.uid,
                position=to_position,
                is_white_on_turn=response.is_white_on_turn,
                is_from_white_player=acting_player_is_white,
                is_current_move_promotion=True,
                promote_to=promote_to,
            )

            try:
                response = make_move(request)
            except GrenightException as e:
                raise ValueError(f"Action {action} rejected by make_move: {type(e).__name__}")

        if len(response.pieces) < len(self.pieces):
            self.steps_without_pawn_move_or_capture = 0

        self.pieces = response.pieces
        self.is_white_on_turn = response.is_white_on_turn
        self.done = response.is_game_finished
        self.is_draw_by_rule = False
        self.draw_reason = None

        if not self.done:

            key = self.position_key()
            self.current_repetition_count = self.position_counts.get(key, 0) + 1
            self.position_counts[key] = self.current_repetition_count

            if self.steps_without_pawn_move_or_capture >= MAX_ALLOWED_STEPS_WITHOUT_PAWN_MOVE_OR_CAPTURING:
                self.done = True
                self.is_draw_by_rule = True
                self.draw_reason = "max_steps_without_progress"

            elif self.is_insufficient_material():
                self.done = True
                self.is_draw_by_rule = True
                self.draw_reason = "insufficient_material"

            elif self.current_repetition_count >= 3:
                self.done = True
                self.is_draw_by_rule = True
                self.draw_reason = "threefold_repetition"

        else:
            if response.is_game_finished and response.is_draw:
                self.draw_reason = "stalemate"

        reward = self.calculate_reward(
            response=response,
            acting_player_is_white=acting_player_is_white,
        )

        next_state = self.get_state()
        info = {
            "is_enemy_in_check": response.is_enemy_in_check,
            "draw_reason": self.draw_reason,
        }

        return next_state, reward, self.done, info

    def sample(self) -> int:

        return np.random.choice(self.legal_actions())

    def position_key(self) -> tuple:

        pieces_key = tuple(sorted(
            (PIECES_NUMBERS[type(p)], p.is_white, p.position)
            for p in self.pieces
        ))
        return pieces_key, self.is_white_on_turn

    def is_insufficient_material(self) -> bool:

        piece_types = [PIECES_NUMBERS[type(p)] for p in self.pieces]

        if any(t in (self.PAWN, self.ROOK, self.QUEEN) for t in piece_types):
            return False
        return True

        """
        minors = [p for p in self.pieces if PIECES_NUMBERS[type(p)] in (self.KNIGHT, self.BISHOP)]

        if len(minors) == 0:
            return True

        if len(minors) == 1:
            return True

        if len(minors) == 2 and all(PIECES_NUMBERS[type(p)] == self.BISHOP for p in minors):
            square_colors = {(p.position[0] + p.position[1]) % 2 for p in minors}
            return len(square_colors) == 1
        
        return False
        """


    def calculate_reward(self, response, acting_player_is_white: bool) -> float:

        if not self.done:
            return 0.0

        if self.is_draw_by_rule or response.is_draw:
            return 0.0

        if response.is_white_winner == acting_player_is_white:
            return 1.0

        return -1.0
