import numpy as np
from domain.configs import MAX_STEPS_WITHOUT_PROGRESS, ROWS, PREVIOUS_K_STEPS_IN_STATE
from domain.pieces import Piece, Pawn, PIECES_NUMBERS
from domain.requests import MoveRequest, AgentMoveRequest
from domain.exceptions import GrenightException
from domain.board_initialization import create_initial_board
from application.game_service import make_move, gather_valid_moves_player
from application.board_getter import get_piece_by_position
from environment.action_encoder import ActionEncoder
from environment.piece_plane_encoder import PiecePlaneEncoder
from environment.previous_pieces_encoded_q import PreviousPiecesEncodedQ


def rotate_pieces_helper(pieces: list[Piece]) -> None:
    for piece in pieces:
        piece.is_white = not piece.is_white

        y, x = piece.position
        piece.position = ROWS - 1 - y, x

        piece.update_after_flipping()


class GrenightEnvironment:

    PAWN, ROOK, QUEEN = 0, 1, 2
    OTHER_DRAWS = -0.5
    THREEFOLD_REPETITION_RULE_VALUE = 0.5
    ENEMY_IN_CHECK_REWARD = 0.1

    def __init__(self, is_canonical_version: bool,
                 will_store_history_in_state: bool,
                 will_do_reward_shaping: bool) -> None:

        self.is_canonical_version = is_canonical_version
        self.will_store_history_in_state = will_store_history_in_state
        self.will_do_reward_shaping = will_do_reward_shaping

        self.action_encoder = ActionEncoder(self.is_canonical_version)
        self.state_encoder = PiecePlaneEncoder(self.will_store_history_in_state)

        self.pieces = None
        self.previous_pieces_encoded_q = PreviousPiecesEncodedQ(
            PREVIOUS_K_STEPS_IN_STATE if self.will_store_history_in_state
            else 0
        )

        self.is_white_on_turn = True
        self.done = False

        self.current_repetition_count = 0
        self.steps_without_pawn_move_or_capture = 0

        self.position_counts: dict[tuple, int] = {}
        self.is_draw_by_rule = False
        self.draw_reason: str | None = None

        self._legal_actions_cache: list[int] | None = None
        self._legal_actions_set_cache: set[int] | None = None

        self._state_cache: np.ndarray | None = None

    def reset(self) -> np.ndarray:
        self.pieces = create_initial_board()
        self.is_white_on_turn = True
        self.done = False
        self.steps_without_pawn_move_or_capture = 0
        self.position_counts = {}
        self.current_repetition_count = 0
        self.is_draw_by_rule = False
        self.draw_reason = None
        self._state_cache = None
        self._invalidate_legal_actions_cache()

        key = self.position_key()
        self.current_repetition_count = self.position_counts.get(key, 0) + 1
        self.position_counts[key] = self.current_repetition_count

        state = self.get_state()
        self.previous_pieces_encoded_q.push(state[-13:-3])
        self._state_cache = state
        return state

    def get_state(self) -> np.ndarray:
        if self._state_cache is not None:
            return self._state_cache

        return self.state_encoder.encode_planes(
            previous_pieces_encoded_q=self.previous_pieces_encoded_q,
            pieces=self.pieces,
            current_player_is_white=self.is_white_on_turn,
            steps_without_progress=self.steps_without_pawn_move_or_capture,
            max_steps_without_progress=MAX_STEPS_WITHOUT_PROGRESS,
            repetition_count=self.current_repetition_count,
            repetition_limit=3,
        )

    def _invalidate_legal_actions_cache(self) -> None:
        self._legal_actions_cache = None
        self._legal_actions_set_cache = None

    def legal_actions(self) -> list[int]:

        if self._legal_actions_cache is not None:
            return self._legal_actions_cache

        if self.done:
            return []

        request = AgentMoveRequest(
            pieces=self.pieces,
            is_for_white=True if self.is_canonical_version else self.is_white_on_turn,
            is_for_white_turn=True if self.is_canonical_version else self.is_white_on_turn,
            is_current_move_promotion=False
        )

        uids_with_valid_moves = gather_valid_moves_player(request)
        uids_with_piece = {piece.uid: piece for piece in self.pieces}
        actions = []

        for uid, positions in uids_with_valid_moves.items():
            piece = uids_with_piece.get(uid)
            if piece is None:
                continue

            is_pawn_about_to_promote = (
                piece.can_implement_pawn_moves and piece.is_next_move_pawn_promotion
            )

            for to_position in positions:
                if is_pawn_about_to_promote:
                    for promote_to in (1, 2): # TODO: put when scaling for knight and bishop: 1, 2, 3, 4, easy to forget place
                        action = self.action_encoder.encode_promotion(
                            from_position=piece.position,
                            to_position=to_position,
                            promote_to=promote_to,
                            current_player_is_white=self.is_white_on_turn
                        )
                        actions.append(action)
                else:
                    action = self.action_encoder.encode(
                        from_position=piece.position,
                        to_position=to_position,
                    )
                    actions.append(action)

        self._legal_actions_cache = actions
        self._legal_actions_set_cache = set(actions)
        return actions

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_encoder.num_actions, dtype=bool)
        if self.done:
            return mask

        actions = self.legal_actions()
        if actions:
            mask[actions] = True
        return mask

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:

        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset() first.")

        self.steps_without_pawn_move_or_capture += 1

        self.legal_actions()
        if action not in self._legal_actions_set_cache:
            raise ValueError(f"Illegal action {action}")

        if self.action_encoder.is_promotion_action(action):
            from_position, to_position, promote_to = self.action_encoder.decode_promotion(action, self.is_white_on_turn)
            is_promotion = True

        else:
            from_position, to_position = self.action_encoder.decode(action)
            promote_to = None
            is_promotion = False

        piece = get_piece_by_position(self.pieces, from_position)
        if piece is None:
            raise ValueError(f"Illegal action {action}: no piece at {from_position}")

        if piece.can_implement_pawn_moves:
            self.steps_without_pawn_move_or_capture = 0

        request = MoveRequest(
            pieces=self.pieces,
            uid=piece.uid,
            position=to_position,
            is_white_on_turn=True if self.is_canonical_version else self.is_white_on_turn,
            is_from_white_player=True if self.is_canonical_version else self.is_white_on_turn,
            is_current_move_promotion=False,
            promote_to=None
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
                is_white_on_turn=True if self.is_canonical_version else self.is_white_on_turn,
                is_from_white_player=True if self.is_canonical_version else self.is_white_on_turn,
                is_current_move_promotion=True,
                promote_to=promote_to
            )

            try:
                response = make_move(request)
            except GrenightException as e:
                raise ValueError(f"Action {action} rejected by make_move: {type(e).__name__}")

        self.pieces = response.pieces

        if (response.attacked_piece_value is not None
                or type(piece) == Pawn):
            self.steps_without_pawn_move_or_capture = 0

        if self.is_canonical_version and not self.is_white_on_turn and not self.done:
            rotate_pieces_helper(self.pieces)
            self.previous_pieces_encoded_q.rotate()

        self.is_white_on_turn = not self.is_white_on_turn
        self.done = response.is_game_finished
        self.is_draw_by_rule = False
        self.draw_reason = None
        self._state_cache = None
        self._invalidate_legal_actions_cache()

        if not self.done:
            key = self.position_key()
            self.current_repetition_count = self.position_counts.get(key, 0) + 1
            self.position_counts[key] = self.current_repetition_count

            if self.is_canonical_version and not self.is_white_on_turn:
                rotate_pieces_helper(self.pieces)
                self.previous_pieces_encoded_q.rotate()

            if self.steps_without_pawn_move_or_capture >= MAX_STEPS_WITHOUT_PROGRESS:
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
                self.done = True
                self.draw_reason = "stalemate"

        reward = self.calculate_reward_registry(response)

        next_state = self.get_state()
        self.previous_pieces_encoded_q.push(next_state[-13:-3])

        self._state_cache = next_state

        info = {
            "is_enemy_in_check": response.is_enemy_in_check,
            "draw_reason": self.draw_reason,
        }

        return next_state, reward, self.done, self.draw_reason is not None, info

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

    def calculate_reward_registry(self, response) -> float:
        if self.will_do_reward_shaping:
            return self.calculate_reward_with_shaping(response)
        else:
            return self.calculate_reward_terminal_only(response)

    def calculate_reward_terminal_only(self, response) -> float:
        if not self.done:
            return 0.0

        if response.is_draw or self.is_draw_by_rule:
            return 0.0
        return 1.0

    def calculate_reward_with_shaping(self, response) -> float:
        rew_sum = 0.0

        if response.attacked_piece_value is not None:
            rew_sum += response.attacked_piece_value

        if response.is_enemy_in_check is not None:
            if response.is_enemy_in_check:
                rew_sum += self.ENEMY_IN_CHECK_REWARD

        if self.done:
            if self.is_draw_by_rule or response.is_draw:
                if self.draw_reason == "threefold_repetition":
                    rew_sum += self.THREEFOLD_REPETITION_RULE_VALUE
                else:
                    rew_sum += self.OTHER_DRAWS
                return rew_sum

            else:
                return rew_sum + 1.0

        else:
            return rew_sum
