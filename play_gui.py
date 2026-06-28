"""
Pygame GUI — Play Scrabble against the AI.

Run:  python play_gui.py
      python play_gui.py --model checkpoints/model_final.pt
      python play_gui.py --model checkpoints/model_final.pt --ai-first

Controls
--------
  Click rack tile        select it
  Click board square     place selected tile there
  Click placed tile      return it to rack
  Enter / S              submit move
  Escape / C             clear placed tiles
  P                      pass turn
  X                      toggle exchange mode
"""

import argparse
import pathlib
import sys
import time
import math
import pygame
import torch

from gaddag import GADDAG
from board import Board, PREMIUM, TILE_SCORES
from bag import Bag
from move_generator import generate_moves
from scoring import score_move as _score_move, _collect_word
from features import board_to_tensor, rack_to_tensor, move_features

# ─────────────────────────────────────────────────────────────────────────────
#  Layout
# ─────────────────────────────────────────────────────────────────────────────

WINDOW_W  = 1100
WINDOW_H  = 760
FPS       = 60

CELL      = 39                        # board square size in px
BOARD_X   = 48                        # left edge of board
BOARD_Y   = 76                        # top edge of board
BOARD_PX  = 15 * CELL                 # 585 px

PANEL_X   = BOARD_X + BOARD_PX + 22  # sidebar x  (≈ 655)
PANEL_W   = WINDOW_W - PANEL_X - 12  # ≈ 433

RACK_Y     = WINDOW_H - 108
RACK_SZ    = 54                       # rack tile size
RACK_GAP   = 6

LOG_MAX    = 14                       # move-log lines to keep

# ─────────────────────────────────────────────────────────────────────────────
#  Colour palette
# ─────────────────────────────────────────────────────────────────────────────

C_BG          = ( 20,  30,  25)
C_BOARD_BG    = ( 36,  70,  48)
C_EMPTY       = ( 48,  88,  60)
C_GRID        = ( 28,  52,  36)

C_TW          = (195,  52,  44)
C_DW          = (208, 118,  92)
C_TL          = ( 52, 102, 172)
C_DL          = ( 90, 155, 192)

C_TILE        = (246, 229, 190)
C_TILE_BORDER = (175, 150, 100)
C_TILE_SEL    = (255, 198,  46)
C_TILE_PLACED = (188, 228, 150)
C_TILE_AI     = (172, 198, 235)
C_TILE_EXCH   = (238, 136,  90)
C_TILE_BLANK  = (224, 212, 248)

C_TEXT_TILE   = ( 28,  18,   8)
C_TEXT_LIGHT  = (212, 228, 218)
C_TEXT_DIM    = (128, 152, 138)
C_TEXT_SCORE  = (255, 222,  80)
C_TEXT_ERR    = (255, 100,  90)
C_TEXT_OK     = (110, 220, 120)

C_BTN_GREEN   = ( 55, 148,  75)
C_BTN_ORANGE  = (180, 100,  45)
C_BTN_GRAY    = ( 82,  96, 112)
C_BTN_PURPLE  = (112,  70, 145)
C_BTN_RED     = (172,  52,  52)
C_BTN_DARK    = ( 35,  45,  40)
C_HOVER_TINT  = ( 30,  30,  30)   # added to RGB on hover

PREM_COLORS = {"TW": C_TW, "DW": C_DW, "TL": C_TL, "DL": C_DL}
PREM_LABEL  = {"TW": "TW", "DW": "DW", "TL": "TL", "DL": "DL"}

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clamp_color(c):
    return tuple(min(255, max(0, v)) for v in c)

def _hover(color):
    return _clamp_color(tuple(v + h for v, h in zip(color, C_HOVER_TINT)))

def _cell_rect(row, col):
    return pygame.Rect(
        BOARD_X + col * CELL,
        BOARD_Y + row * CELL,
        CELL, CELL,
    )

def _rack_rect(idx):
    total = 7 * RACK_SZ + 6 * RACK_GAP
    start_x = BOARD_X + (BOARD_PX - total) // 2
    return pygame.Rect(
        start_x + idx * (RACK_SZ + RACK_GAP),
        RACK_Y,
        RACK_SZ, RACK_SZ,
    )

# ─────────────────────────────────────────────────────────────────────────────
#  Drawing primitives
# ─────────────────────────────────────────────────────────────────────────────

def draw_rounded_rect(surf, color, rect, radius=6, border=0, border_color=None):
    pygame.draw.rect(surf, color, rect, border_radius=radius)
    if border and border_color:
        pygame.draw.rect(surf, border_color, rect, border, border_radius=radius)

def draw_tile(surf, rect, letter, is_blank=False, color=None, fonts=None,
              score_override=None, alpha=255):
    """Render a single Scrabble tile with letter and point value."""
    base = color or (C_TILE_BLANK if is_blank else C_TILE)

    # Shadow
    shadow = pygame.Rect(rect.x + 2, rect.y + 3, rect.w, rect.h)
    shadow_surf = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
    pygame.draw.rect(shadow_surf, (0, 0, 0, 80), shadow_surf.get_rect(), border_radius=6)
    surf.blit(shadow_surf, shadow.topleft)

    # Tile body
    draw_rounded_rect(surf, base, rect, radius=6,
                      border=2, border_color=C_TILE_BORDER)

    if letter and letter != ' ':
        # Main letter
        big = fonts['tile_big']
        txt = big.render(letter.upper(), True, C_TEXT_TILE)
        surf.blit(txt, txt.get_rect(center=(rect.centerx - 1, rect.centery - 4)))

        # Point value (bottom-right)
        pts = score_override if score_override is not None else (
            0 if is_blank else TILE_SCORES.get(letter.upper(), 0)
        )
        small = fonts['tile_small']
        pts_txt = small.render(str(pts), True, C_TEXT_TILE)
        surf.blit(pts_txt, (rect.right - pts_txt.get_width() - 4,
                             rect.bottom - pts_txt.get_height() - 3))


def draw_button(surf, rect, text, color, fonts, mouse_pos, radius=8):
    c = _hover(color) if rect.collidepoint(mouse_pos) else color
    draw_rounded_rect(surf, c, rect, radius=radius,
                      border=1, border_color=_clamp_color(tuple(v + 40 for v in c)))
    lbl = fonts['ui'].render(text, True, C_TEXT_LIGHT)
    surf.blit(lbl, lbl.get_rect(center=rect.center))
    return rect.collidepoint(mouse_pos)


# ─────────────────────────────────────────────────────────────────────────────
#  Board rendering
# ─────────────────────────────────────────────────────────────────────────────

def draw_board(surf, board, placed, ai_last, fonts, hover_cell, show_coords):
    # Board background
    board_rect = pygame.Rect(BOARD_X - 2, BOARD_Y - 2, BOARD_PX + 4, BOARD_PX + 4)
    draw_rounded_rect(surf, C_BOARD_BG, board_rect, radius=4)

    placed_cells = {(p['row'], p['col']): p for p in placed}
    ai_cells     = {(r, c) for r, c, _, _ in ai_last} if ai_last else set()

    # Column labels
    for c in range(15):
        lbl = fonts['coord'].render(str(c), True, C_TEXT_DIM)
        x = BOARD_X + c * CELL + CELL // 2 - lbl.get_width() // 2
        surf.blit(lbl, (x, BOARD_Y - 18))

    # Row labels
    for r in range(15):
        lbl = fonts['coord'].render(str(r), True, C_TEXT_DIM)
        y = BOARD_Y + r * CELL + CELL // 2 - lbl.get_height() // 2
        surf.blit(lbl, (BOARD_X - lbl.get_width() - 5, y))

    for r in range(15):
        for c in range(15):
            rect = _cell_rect(r, c)
            sq   = board.get(r, c)

            # ── Background colour ──
            if (r, c) in placed_cells:
                cell_color = C_TILE_PLACED
            elif (r, c) in ai_cells:
                cell_color = C_TILE_AI
            elif sq.tile:
                cell_color = C_TILE
            else:
                prem = PREMIUM.get((r, c))
                cell_color = PREM_COLORS.get(prem, C_EMPTY) if prem else C_EMPTY

            # Hover highlight on empty squares
            if (r, c) == hover_cell and sq.tile is None and (r, c) not in placed_cells:
                cell_color = _hover(cell_color)

            draw_rounded_rect(surf, cell_color, rect.inflate(-2, -2), radius=3)

            # ── Content ──
            if (r, c) in placed_cells:
                p = placed_cells[(r, c)]
                draw_tile(surf, rect.inflate(-4, -4), p['letter'],
                          p['is_blank'], C_TILE_PLACED, fonts)

            elif sq.tile:
                tc = C_TILE_AI if (r, c) in ai_cells else C_TILE
                draw_tile(surf, rect.inflate(-4, -4), sq.tile,
                          sq.is_blank, tc, fonts)

            else:
                prem = PREMIUM.get((r, c))
                if r == 7 and c == 7 and not prem:
                    # Centre star
                    cx, cy = rect.center
                    star = fonts['star'].render('★', True, C_DW)
                    surf.blit(star, star.get_rect(center=(cx, cy)))
                elif prem:
                    lbl = fonts['prem'].render(PREM_LABEL[prem], True,
                                               C_TEXT_LIGHT)
                    surf.blit(lbl, lbl.get_rect(center=rect.center))

            # Grid lines
            pygame.draw.rect(surf, C_GRID, rect, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar / rack rendering
# ─────────────────────────────────────────────────────────────────────────────

def draw_sidebar(surf, fonts, mouse_pos,
                 human_score, ai_score, bag_size,
                 human_turn, game_over, mode,
                 log, message, message_color):
    x, w = PANEL_X, PANEL_W

    # Title
    title = fonts['title'].render('SCRABBLE  AI', True, C_TEXT_LIGHT)
    surf.blit(title, (x + w // 2 - title.get_width() // 2, 14))

    # Scores
    y = 62
    draw_rounded_rect(surf, C_BTN_DARK,
                      pygame.Rect(x, y, w, 70), radius=8)
    you_c = C_TEXT_SCORE if human_turn and not game_over else C_TEXT_LIGHT
    ai_c  = C_TEXT_SCORE if not human_turn and not game_over else C_TEXT_LIGHT

    you_lbl = fonts['score_label'].render('YOU', True, C_TEXT_DIM)
    you_val = fonts['score_big'].render(str(human_score), True, you_c)
    ai_lbl  = fonts['score_label'].render('AI',  True, C_TEXT_DIM)
    ai_val  = fonts['score_big'].render(str(ai_score),    True, ai_c)

    surf.blit(you_lbl, (x + 18, y + 8))
    surf.blit(you_val, (x + 18, y + 26))
    surf.blit(ai_lbl,  (x + w - 18 - ai_lbl.get_width(), y + 8))
    surf.blit(ai_val,  (x + w - 18 - ai_val.get_width(), y + 26))

    # Separator
    pygame.draw.line(surf, C_TEXT_DIM,
                     (x + w // 2, y + 10), (x + w // 2, y + 60), 1)

    # Bag count
    y += 78
    bag_txt = fonts['ui'].render(f'Bag: {bag_size} tiles', True, C_TEXT_DIM)
    surf.blit(bag_txt, (x + w // 2 - bag_txt.get_width() // 2, y))

    # Buttons
    btn_y = y + 28
    btn_h = 36
    btn_gap = 8
    btn_w2 = (w - btn_gap) // 2

    buttons = {}

    if game_over:
        pass
    elif mode == 'exchange':
        r = pygame.Rect(x, btn_y, w, btn_h)
        buttons['confirm_exchange'] = r
        draw_button(surf, r, 'CONFIRM EXCHANGE', C_BTN_GREEN, fonts, mouse_pos)
        btn_y += btn_h + btn_gap
        r2 = pygame.Rect(x, btn_y, w, btn_h)
        buttons['cancel_exchange'] = r2
        draw_button(surf, r2, 'CANCEL', C_BTN_GRAY, fonts, mouse_pos)
        btn_y += btn_h + btn_gap
    elif mode in ('play', 'ai_thinking'):
        disabled = (mode == 'ai_thinking')

        # Submit / Clear
        r1 = pygame.Rect(x,                   btn_y, btn_w2, btn_h)
        r2 = pygame.Rect(x + btn_w2 + btn_gap, btn_y, btn_w2, btn_h)
        buttons['submit'] = r1
        buttons['clear']  = r2
        draw_button(surf, r1, 'SUBMIT  [↵]',
                    C_BTN_GRAY if disabled else C_BTN_GREEN, fonts, mouse_pos)
        draw_button(surf, r2, 'CLEAR  [Esc]',
                    C_BTN_GRAY if disabled else C_BTN_ORANGE, fonts, mouse_pos)
        btn_y += btn_h + btn_gap

        # Pass / Exchange
        r3 = pygame.Rect(x,                   btn_y, btn_w2, btn_h)
        r4 = pygame.Rect(x + btn_w2 + btn_gap, btn_y, btn_w2, btn_h)
        buttons['pass']     = r3
        buttons['exchange'] = r4
        draw_button(surf, r3, 'PASS  [P]',
                    C_BTN_GRAY, fonts, mouse_pos)
        draw_button(surf, r4, 'EXCHANGE  [X]',
                    C_BTN_GRAY if disabled else C_BTN_PURPLE, fonts, mouse_pos)
        btn_y += btn_h + btn_gap

    # Turn indicator / message
    y = btn_y + 6
    if game_over:
        msg_c = C_TEXT_SCORE
        msg   = 'GAME  OVER'
    elif mode == 'ai_thinking':
        msg_c = C_TEXT_DIM
        msg   = 'AI is thinking...'
    elif mode == 'exchange':
        msg_c = C_TEXT_LIGHT
        msg   = 'Click tiles to exchange'
    elif human_turn:
        msg_c = C_TEXT_OK
        msg   = 'Your turn'
    else:
        msg_c = C_TEXT_DIM
        msg   = ''

    if msg:
        t = fonts['ui'].render(msg, True, msg_c)
        surf.blit(t, (x + w // 2 - t.get_width() // 2, y))
        y += 22

    if message:
        lines = message.split('\n')
        for line in lines:
            t = fonts['msg'].render(line, True, message_color)
            surf.blit(t, (x + w // 2 - t.get_width() // 2, y))
            y += 19

    # Move log
    y = max(y + 12, 330)
    log_lbl = fonts['coord'].render('MOVE LOG', True, C_TEXT_DIM)
    surf.blit(log_lbl, (x, y))
    y += 16
    pygame.draw.line(surf, C_TEXT_DIM, (x, y), (x + w, y), 1)
    y += 4

    for entry in log[-LOG_MAX:]:
        colour, text = entry
        t = fonts['log'].render(text, True, colour)
        surf.blit(t, (x, y))
        y += 17

    return buttons


def draw_rack(surf, rack, selected_idx, exchange_sel, mode, fonts, mouse_pos):
    """Draw the 7-tile player rack along the bottom of the board."""
    label = fonts['ui'].render('YOUR RACK', True, C_TEXT_DIM)
    surf.blit(label, (BOARD_X, RACK_Y - 22))

    rects = []
    for i, tile in enumerate(rack):
        rect = _rack_rect(i)
        if tile is None:
            # Empty slot (tile has been placed on board)
            draw_rounded_rect(surf, C_BTN_DARK, rect, radius=6,
                              border=1, border_color=C_GRID)
            rects.append(rect)
            continue

        if mode == 'exchange' and i in exchange_sel:
            color = C_TILE_EXCH
        elif i == selected_idx:
            color = C_TILE_SEL
        else:
            color = C_TILE_BLANK if tile == '?' else C_TILE

        # Hover
        if rect.collidepoint(mouse_pos) and tile is not None:
            color = _hover(color)

        draw_tile(surf, rect, tile if tile != '?' else '?',
                  is_blank=(tile == '?'), color=color, fonts=fonts,
                  score_override=0 if tile == '?' else None)
        rects.append(rect)

    return rects


def draw_blank_picker(surf, fonts, mouse_pos):
    """Overlay to choose which letter a blank tile represents."""
    overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    surf.blit(overlay, (0, 0))

    bw, bh = 48, 48
    cols, gap = 9, 6
    rows = 3
    total_w = cols * bw + (cols - 1) * gap
    total_h = rows * bh + (rows - 1) * gap
    ox = WINDOW_W // 2 - total_w // 2
    oy = WINDOW_H // 2 - total_h // 2 - 30

    bg = pygame.Rect(ox - 20, oy - 42, total_w + 40, total_h + 80)
    draw_rounded_rect(surf, C_BTN_DARK, bg, radius=12,
                      border=2, border_color=C_TEXT_DIM)

    title = fonts['ui'].render('Choose letter for blank tile:', True, C_TEXT_LIGHT)
    surf.blit(title, title.get_rect(center=(WINDOW_W // 2, oy - 20)))

    letter_rects = {}
    for i, ch in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        row, col = divmod(i, cols)
        rx = ox + col * (bw + gap)
        ry = oy + row * (bh + gap)
        rect = pygame.Rect(rx, ry, bw, bh)
        color = _hover(C_BTN_GRAY) if rect.collidepoint(mouse_pos) else C_BTN_GRAY
        draw_tile(surf, rect, ch, color=color, fonts=fonts)
        letter_rects[ch] = rect

    return letter_rects


def draw_game_over(surf, fonts, human_score, ai_score):
    overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 140))
    surf.blit(overlay, (0, 0))

    cx, cy = WINDOW_W // 2, WINDOW_H // 2
    box = pygame.Rect(cx - 200, cy - 120, 400, 240)
    draw_rounded_rect(surf, C_BG, box, radius=14,
                      border=2, border_color=C_TEXT_DIM)

    if human_score > ai_score:
        result, color = 'YOU WIN!', C_TEXT_OK
    elif ai_score > human_score:
        result, color = 'AI WINS', C_TEXT_ERR
    else:
        result, color = "IT'S A TIE", C_TEXT_LIGHT

    t1 = fonts['title'].render(result, True, color)
    t2 = fonts['score_big'].render(f'{human_score}  —  {ai_score}', True, C_TEXT_LIGHT)
    t3 = fonts['ui'].render('Press R to play again  or  Q to quit', True, C_TEXT_DIM)

    surf.blit(t1, t1.get_rect(center=(cx, cy - 60)))
    surf.blit(t2, t2.get_rect(center=(cx, cy)))
    surf.blit(t3, t3.get_rect(center=(cx, cy + 60)))


# ─────────────────────────────────────────────────────────────────────────────
#  Move validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_placed(board, placed, gaddag):
    """Return (ok, error_msg, placements_list, is_horizontal)."""
    if not placed:
        return False, 'Place at least one tile first.', [], True

    rows = [p['row'] for p in placed]
    cols = [p['col'] for p in placed]

    all_same_row = len(set(rows)) == 1
    all_same_col = len(set(cols)) == 1

    if not all_same_row and not all_same_col:
        return False, 'Tiles must be in one row or one column.', [], True

    horizontal = all_same_row
    dr, dc = (0, 1) if horizontal else (1, 0)
    r0, c0 = min(rows), min(cols)

    # Check no gaps in the placement (existing board tiles can fill gaps)
    r, c = r0, c0
    placed_cells = {(p['row'], p['col']) for p in placed}
    while (r, c) in placed_cells or (board.get(r, c).tile is not None):
        r += dr
        c += dc
    end_r = r - dr
    end_c = c - dc
    if horizontal:
        if end_c < max(cols):
            return False, 'Gap in word — place tiles to fill it.', [], horizontal
    else:
        if end_r < max(rows):
            return False, 'Gap in word — place tiles to fill it.', [], horizontal

    # First move must cover centre
    if board.is_empty():
        if (7, 7) not in placed_cells:
            return False, 'First move must cover the centre square (7,7).', [], horizontal
    else:
        touches = any(board.has_adjacent_tile(p['row'], p['col']) or
                      board.get(p['row'], p['col']).tile is not None
                      for p in placed)
        if not touches:
            return False, 'Word must connect to existing tiles.', [], horizontal

    # Build placements list
    placements = [(p['row'], p['col'], p['letter'], p['is_blank']) for p in placed]

    # Temporarily place tiles to check words
    for r2, c2, letter, is_blank in placements:
        board.place_tile(r2, c2, letter, is_blank)

    words_to_check = []
    main_cells = _collect_word(board, r0, c0, horizontal)
    main_word = ''.join(board.get(rr, cc).tile for rr, cc in main_cells)
    words_to_check.append(main_word)

    for r2, c2, _, _ in placements:
        cross = _collect_word(board, r2, c2, not horizontal)
        if len(cross) > 1:
            words_to_check.append(''.join(board.get(rr, cc).tile for rr, cc in cross))

    # Undo
    for r2, c2, _, _ in placements:
        board.remove_tile(r2, c2)

    invalid = [w for w in words_to_check if not gaddag.contains(w)]
    if invalid:
        return False, f"Invalid: {', '.join(invalid)}", [], horizontal

    return True, '', placements, horizontal


# ─────────────────────────────────────────────────────────────────────────────
#  AI helpers
# ─────────────────────────────────────────────────────────────────────────────

def ai_choose_move(board, rack, bag, gaddag, model):
    moves = generate_moves(board, gaddag, rack)
    if not moves:
        return None
    for m in moves:
        for r, c, l, b in m.placements:
            board.place_tile(r, c, l, b)
        m.score = _score_move(board, m.placements, m.is_horizontal)
        for r, c, _, _ in m.placements:
            board.remove_tile(r, c)
    if model is None:
        return max(moves, key=lambda m: m.score)
    model.eval()
    device = next(model.parameters()).device
    board_t = board_to_tensor(board).to(device)
    rack_t  = rack_to_tensor(rack).to(device)
    with torch.no_grad():
        mf = torch.stack([move_features(m, rack, len(bag), board) for m in moves]).to(device)
        n  = len(moves)
        vals = model(
            board_t.unsqueeze(0).expand(n, -1, -1, -1),
            rack_t.unsqueeze(0).expand(n, -1),
            mf,
        ).squeeze(1)
    return moves[vals.argmax().item()]


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

RACK_SIZE = 7

def new_game(gaddag, model, ai_first):
    bag = Bag()
    if ai_first:
        ai_rack, human_rack = bag.draw(RACK_SIZE), bag.draw(RACK_SIZE)
    else:
        human_rack, ai_rack = bag.draw(RACK_SIZE), bag.draw(RACK_SIZE)
    return {
        'board':        Board(),
        'bag':          bag,
        'gaddag':       gaddag,
        'model':        model,
        'human_rack':   human_rack,   # list of str | None (None = placed on board)
        'ai_rack':      ai_rack,
        'human_score':  0,
        'ai_score':     0,
        'human_turn':   not ai_first,
        'game_over':    False,
        'human_passes': 0,
        'ai_passes':    0,
        # UI state
        'placed':         [],     # {'row','col','letter','is_blank','rack_idx'}
        'selected_idx':   None,   # rack slot index currently selected
        'exchange_sel':   set(),  # rack indices marked for exchange
        'mode':           'play' if not ai_first else 'ai_thinking',
        'ai_last':        [],     # placements from AI's last move (highlight)
        'message':        '',
        'msg_color':      (212, 228, 218),
        'msg_timer':      0,
        'log':            [],
    }


def run(args):
    pygame.init()
    pygame.display.set_caption('Scrabble AI')
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock  = pygame.time.Clock()

    # ── Fonts ──────────────────────────────────────────────────────────
    def load_font(size, bold=False):
        for name in ('Segoe UI', 'Arial', 'Helvetica', 'DejaVu Sans', None):
            try:
                f = pygame.font.SysFont(name, size, bold=bold)
                if f:
                    return f
            except Exception:
                pass
        return pygame.font.Font(None, size)

    fonts = {
        'title':       load_font(30, bold=True),
        'score_big':   load_font(34, bold=True),
        'score_label': load_font(14),
        'ui':          load_font(15, bold=True),
        'msg':         load_font(13),
        'tile_big':    load_font(22, bold=True),
        'tile_small':  load_font(11),
        'prem':        load_font(9,  bold=True),
        'coord':       load_font(11),
        'star':        load_font(18),
        'log':         load_font(12),
    }

    # ── Load dictionary + model ─────────────────────────────────────────
    screen.fill(C_BG)
    loading = fonts['title'].render('Loading dictionary...', True, C_TEXT_LIGHT)
    screen.blit(loading, loading.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2)))
    pygame.display.flip()

    gaddag = GADDAG.load_or_build('TWL06.txt', 'gaddag.pkl')

    model = None
    if args.model:
        mp = pathlib.Path(args.model)
        if mp.exists():
            from model import ScrabbleNet
            net = ScrabbleNet()
            ckpt = torch.load(mp, map_location='cpu')
            net.load_state_dict(ckpt['model'])
            net.eval()
            model = net

    # ── Game state ─────────────────────────────────────────────────────
    gs = new_game(gaddag, model, args.ai_first)

    def set_msg(text, ok=True, ticks=200):
        gs['message']   = text
        gs['msg_color'] = C_TEXT_OK if ok else C_TEXT_ERR
        gs['msg_timer'] = ticks

    def add_log(text, human=True):
        color = C_TEXT_LIGHT if human else C_TILE_AI
        gs['log'].append((color, text))

    def replenish(rack):
        drawn = gs['bag'].draw(RACK_SIZE - sum(1 for t in rack if t is not None))
        for t in drawn:
            for i in range(len(rack)):
                if rack[i] is None:
                    rack[i] = t
                    break
            else:
                rack.append(t)

    def check_end_game():
        all_none_human = all(t is None for t in gs['human_rack'])
        all_none_ai    = all(t is None for t in gs['ai_rack'])
        if (all_none_human or all_none_ai) and gs['bag'].is_empty():
            # Tile penalties
            from board import TILE_SCORES as TS
            hp = sum(TS.get(t, 0) for t in gs['human_rack'] if t)
            ap = sum(TS.get(t, 0) for t in gs['ai_rack']    if t)
            gs['human_score'] -= hp
            gs['ai_score']    -= ap
            if all_none_human:
                gs['human_score'] += ap
            else:
                gs['ai_score'] += hp
            gs['game_over'] = True

    # ─── Event / render loop ────────────────────────────────────────────
    hover_cell = None
    ai_timer   = 0       # frames to wait before AI moves
    ai_move_pending = None

    running = True
    while running:
        clock.tick(FPS)
        mouse_pos = pygame.mouse.get_pos()

        # Hover cell detection
        hover_cell = None
        mx, my = mouse_pos
        if BOARD_X <= mx < BOARD_X + BOARD_PX and BOARD_Y <= my < BOARD_Y + BOARD_PX:
            hc = (my - BOARD_Y) // CELL
            hr = (my - BOARD_Y) // CELL
            hcol = (mx - BOARD_X) // CELL
            hover_cell = (hr, hcol)

        # ── AI turn logic ────────────────────────────────────────────────
        if gs['mode'] == 'ai_thinking':
            ai_timer += 1
            if ai_timer == 1:
                # Compute move on first frame (blocks briefly, acceptable for ~15ms)
                ai_rack_clean = [t for t in gs['ai_rack'] if t is not None]
                chosen = ai_choose_move(
                    gs['board'], ai_rack_clean, gs['bag'], gaddag, model
                )
                ai_move_pending = chosen

            if ai_timer >= 45:  # ~0.75s display delay so player can see
                chosen = ai_move_pending
                if chosen is None:
                    gs['ai_passes'] += 1
                    add_log('AI passed.', human=False)
                else:
                    for r, c, letter, is_blank in chosen.placements:
                        gs['board'].place_tile(r, c, letter, is_blank)
                    pts = _score_move(gs['board'], chosen.placements, chosen.is_horizontal)
                    gs['ai_score'] += pts
                    for _, _, letter, is_blank in chosen.placements:
                        tile = '?' if is_blank else letter
                        gs['ai_rack'].remove(tile)
                    replenish(gs['ai_rack'])
                    gs['ai_last'] = chosen.placements
                    gs['ai_passes'] = 0
                    add_log(f'AI: {chosen.word} +{pts}', human=False)

                check_end_game()
                if not gs['game_over']:
                    gs['mode']      = 'play'
                    gs['human_turn'] = True
                ai_timer = 0
                ai_move_pending = None

        # ── Message timer ────────────────────────────────────────────────
        if gs['msg_timer'] > 0:
            gs['msg_timer'] -= 1
            if gs['msg_timer'] == 0:
                gs['message'] = ''

        # ── Events ──────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                k = event.key
                if gs['game_over']:
                    if k == pygame.K_r:
                        gs = new_game(gaddag, model, args.ai_first)
                    elif k in (pygame.K_q, pygame.K_ESCAPE):
                        running = False

                elif gs['mode'] == 'blank_picker':
                    pass   # handled below in MOUSEBUTTONDOWN

                elif gs['mode'] == 'play':
                    if k in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_s):
                        # Submit
                        ok, err, placements, horiz = validate_placed(
                            gs['board'], gs['placed'], gaddag)
                        if not ok:
                            set_msg(err, ok=False)
                        else:
                            for r, c, letter, ib in placements:
                                gs['board'].place_tile(r, c, letter, ib)
                            pts = _score_move(gs['board'], placements, horiz)
                            gs['human_score'] += pts
                            for p in gs['placed']:
                                tile = '?' if p['is_blank'] else p['letter']
                                gs['human_rack'][p['rack_idx']] = None
                            replenish(gs['human_rack'])
                            word = ''.join(p['letter'] for p in sorted(
                                gs['placed'], key=lambda p: (p['col'] if horiz else p['row'])))
                            add_log(f'You: {word} +{pts}', human=True)
                            gs['placed']       = []
                            gs['selected_idx'] = None
                            gs['ai_last']      = []
                            gs['human_passes'] = 0
                            gs['human_turn']   = False
                            gs['mode']         = 'ai_thinking'
                            check_end_game()

                    elif k in (pygame.K_ESCAPE, pygame.K_c):
                        # Clear
                        for p in gs['placed']:
                            gs['human_rack'][p['rack_idx']] = p['rack_tile']
                        gs['placed']       = []
                        gs['selected_idx'] = None

                    elif k == pygame.K_p:
                        # Pass
                        for p in gs['placed']:
                            gs['human_rack'][p['rack_idx']] = p['rack_tile']
                        gs['placed']       = []
                        gs['selected_idx'] = None
                        gs['human_passes'] += 1
                        add_log('You passed.', human=True)
                        gs['human_turn'] = False
                        gs['mode']       = 'ai_thinking'

                    elif k == pygame.K_x:
                        for p in gs['placed']:
                            gs['human_rack'][p['rack_idx']] = p['rack_tile']
                        gs['placed']       = []
                        gs['selected_idx'] = None
                        gs['exchange_sel'] = set()
                        gs['mode']         = 'exchange'

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos

                # ── Blank picker ────────────────────────────────────────
                if gs['mode'] == 'blank_picker':
                    letter_rects = draw_blank_picker(screen, fonts, pos)
                    for ch, rect in letter_rects.items():
                        if rect.collidepoint(pos):
                            pd = gs['_blank_pending']
                            gs['placed'].append({
                                'row': pd['row'], 'col': pd['col'],
                                'letter': ch, 'is_blank': True,
                                'rack_idx': pd['rack_idx'],
                                'rack_tile': '?',
                            })
                            gs['human_rack'][pd['rack_idx']] = None
                            gs['_blank_pending'] = None
                            gs['mode'] = 'play'
                            break
                    continue

                if gs['game_over']:
                    continue

                # ── Board click ─────────────────────────────────────────
                if (gs['mode'] == 'play' and
                        BOARD_X <= pos[0] < BOARD_X + BOARD_PX and
                        BOARD_Y <= pos[1] < BOARD_Y + BOARD_PX):

                    col = (pos[0] - BOARD_X) // CELL
                    row = (pos[1] - BOARD_Y) // CELL

                    placed_here = next(
                        (p for p in gs['placed'] if p['row'] == row and p['col'] == col),
                        None
                    )

                    if placed_here:
                        # Click placed tile → return to rack
                        gs['human_rack'][placed_here['rack_idx']] = placed_here['rack_tile']
                        gs['placed'].remove(placed_here)
                        gs['selected_idx'] = None

                    elif gs['selected_idx'] is not None:
                        tile = gs['human_rack'][gs['selected_idx']]
                        sq   = gs['board'].get(row, col)
                        if tile is not None and sq.tile is None:
                            if tile == '?':
                                # Need letter picker
                                gs['_blank_pending'] = {
                                    'row': row, 'col': col,
                                    'rack_idx': gs['selected_idx'],
                                }
                                gs['mode'] = 'blank_picker'
                            else:
                                gs['placed'].append({
                                    'row': row, 'col': col,
                                    'letter': tile, 'is_blank': False,
                                    'rack_idx': gs['selected_idx'],
                                    'rack_tile': tile,
                                })
                                gs['human_rack'][gs['selected_idx']] = None
                            gs['selected_idx'] = None

                # ── Rack click ──────────────────────────────────────────
                for i in range(7):
                    rect = _rack_rect(i)
                    if rect.collidepoint(pos):
                        tile = gs['human_rack'][i] if i < len(gs['human_rack']) else None
                        if tile is None:
                            break
                        if gs['mode'] == 'exchange':
                            if i in gs['exchange_sel']:
                                gs['exchange_sel'].discard(i)
                            else:
                                gs['exchange_sel'].add(i)
                        elif gs['mode'] == 'play':
                            gs['selected_idx'] = i if gs['selected_idx'] != i else None
                        break

                # ── Button clicks (rendered each frame, checked here) ───
                # We'll handle these by re-checking rects directly
                bx, bw2 = PANEL_X, (PANEL_W - 8) // 2

                def btn_rect(col_idx, row_idx, row_start=None):
                    bh, gap = 36, 8
                    by_base = 62 + 78 + 28   # matches draw_sidebar
                    y = by_base + row_idx * (bh + gap)
                    x = bx if col_idx == 0 else bx + bw2 + gap
                    return pygame.Rect(x, y, bw2, bh)

                if gs['mode'] == 'play':
                    if btn_rect(0, 0).collidepoint(pos):  # SUBMIT
                        pygame.event.post(pygame.event.Event(
                            pygame.KEYDOWN,
                            {'key': pygame.K_RETURN, 'mod': 0, 'unicode': '\r',
                             'scancode': 28}))
                    elif btn_rect(1, 0).collidepoint(pos):  # CLEAR
                        pygame.event.post(pygame.event.Event(
                            pygame.KEYDOWN,
                            {'key': pygame.K_ESCAPE, 'mod': 0, 'unicode': '',
                             'scancode': 1}))
                    elif btn_rect(0, 1).collidepoint(pos):  # PASS
                        pygame.event.post(pygame.event.Event(
                            pygame.KEYDOWN,
                            {'key': pygame.K_p, 'mod': 0, 'unicode': 'p',
                             'scancode': 25}))
                    elif btn_rect(1, 1).collidepoint(pos):  # EXCHANGE
                        pygame.event.post(pygame.event.Event(
                            pygame.KEYDOWN,
                            {'key': pygame.K_x, 'mod': 0, 'unicode': 'x',
                             'scancode': 45}))

                elif gs['mode'] == 'exchange':
                    full_btn = pygame.Rect(bx, 168, PANEL_W, 36)
                    cancel   = pygame.Rect(bx, 168 + 44, PANEL_W, 36)
                    if full_btn.collidepoint(pos):
                        sel = sorted(gs['exchange_sel'])
                        if not sel:
                            set_msg('Select tiles to exchange first.', ok=False)
                        elif len(gs['bag']) < RACK_SIZE:
                            set_msg('Not enough tiles in bag.', ok=False)
                            gs['mode'] = 'play'
                        else:
                            tiles_out = [gs['human_rack'][i] for i in sel]
                            for i in sel:
                                gs['human_rack'][i] = None
                            drawn = gs['bag'].draw(len(tiles_out))
                            gs['bag'].return_tiles(tiles_out)
                            for i, t in zip(sel, drawn):
                                gs['human_rack'][i] = t
                            add_log(f'You exchanged {len(tiles_out)}.', human=True)
                            gs['exchange_sel'] = set()
                            gs['human_passes'] += 1
                            gs['human_turn']   = False
                            gs['mode']         = 'ai_thinking'
                    elif cancel.collidepoint(pos):
                        gs['exchange_sel'] = set()
                        gs['mode'] = 'play'

        # ────────────────────────────────────────────────────────────────
        #  RENDER
        # ────────────────────────────────────────────────────────────────
        screen.fill(C_BG)

        draw_board(screen, gs['board'], gs['placed'],
                   gs['ai_last'], fonts, hover_cell, show_coords=True)

        rack_rects = draw_rack(screen, gs['human_rack'],
                               gs['selected_idx'], gs['exchange_sel'],
                               gs['mode'], fonts, mouse_pos)

        draw_sidebar(screen, fonts, mouse_pos,
                     gs['human_score'], gs['ai_score'],
                     len(gs['bag']),
                     gs['human_turn'], gs['game_over'],
                     gs['mode'],
                     gs['log'],
                     gs['message'], gs['msg_color'])

        if gs['mode'] == 'blank_picker':
            draw_blank_picker(screen, fonts, mouse_pos)

        if gs['game_over']:
            draw_game_over(screen, fonts, gs['human_score'], gs['ai_score'])

        pygame.display.flip()

    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description='Play Scrabble (Pygame GUI)')
    parser.add_argument('--model',    default=None,
                        help='Path to trained model .pt file')
    parser.add_argument('--ai-first', action='store_true', dest='ai_first')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
