"""
Real-Time Computer Vision Rock, Paper, Scissors Game
=====================================================
Custom GUI edition utilizing user-provided image assets:
- bg_layout.png: Master board layout with AI box, Player box, Score tabs & Center badge
- rock.png: Custom circular Pink Rock hand asset
- paper.png: Custom circular Purple Paper hand asset
- scissors.png: Custom circular Blue Scissors hand asset

Requirements & Features:
- Real-time hand landmark tracking with MediaPipe (Tasks API and legacy Solutions API)
- Geometric finger curling classification for Rock, Paper, and Scissors
- 3-round match structure (First to 2 wins or highest score after 3 decisive rounds)
- Dynamic on-screen countdown synchronized with asset animations ("ROCK!", "PAPER!", "SCISSORS!", "SHOOT!")
- Strict 3-second delay/pause between consecutive rounds with live timer & progress indicator
- Post-match celebration display for the winner: "🎉 Congratulations! 🏆"
- Keyboard controls: 'R' to restart, 'Q' or ESC to quit, SPACE to start round
- Robust camera initialization with fallback search and clean resource cleanup
"""

import os
import sys
import time
import math
import random
import argparse
import urllib.request
from typing import Dict, List, Tuple, Optional, Any

# Ensure terminal stdout supports UTF-8 emojis across platforms
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def safe_print(*args, **kwargs):
    """Safely print UTF-8 text without throwing UnicodeEncodeError on cp1252 consoles."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        safe_args = [
            str(arg).encode("ascii", errors="backslashreplace").decode("ascii")
            for arg in args
        ]
        print(*safe_args, **kwargs)

import cv2
import numpy as np

# Pillow for high-fidelity composition & text rendering
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# MediaPipe
import mediapipe as mp


# ============================================================================
# 1. GESTURE CLASSIFIER & LANDMARK GEOMETRY
# ============================================================================

class GestureClassifier:
    """
    Classifies player hand gestures into 'Rock', 'Paper', 'Scissors', or 'Detecting...'
    based on geometric analysis of the 21 MediaPipe hand landmarks.

    Landmark reference indices:
      0: Wrist
      Thumb:  CMC=1,  MCP=2,  IP=3,  TIP=4
      Index:  MCP=5,  PIP=6,  DIP=7, TIP=8
      Middle: MCP=9,  PIP=10, DIP=11, TIP=12
      Ring:   MCP=13, PIP=14, DIP=15, TIP=16
      Pinky:  MCP=17, PIP=18, DIP=19, TIP=20
    """

    WRIST = 0
    FINGER_TIPS = [4, 8, 12, 16, 20]
    FINGER_PIPS = [3, 6, 10, 14, 18]
    FINGER_MCPS = [2, 5, 9, 13, 17]

    @staticmethod
    def _dist(p1: Tuple[float, float, ...], p2: Tuple[float, float, ...]) -> float:
        """Calculate 2D Euclidean distance between two points."""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @classmethod
    def get_finger_states(cls, landmarks: List[Tuple[float, float, float]]) -> Dict[str, bool]:
        """
        Determine whether each finger is extended (True) or curled (False).
        """
        wrist = landmarks[cls.WRIST]
        states: Dict[str, bool] = {}

        # 1. Four fingers: Index (8), Middle (12), Ring (16), Pinky (20)
        finger_specs = [
            ('index', 8, 6, 5),
            ('middle', 12, 10, 9),
            ('ring', 16, 14, 13),
            ('pinky', 20, 18, 17)
        ]

        for name, tip_idx, pip_idx, mcp_idx in finger_specs:
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            mcp = landmarks[mcp_idx]

            d_tip_wrist = cls._dist(tip, wrist)
            d_pip_wrist = cls._dist(pip, wrist)
            d_tip_mcp = cls._dist(tip, mcp)
            d_pip_mcp = cls._dist(pip, mcp)

            # Extended if tip is clearly further from wrist and knuckle than the PIP joint
            is_extended = (d_tip_wrist > d_pip_wrist * 1.05) and (d_tip_mcp > d_pip_mcp * 0.95)

            # Upright orientation sanity check: tip should be above PIP (y increases downwards)
            if tip[1] > pip[1] and d_tip_wrist < d_pip_wrist:
                is_extended = False

            states[name] = is_extended

        # 2. Thumb: Tip=4, IP=3, MCP=2
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        pinky_mcp = landmarks[17]
        middle_mcp = landmarks[9]

        d_tip_pinky = cls._dist(thumb_tip, pinky_mcp)
        d_ip_pinky = cls._dist(thumb_ip, pinky_mcp)
        d_tip_mid = cls._dist(thumb_tip, middle_mcp)
        d_ip_mid = cls._dist(thumb_ip, middle_mcp)
        d_tip_mcp = cls._dist(thumb_tip, thumb_mcp)
        d_ip_mcp = cls._dist(thumb_ip, thumb_mcp)

        # Extended if stretched outward away from palm
        thumb_extended = (
            (d_tip_pinky > d_ip_pinky * 1.05) and
            (d_tip_mid > d_ip_mid * 1.05) and
            (d_tip_mcp > d_ip_mcp)
        )
        states['thumb'] = thumb_extended

        return states

    @classmethod
    def classify(cls, landmarks: Optional[List[Tuple[float, float, float]]]) -> str:
        """
        Classifies hand landmarks into 'Rock', 'Paper', 'Scissors', or 'Detecting...'.
        """
        if not landmarks or len(landmarks) < 21:
            return "No Hand"

        states = cls.get_finger_states(landmarks)
        idx = states['index']
        mid = states['middle']
        rng = states['ring']
        pnk = states['pinky']
        thm = states['thumb']

        # Rock: All 4 main fingers curled tight into fist
        if not idx and not mid and not rng and not pnk:
            return "Rock"

        # Paper: All 5 fingers extended flat (or all 4 extended with thumb open)
        elif idx and mid and rng and pnk:
            return "Paper"

        # Scissors: Index and middle extended; ring, pinky curled tight
        elif idx and mid and not rng and not pnk:
            return "Scissors"

        # Ambiguous / transitional gestures
        else:
            return "Detecting..."


# ============================================================================
# 2. UNIVERSAL MEDIAPIPE HAND DETECTOR
# ============================================================================

class HandDetector:
    """
    Universal hand detection wrapper supporting both:
    1. MediaPipe Tasks API (mediapipe >= 0.10.x / 1.x) using HandLandmarker
    2. Legacy MediaPipe Solutions API (mediapipe.solutions.hands)
    """

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    MODEL_NAME = "hand_landmarker.task"

    FINGER_TIPS = [4, 8, 12, 16, 20]

    SKELETON_CONNECTIONS = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle
        (9, 10), (10, 11), (11, 12),
        # Ring
        (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm base
        (5, 9), (9, 13), (13, 17)
    ]

    def __init__(self, max_hands: int = 1, min_detection_con: float = 0.5):
        self.max_hands = max_hands
        self.min_detection_con = min_detection_con
        self.api_type = None
        self.detector = None

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            safe_print("[HandDetector] Initializing with MediaPipe Solutions API...")
            self.api_type = "solutions"
            self.detector = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_hands,
                min_detection_confidence=self.min_detection_con,
                min_tracking_confidence=0.5
            )
        else:
            safe_print("[HandDetector] Initializing with MediaPipe Tasks API (HandLandmarker)...")
            self.api_type = "tasks"
            self._ensure_model_file()
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=self._get_model_path())
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=self.max_hands,
                min_hand_detection_confidence=self.min_detection_con,
                min_hand_presence_confidence=self.min_detection_con,
                min_tracking_confidence=0.5
            )
            self.detector = vision.HandLandmarker.create_from_options(options)

    def _get_model_path(self) -> str:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, self.MODEL_NAME)
        if not os.path.exists(path):
            path = self.MODEL_NAME
        return path

    def _ensure_model_file(self):
        path = self._get_model_path()
        if not os.path.exists(path):
            safe_print(f"[HandDetector] Downloading model file to '{path}'...")
            try:
                urllib.request.urlretrieve(self.MODEL_URL, path)
                safe_print("[HandDetector] Download complete.")
            except Exception as e:
                safe_print(f"[HandDetector] Warning: Auto-download failed: {e}")

    def find_hands(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Process BGR frame and return list of detected hands."""
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_list = []

        if self.api_type == "solutions":
            results = self.detector.process(rgb_frame)
            if results.multi_hand_landmarks:
                for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                    lm_norm = [(lm.x, lm.y, lm.z) for lm in hand_lms.landmark]
                    lm_px = [(int(lm.x * w), int(lm.y * h), lm.z) for lm in hand_lms.landmark]

                    xs = [pt[0] for pt in lm_px]
                    ys = [pt[1] for pt in lm_px]
                    bbox = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

                    handedness = "Hand"
                    if results.multi_handedness and idx < len(results.multi_handedness):
                        handedness = results.multi_handedness[idx].classification[0].label

                    hands_list.append({
                        "landmarks_norm": lm_norm,
                        "landmarks_px": lm_px,
                        "bbox": bbox,
                        "handedness": handedness
                    })

        elif self.api_type == "tasks":
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.detector.detect(mp_image)

            if results.hand_landmarks:
                for idx, hand_lms in enumerate(results.hand_landmarks):
                    lm_norm = [(lm.x, lm.y, lm.z) for lm in hand_lms]
                    lm_px = [(int(lm.x * w), int(lm.y * h), lm.z) for lm in hand_lms]

                    xs = [pt[0] for pt in lm_px]
                    ys = [pt[1] for pt in lm_px]
                    bbox = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

                    handedness = "Hand"
                    if results.handedness and idx < len(results.handedness):
                        handedness = results.handedness[idx][0].category_name

                    hands_list.append({
                        "landmarks_norm": lm_norm,
                        "landmarks_px": lm_px,
                        "bbox": bbox,
                        "handedness": handedness
                    })

        return hands_list

    def draw_hand_skeleton(self, frame: np.ndarray, hand: Dict[str, Any], gesture_label: str = ""):
        """Draw sleek glowing joints and bones on the detected hand inside the player frame."""
        lm_px = hand["landmarks_px"]

        # Bones (vibrant purple/blue connecting lines matching layout theme)
        for p1_idx, p2_idx in self.SKELETON_CONNECTIONS:
            pt1 = (lm_px[p1_idx][0], lm_px[p1_idx][1])
            pt2 = (lm_px[p2_idx][0], lm_px[p2_idx][1])
            cv2.line(frame, pt1, pt2, (255, 120, 100), 2, cv2.LINE_AA)

        # Joints (white center with cyan glow)
        for idx, pt in enumerate(lm_px):
            pos = (pt[0], pt[1])
            if idx in self.FINGER_TIPS:
                cv2.circle(frame, pos, 6, (0, 255, 200), -1, cv2.LINE_AA)
                cv2.circle(frame, pos, 8, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, pos, 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, pos, 4, (200, 100, 255), 1, cv2.LINE_AA)

    def release(self):
        """Release detector resources."""
        if self.detector and hasattr(self.detector, "close"):
            self.detector.close()


# ============================================================================
# 3. ASSET MANAGER & CUSTOM LAYOUT COMPOSITOR
# ============================================================================

class AssetManager:
    """
    Loads, caches, and pre-scales the custom game image assets:
      - bg_layout.png (1024x576)
      - rock.png (pink circular icon)
      - paper.png (purple circular icon)
      - scissors.png (blue circular icon)
    """

    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.assets_dir = os.path.join(script_dir, "assets")

        # Discover paths
        self.bg_path = self._find_asset("bg_layout.png")
        self.rock_path = self._find_asset("rock.png")
        self.paper_path = self._find_asset("paper.png")
        self.scissors_path = self._find_asset("scissors.png")

        # Load Master Layout
        if self.bg_path and os.path.exists(self.bg_path):
            safe_print(f"[AssetManager] Loading custom board layout from: {self.bg_path}")
            self.bg_image_pil = Image.open(self.bg_path).convert("RGB")
        else:
            safe_print("[AssetManager] Creating default board canvas...")
            self.bg_image_pil = Image.new("RGB", (1024, 576), (245, 245, 248))

        # Layout Coordinates (measured exactly from bg_layout.png)
        # AI Box: x=74..391 (width=317), y=188..522 (height=334)
        self.AI_BOX = (74, 188, 317, 334)
        # Player Box: x=637..954 (width=317), y=188..522 (height=334)
        self.PLAYER_BOX = (637, 188, 317, 334)
        # AI Score Tab (Red): center (346, 154)
        self.AI_SCORE_CENTER = (346, 154)
        # Player Score Tab (Green): center (909, 154)
        self.PLAYER_SCORE_CENTER = (909, 154)
        # Center Orange Badge: center (510, 324)
        self.CENTER_BADGE = (422, 279, 177, 91)
        self.CENTER_BADGE_MID = (510, 324)

        # Pre-scale move icons for AI Box (230x230) and mini badges (60x60)
        self.icons_large: Dict[str, Image.Image] = {}
        self.icons_small: Dict[str, Image.Image] = {}

        raw_icons = {
            "Rock": self.rock_path,
            "Paper": self.paper_path,
            "Scissors": self.scissors_path
        }

        for gesture, pth in raw_icons.items():
            if pth and os.path.exists(pth):
                im = Image.open(pth).convert("RGBA")
                self.icons_large[gesture] = im.resize((230, 230), Image.Resampling.LANCZOS)
                self.icons_small[gesture] = im.resize((60, 60), Image.Resampling.LANCZOS)
            else:
                # Fallback colored circles
                safe_print(f"[AssetManager] Warning: Generating placeholder for {gesture}...")
                placeholder = Image.new("RGBA", (230, 230), (0, 0, 0, 0))
                pdraw = ImageDraw.Draw(placeholder)
                pdraw.ellipse([5, 5, 225, 225], fill=(220, 80, 120, 255), outline=(255, 255, 255), width=6)
                self.icons_large[gesture] = placeholder
                self.icons_small[gesture] = placeholder.resize((60, 60), Image.Resampling.LANCZOS)

    def _find_asset(self, filename: str) -> Optional[str]:
        """Search for asset in assets/ or local directory."""
        candidates = [
            os.path.join(self.assets_dir, filename),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
            filename
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None


# ============================================================================
# 4. UI OVERLAY & DISPLAY COMPOSITOR
# ============================================================================

class UIOverlayRenderer:
    """
    Assembles the complete frame on top of bg_layout.png:
      - Crops and inserts player webcam feed into Player Box (right)
      - Inserts active move icon into AI Box (left)
      - Draws crisp scores in AI Red & Player Green tabs
      - Renders countdowns and round results in Center Orange Badge
      - Renders winner celebration modal when match concludes
    """

    def __init__(self, assets: AssetManager):
        self.assets = assets
        self.font_cache: Dict[int, Any] = {}
        self.font_path = self._discover_system_font()

    def _discover_system_font(self) -> Optional[str]:
        candidates = [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/seguiemj.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def get_font(self, size: int, bold: bool = False):
        if not HAS_PIL:
            return None
        key = (size, bold)
        if key not in self.font_cache:
            font = None
            if self.font_path:
                try:
                    font = ImageFont.truetype(self.font_path, size)
                except Exception:
                    font = ImageFont.load_default()
            else:
                font = ImageFont.load_default()
            self.font_cache[key] = font
        return self.font_cache[key]

    def render_game_frame(
        self,
        player_frame_cropped: np.ndarray,
        state: str,
        round_number: int,
        max_rounds: int,
        player_score: int,
        ai_score: int,
        target_score: int,
        countdown_word: str,
        countdown_num: str,
        ai_current_move: str,
        player_current_move: str,
        round_outcome_msg: str,
        cooldown_remaining: float,
        match_winner: Optional[str] = None
    ) -> np.ndarray:
        """
        Produce the final 1024x576 BGR composite frame.
        """
        # 1. Base board canvas
        canvas = self.assets.bg_image_pil.copy()

        # 2. Insert Player Camera Feed into Player Box (637, 188, 317, 334)
        # Convert OpenCV BGR to PIL RGB
        cam_rgb = cv2.cvtColor(player_frame_cropped, cv2.COLOR_BGR2RGB)
        cam_pil = Image.fromarray(cam_rgb)
        canvas.paste(cam_pil, (self.assets.PLAYER_BOX[0], self.assets.PLAYER_BOX[1]))

        # 3. Insert AI Move Icon into AI Box (74, 188, 317, 334)
        ai_icon = self.assets.icons_large.get(ai_current_move)
        if ai_icon:
            # Center 230x230 icon inside 317x334 box
            ai_x = self.assets.AI_BOX[0] + (self.assets.AI_BOX[2] - 230) // 2
            ai_y = self.assets.AI_BOX[1] + (self.assets.AI_BOX[3] - 230) // 2
            canvas.paste(ai_icon, (ai_x, ai_y), ai_icon)
        elif state == "READY":
            # Display gentle pulse or prompt in AI box
            pass

        # 4. Optional: Mini gesture icon badge on Player Box corner
        p_mini = self.assets.icons_small.get(player_current_move)
        if p_mini and player_current_move in ["Rock", "Paper", "Scissors"]:
            px = self.assets.PLAYER_BOX[0] + 12
            py = self.assets.PLAYER_BOX[1] + 12
            canvas.paste(p_mini, (px, py), p_mini)

        # 5. Draw Scores, Status, and Overlays via Pillow
        draw = ImageDraw.Draw(canvas)

        f_score = self.get_font(32)
        f_badge_large = self.get_font(26)
        f_badge_sub = self.get_font(16)
        f_info = self.get_font(18)
        f_celebration = self.get_font(36)
        f_footer = self.get_font(15)

        # A. Scores inside tabs
        # AI Score (Red Box, center 346, 154)
        draw.text(self.assets.AI_SCORE_CENTER, str(ai_score), anchor="mm", font=f_score, fill=(255, 255, 255))
        # Player Score (Green Box, center 909, 154)
        draw.text(self.assets.PLAYER_SCORE_CENTER, str(player_score), anchor="mm", font=f_score, fill=(255, 255, 255))

        # B. Round Info & Match Goal (under header, center y=105)
        round_txt = f"ROUND {round_number} OF {max_rounds}" if round_number <= max_rounds else f"ROUND {round_number}"
        draw.text((510, 100), round_txt, anchor="mm", font=f_info, fill=(50, 60, 120))
        draw.text((510, 122), f"Target: First to {target_score} Wins", anchor="mm", font=f_badge_sub, fill=(110, 120, 145))

        # C. Center Orange Badge texts (center 510, 324)
        cx, cy = self.assets.CENTER_BADGE_MID

        if state == "READY":
            draw.text((cx, cy - 12), "READY", anchor="mm", font=f_badge_large, fill=(255, 255, 255))
            draw.text((cx, cy + 16), "Press [SPACE]", anchor="mm", font=f_badge_sub, fill=(255, 240, 200))

        elif state == "COUNTDOWN":
            draw.text((cx, cy - 12), countdown_word, anchor="mm", font=f_badge_large, fill=(255, 255, 255))
            draw.text((cx, cy + 18), countdown_num, anchor="mm", font=f_badge_sub, fill=(255, 240, 200))

        elif state == "ROUND_RESULT":
            # Round Outcome Banner
            draw.text((cx, cy - 14), round_outcome_msg, anchor="mm", font=f_badge_large, fill=(255, 255, 255))
            draw.text((cx, cy + 16), f"Next: {cooldown_remaining:.1f}s", anchor="mm", font=f_badge_sub, fill=(255, 245, 220))

            # Progress bar for the 3-second cooldown
            bx1, by1, bw, bh = self.assets.CENTER_BADGE
            p_bar_w = bw - 28
            p_bar_h = 6
            progress = max(0.0, min(1.0, (3.0 - cooldown_remaining) / 3.0))
            px1 = cx - p_bar_w // 2
            py1 = by1 + bh - 12
            draw.rectangle([px1, py1, px1 + p_bar_w, py1 + p_bar_h], fill=(220, 140, 50))
            draw.rectangle([px1, py1, px1 + int(p_bar_w * progress), py1 + p_bar_h], fill=(255, 255, 255))

        elif state == "MATCH_OVER":
            draw.text((cx, cy - 12), "MATCH OVER", anchor="mm", font=f_badge_large, fill=(255, 255, 255))
            draw.text((cx, cy + 16), "Press [R]", anchor="mm", font=f_badge_sub, fill=(255, 240, 200))

        # D. Bottom Footer Controls
        foot_y = 552
        draw.text((510, foot_y), "[SPACE] Start Round   •   [R] Restart Match   •   [Q] Quit Game",
                  anchor="mm", font=f_footer, fill=(100, 110, 130))

        # E. Post-Match Celebration Modal (Center Overlay)
        if state == "MATCH_OVER":
            # Dark semi-transparent backdrop card
            overlay_w = 600
            overlay_h = 220
            ox1 = cx - overlay_w // 2
            oy1 = cy - overlay_h // 2 - 10
            ox2 = ox1 + overlay_w
            oy2 = oy1 + overlay_h

            # Draw modal with rounded look
            draw.rectangle([ox1, oy1, ox2, oy2], fill=(20, 24, 38))
            border_c = (0, 240, 200) if match_winner == "PLAYER" else (255, 90, 80)
            draw.rectangle([ox1, oy1, ox2, oy2], outline=border_c, width=4)

            if match_winner == "PLAYER":
                draw.text((cx, oy1 + 45), "🎉 Congratulations! 🏆", anchor="mm", font=f_celebration, fill=(255, 215, 0))
                draw.text((cx, oy1 + 95), "YOU ARE THE RPS CHAMPION!", anchor="mm", font=f_badge_large, fill=(0, 255, 200))
            elif match_winner == "AI":
                draw.text((cx, oy1 + 45), "🤖 AI WON THE SHOWDOWN! 🤖", anchor="mm", font=f_celebration, fill=(255, 100, 90))
                draw.text((cx, oy1 + 95), "Better luck next time!", anchor="mm", font=f_badge_large, fill=(240, 240, 250))
            else:
                draw.text((cx, oy1 + 45), "🤝 IT'S A MATCH TIE! 🤝", anchor="mm", font=f_celebration, fill=(0, 220, 255))
                draw.text((cx, oy1 + 95), "A deadlock duel!", anchor="mm", font=f_badge_large, fill=(240, 240, 250))

            draw.text((cx, oy1 + 140), f"Final Match Score: Player {player_score}  —  {ai_score} AI",
                      anchor="mm", font=f_info, fill=(255, 255, 255))
            draw.text((cx, oy1 + 175), "Press [R] to Play Again   •   Press [Q] to Quit",
                      anchor="mm", font=f_badge_sub, fill=(255, 220, 0))

        # Convert back to OpenCV BGR
        final_bgr = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
        return final_bgr


# ============================================================================
# 5. ROCK, PAPER, SCISSORS GAME ENGINE
# ============================================================================

class RPSGameEngine:
    """
    Manages game state, countdown rhythms, 3-second round transitions,
    score tracking, and match victory conditions.

    States:
      - 'READY': Waiting for player to position hand / press SPACE.
      - 'COUNTDOWN': 3-second animated countdown ("ROCK!", "PAPER!", "SCISSORS!", "SHOOT!").
      - 'ROUND_RESULT': 3-second strict freeze & pause between rounds with outcome display.
      - 'MATCH_OVER': Final match victory celebration with restart ('R') or quit ('Q').
    """

    MAX_ROUNDS = 3
    TARGET_WINS = 2
    COUNTDOWN_DURATION = 3.0  # seconds
    COOLDOWN_DURATION = 3.0   # seconds
    CHOICES = ["Rock", "Paper", "Scissors"]

    def __init__(self):
        self.state = "READY"
        self.round_number = 1
        self.player_score = 0
        self.ai_score = 0
        self.match_winner: Optional[str] = None

        # Timers
        self.state_start_time = time.time()
        self.countdown_end_time = 0.0
        self.cooldown_end_time = 0.0

        # Current choices & messages
        self.player_choice = "Detecting..."
        self.ai_choice = "Rock"
        self.frozen_player_choice = "None"
        self.frozen_ai_choice = "None"
        self.frozen_player_frame: Optional[np.ndarray] = None

        self.round_outcome_msg = ""
        self.round_outcome_sub = ""
        self.countdown_word = "ROCK!"
        self.countdown_num = "3"

        # Smooth gesture stabilizer
        self.gesture_buffer: List[str] = []
        self.buffer_size = 5

    def reset_match(self):
        """Reset the match to Round 1 with zero scores."""
        safe_print("\n=======================================================")
        safe_print("           🔄 RESTARTING ROCK, PAPER, SCISSORS         ")
        safe_print("=======================================================")
        self.state = "READY"
        self.round_number = 1
        self.player_score = 0
        self.ai_score = 0
        self.match_winner = None
        self.player_choice = "Detecting..."
        self.ai_choice = "Rock"
        self.frozen_player_choice = "None"
        self.frozen_ai_choice = "None"
        self.frozen_player_frame = None
        self.round_outcome_msg = ""
        self.round_outcome_sub = ""
        self.gesture_buffer.clear()
        self.state_start_time = time.time()

    def start_countdown(self):
        """Initiate the on-screen countdown for the active round."""
        self.state = "COUNTDOWN"
        now = time.time()
        self.state_start_time = now
        self.countdown_end_time = now + self.COUNTDOWN_DURATION
        self.player_choice = "Detecting..."
        self.ai_choice = "Rock"
        safe_print(f"\n--- [ROUND {self.round_number} of {self.MAX_ROUNDS}] STARTING COUNTDOWN ---")

    def update_live_gesture(self, raw_gesture: str) -> str:
        """Add gesture to rolling buffer and compute majority vote."""
        self.gesture_buffer.append(raw_gesture)
        if len(self.gesture_buffer) > self.buffer_size:
            self.gesture_buffer.pop(0)

        counts = {}
        for g in self.gesture_buffer:
            counts[g] = counts.get(g, 0) + 1
        return max(counts, key=counts.get)

    def update(self, detected_gesture: str, current_player_frame: np.ndarray):
        """Update game state machine based on elapsed time and current gesture."""
        now = time.time()
        smoothed_gesture = self.update_live_gesture(detected_gesture)

        # -------------------------------------------------------------
        # STATE: READY
        # -------------------------------------------------------------
        if self.state == "READY":
            self.player_choice = smoothed_gesture
            # Auto-start if hand detected steadily for 1.8 seconds, or on SPACE
            if smoothed_gesture in self.CHOICES:
                if now - self.state_start_time > 1.8:
                    self.start_countdown()

        # -------------------------------------------------------------
        # STATE: COUNTDOWN ("ROCK!", "PAPER!", "SCISSORS!", "SHOOT!")
        # -------------------------------------------------------------
        elif self.state == "COUNTDOWN":
            rem = self.countdown_end_time - now
            self.player_choice = smoothed_gesture

            # Synchronize AI box icon with countdown stage
            if rem > 2.0:
                self.countdown_word = "ROCK!"
                self.countdown_num = "3"
                self.ai_choice = "Rock"
            elif rem > 1.0:
                self.countdown_word = "PAPER!"
                self.countdown_num = "2"
                self.ai_choice = "Paper"
            elif rem > 0.0:
                self.countdown_word = "SCISSORS!"
                self.countdown_num = "1"
                self.ai_choice = "Scissors"
            else:
                # COUNTDOWN EXPIRED -> FREEZE GESTURES AT "SHOOT!"
                self._execute_shoot(smoothed_gesture, current_player_frame)

        # -------------------------------------------------------------
        # STATE: ROUND_RESULT (Strict 3-second pause)
        # -------------------------------------------------------------
        elif self.state == "ROUND_RESULT":
            rem = max(0.0, self.cooldown_end_time - now)
            if rem <= 0.0:
                self._evaluate_match_progression()

        # -------------------------------------------------------------
        # STATE: MATCH_OVER
        # -------------------------------------------------------------
        elif self.state == "MATCH_OVER":
            pass

    def _execute_shoot(self, player_gesture: str, player_frame: np.ndarray):
        """Freeze gestures at the 'SHOOT!' instant and determine round outcome."""
        now = time.time()
        self.state = "ROUND_RESULT"
        self.cooldown_end_time = now + self.COOLDOWN_DURATION

        # Freeze player gesture & camera frame
        self.frozen_player_choice = player_gesture
        self.player_choice = player_gesture
        self.frozen_player_frame = player_frame.copy()

        # AI picks random gesture
        self.frozen_ai_choice = random.choice(self.CHOICES)
        self.ai_choice = self.frozen_ai_choice

        p = self.frozen_player_choice
        c = self.frozen_ai_choice

        safe_print("\n💥 ================= SHOOT! =================")
        safe_print(f"  Player Choice: {p}")
        safe_print(f"  AI Choice:     {c}")

        if p not in self.CHOICES:
            self.round_outcome_msg = "UNCLEAR!"
            self.round_outcome_sub = "No clean gesture detected. Replaying round!"
            safe_print("  Result: Unknown gesture! Replaying round...")
        elif p == c:
            self.round_outcome_msg = "ROUND TIED!"
            self.round_outcome_sub = f"Both chose {p}. Replaying round {self.round_number}!"
            safe_print(f"  Result: Tied round! Both picked {p}. Replaying round...")
        elif (
            (p == "Rock" and c == "Scissors") or
            (p == "Paper" and c == "Rock") or
            (p == "Scissors" and c == "Paper")
        ):
            self.player_score += 1
            self.round_outcome_msg = "YOU WIN!"
            reason = (
                "Rock crushes Scissors" if p == "Rock" else (
                    "Paper covers Rock" if p == "Paper" else "Scissors cut Paper"
                )
            )
            self.round_outcome_sub = f"{reason}! Point for Player!"
            safe_print(f"  Result: YOU WIN ROUND {self.round_number}! {reason}!")
            self.round_number += 1
        else:
            self.ai_score += 1
            self.round_outcome_msg = "AI WINS!"
            reason = (
                "Rock crushes Scissors" if c == "Rock" else (
                    "Paper covers Rock" if c == "Paper" else "Scissors cut Paper"
                )
            )
            self.round_outcome_sub = f"{reason}! Point for AI!"
            safe_print(f"  Result: AI WINS ROUND {self.round_number}! {reason}!")
            self.round_number += 1

        safe_print(f"  Score: Player [{self.player_score}]  —  [{self.ai_score}] AI")
        safe_print("==============================================")

    def _evaluate_match_progression(self):
        """Check if match victory condition has been met or proceed to next round."""
        match_finished = False

        if self.player_score >= self.TARGET_WINS:
            self.match_winner = "PLAYER"
            match_finished = True
        elif self.ai_score >= self.TARGET_WINS:
            self.match_winner = "AI"
            match_finished = True
        elif self.round_number > self.MAX_ROUNDS:
            if self.player_score > self.ai_score:
                self.match_winner = "PLAYER"
            elif self.ai_score > self.player_score:
                self.match_winner = "AI"
            else:
                self.match_winner = "TIE"
            match_finished = True

        if match_finished:
            self.state = "MATCH_OVER"
            safe_print("\n🏆 ================= MATCH COMPLETE =================")
            if self.match_winner == "PLAYER":
                safe_print("  🎉 Congratulations! 🏆 YOU ARE THE RPS CHAMPION! 🎉")
            elif self.match_winner == "AI":
                safe_print("  🤖 AI WON THE SHOWDOWN! Better luck next time! 💥")
            else:
                safe_print("  🤝 MATCH TIED! A deadlock finish! 🤝")
            safe_print(f"  Final Match Score: Player {self.player_score} - {self.ai_score} AI")
            safe_print("  Press 'R' to Restart a New Match | Press 'Q' to Quit")
            safe_print("======================================================\n")
        else:
            self.start_countdown()


# ============================================================================
# 6. WEBCAM INITIALIZATION & FRAME ADAPTATION
# ============================================================================

def open_webcam(camera_index: int = 0) -> Optional[cv2.VideoCapture]:
    """Attempt to open webcam with fallback across candidate indices."""
    candidate_indices = [camera_index, 0, 1, 2]
    seen = set()
    indices = [x for x in candidate_indices if not (x in seen or seen.add(x))]

    for idx in indices:
        safe_print(f"[Camera] Checking camera index {idx}...")
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(idx)

        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                safe_print(f"[Camera] Successfully initialized camera index {idx}!")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                return cap
            else:
                cap.release()
    return None


def crop_to_player_box(frame: np.ndarray, target_w: int = 317, target_h: int = 334) -> np.ndarray:
    """
    Crop webcam frame to match the target box aspect ratio and resize to (317, 334).
    """
    h, w, _ = frame.shape
    target_aspect = target_w / target_h
    current_aspect = w / h

    if current_aspect > target_aspect:
        # Crop horizontal sides
        crop_w = int(h * target_aspect)
        x_start = (w - crop_w) // 2
        cropped = frame[:, x_start : x_start + crop_w]
    else:
        # Crop vertical sides
        crop_h = int(w / target_aspect)
        y_start = (h - crop_h) // 2
        cropped = frame[y_start : y_start + crop_h, :]

    resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return resized


# ============================================================================
# 7. MAIN APPLICATION LOOP
# ============================================================================

def run_game(camera_index: int = 0, demo_mode: bool = False):
    """
    Main game loop: captures frames, detects hand landmarks on the cropped feed,
    updates RPS game engine, and renders custom image layout.
    """
    safe_print("\n" + "=" * 60)
    safe_print("      🚀 COMPUTER VISION ROCK, PAPER, SCISSORS GAME       ")
    safe_print("=" * 60)
    safe_print("  Custom Assets Loaded:")
    safe_print("    • Board Layout : bg_layout.png")
    safe_print("    • Moves        : rock.png, paper.png, scissors.png")
    safe_print("  Controls:")
    safe_print("    • [SPACE] : Start Round Immediately")
    safe_print("    • [R]     : Restart New Match")
    safe_print("    • [Q]/ESC : Quit Game")
    safe_print("=" * 60 + "\n")

    # 1. Initialize Assets, Hand Detector, Renderer & Game Engine
    assets = AssetManager()
    detector = HandDetector(max_hands=1, min_detection_con=0.5)
    overlay = UIOverlayRenderer(assets)
    game = RPSGameEngine()

    # 2. Initialize Camera
    cap = None
    if not demo_mode:
        cap = open_webcam(camera_index=camera_index)
        if cap is None:
            safe_print("\n[Camera] Warning: No active physical webcam was detected on indices [0, 1, 2].")
            safe_print("Switching automatically to simulated DEMO MODE to demonstrate game loop...\n")
            demo_mode = True

    window_name = "Rock Paper Scissors - Computer Vision Game"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 576)

    # Demo mode gesture cycle
    demo_cycle = ["Rock", "Paper", "Scissors", "Detecting..."]
    demo_idx = 0
    demo_timer = time.time()

    try:
        while True:
            current_time = time.time()

            # A. Obtain player camera frame
            if not demo_mode and cap is not None:
                ret, raw_frame = cap.read()
                if not ret or raw_frame is None:
                    time.sleep(0.03)
                    continue
                # Mirror camera for natural interaction
                raw_frame = cv2.flip(raw_frame, 1)
                # Crop to match the 317x334 Player Box
                player_cam = crop_to_player_box(raw_frame, target_w=317, target_h=334)
            else:
                # Simulated video frame in demo mode
                player_cam = np.full((334, 317, 3), (35, 30, 25), dtype=np.uint8)
                # Simulated animated background grid
                offset = int(current_time * 20) % 30
                for x in range(offset, 317, 30):
                    cv2.line(player_cam, (x, 0), (x, 334), (50, 45, 38), 1)
                for y in range(offset, 334, 30):
                    cv2.line(player_cam, (0, y), (317, y), (50, 45, 38), 1)

            # B. Hand Landmark Tracking on the cropped player frame
            hands = detector.find_hands(player_cam)
            live_gesture = "No Hand"

            if hands:
                hand = hands[0]
                live_gesture = GestureClassifier.classify(hand["landmarks_norm"])
                detector.draw_hand_skeleton(player_cam, hand, gesture_label=live_gesture)
            elif demo_mode:
                if current_time - demo_timer > 3.0:
                    demo_idx = (demo_idx + 1) % len(demo_cycle)
                    demo_timer = current_time
                live_gesture = demo_cycle[demo_idx]

            # C. Update Game State Machine
            game.update(live_gesture, player_cam)

            # In ROUND_RESULT freeze state, display the frozen captured player frame
            display_player_cam = player_cam
            if game.state == "ROUND_RESULT" and game.frozen_player_frame is not None:
                display_player_cam = game.frozen_player_frame

            cooldown_rem = max(0.0, game.cooldown_end_time - current_time)

            # D. Render Complete Game Board Frame with Custom Assets
            final_board = overlay.render_game_frame(
                player_frame_cropped=display_player_cam,
                state=game.state,
                round_number=game.round_number,
                max_rounds=game.MAX_ROUNDS,
                player_score=game.player_score,
                ai_score=game.ai_score,
                target_score=game.TARGET_WINS,
                countdown_word=game.countdown_word,
                countdown_num=game.countdown_num,
                ai_current_move=game.ai_choice,
                player_current_move=game.player_choice,
                round_outcome_msg=game.round_outcome_msg,
                cooldown_remaining=cooldown_rem,
                match_winner=game.match_winner
            )

            # Show window
            cv2.imshow(window_name, final_board)

            # E. Keyboard Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:  # 'Q' or ESC
                safe_print("[App] Quit key pressed. Exiting...")
                break
            elif key == ord('r') or key == ord('R'):              # 'R' to Restart
                game.reset_match()
            elif key == 32:                                       # SPACE to trigger countdown
                if game.state == "READY":
                    game.start_countdown()

    except KeyboardInterrupt:
        safe_print("\n[App] Interrupted by user (Ctrl+C). Cleaning up...")
    except Exception as e:
        safe_print(f"\n[App] Unexpected runtime error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        safe_print("[App] Releasing camera and OpenCV windows...")
        if cap is not None and cap.isOpened():
            cap.release()
        detector.release()
        cv2.destroyAllWindows()
        safe_print("[App] Cleanup complete. Goodbye!\n")


# ============================================================================
# 8. SCRIPT ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Real-Time Computer Vision Rock, Paper, Scissors Game")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--demo", action="store_true", help="Run in simulated demo mode without physical camera")
    args = parser.parse_args()

    run_game(camera_index=args.camera, demo_mode=args.demo)


if __name__ == "__main__":
    main()
