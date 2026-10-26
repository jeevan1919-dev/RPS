# Real-Time Computer Vision Rock, Paper, Scissors Showdown

An interactive, computer vision-based **Rock, Paper, Scissors** game built in Python using OpenCV, MediaPipe, and Pillow — styled with custom image assets!

---

## 🎨 Custom Visual Assets Included

1. **Master Game Board (`assets/bg_layout.png`)**:
   - Master layout canvas (`1024 x 576`).
   - Left side: **AI Box** with Red Score tab.
   - Right side: **Player Box** with Green Score tab where your live mirrored camera feed and hand skeleton are embedded.
   - Center: **Orange Badge** displaying countdowns, shoot triggers, round outcomes, and timer progress bars.

2. **Move Icons**:
   - 🪨 **Rock (`assets/rock.png`)**: Custom pink circular hand icon.
   - 📄 **Paper (`assets/paper.png`)**: Custom purple circular hand icon.
   - ✂️ **Scissors (`assets/scissors.png`)**: Custom blue circular hand icon.

---

## 🌟 Key Features

1. **Robust Hand Tracking & Landmark Geometry**:
   - Uses MediaPipe Hands to detect 21 3D landmarks.
   - Supports both **MediaPipe Tasks API** (`mediapipe >= 0.10.x / 1.x`) and legacy **MediaPipe Solutions API** (`mediapipe.solutions.hands`).
   - Geometry-based finger curling:
     - 🪨 **Rock**: All 4 main fingers curled tight into a fist.
     - 📄 **Paper**: All 5 fingers extended flat.
     - ✂️ **Scissors**: Index and middle fingers extended; ring, pinky, and thumb curled.
     - ⏳ **Detecting...**: Ambiguous or transitional hand gestures.

2. **Match Structure & Rules**:
   - **Best of 3 Rounds** (First to 2 wins or leader after 3 decisive rounds).
   - Random AI move generation for each round.
   - Standard rules: Rock crushes Scissors, Scissors cuts Paper, Paper covers Rock.
   - Unclear gestures or tied rounds trigger clean replays without penalizing players.

3. **Synchronized Countdown & Strict 3-Second Cooldown**:
   - The AI window visually cycles through your custom icons during the countdown:
     - `"ROCK! (3)"` with the pink Rock icon
     - `"PAPER! (2)"` with the purple Paper icon
     - `"SCISSORS! (1)"` with the blue Scissors icon
     - `"SHOOT!"` where player gesture freezes and AI reveals its final move!
   - Enforces a strict **3-second delay/pause** between consecutive rounds with a countdown timer and animated visual progress bar.

4. **Post-Match Winner Celebration**:
   - Once a player reaches 2 wins (or 3 rounds complete), a victory celebration overlay declares the match winner:
     - **"🎉 Congratulations! 🏆 YOU ARE THE RPS CHAMPION!"**
   - Press **`[R]`** to restart a new match or **`[Q]`** to quit.

---

## 🚀 Getting Started

### 1. Requirements
```bash
pip install opencv-python mediapipe pillow numpy
```

### 2. Run the Game
Run the game using your default webcam (index `0`):
```bash
python rps_game.py
```

To specify a different camera index:
```bash
python rps_game.py --camera 1
```

To run the game in simulated demo mode (ideal for testing without a webcam):
```bash
python rps_game.py --demo
```

---

## 🎮 How to Play

1. Position your hand in front of the camera so the tracked skeleton appears inside the right **PLAYER** box.
2. The game will automatically start Round 1 once your hand is recognized (or press **`[SPACE]`**).
3. Follow the center orange countdown badge:
   - **ROCK!** (3)
   - **PAPER!** (2)
   - **SCISSORS!** (1)
   - **SHOOT!**
4. Strike your gesture at **"SHOOT!"**. Your gesture freezes and compares against the AI's icon.
5. The strict **3-second cooldown** gives you time to reset your hand position for the next round.
6. The first to reach 2 round wins claims the championship trophy!
7. Press **`[R]`** to restart a new match or **`[Q]`** to quit.
