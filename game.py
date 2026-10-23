import random
import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import time


def draw_centered_text(img, text, cx, cy, font=cv2.FONT_HERSHEY_PLAIN, scale=3, thickness=4, color=(255, 255, 255)):
    """Helper to draw text centered horizontally and vertically at (cx, cy)."""
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = cx - w // 2
    y = cy + h // 2
    cv2.putText(img, text, (x, y), font, scale, color, thickness)


def get_player_move(fingers):
    """
    Determines gesture from finger states [thumb, index, middle, ring, pinky].
    Focuses on the 4 primary fingers (index, middle, ring, pinky) to make
    detection robust across left/right hands and varying thumb orientations:
      - Rock: 4 fingers closed [0, 0, 0, 0]
      - Paper: 4 fingers open [1, 1, 1, 1]
      - Scissors: index & middle open [1, 1], ring & pinky closed [0, 0]
    """
    if fingers[1:] == [0, 0, 0, 0]:
        return 1  # Rock
    elif fingers[1:] == [1, 1, 1, 1]:
        return 2  # Paper
    elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
        return 3  # Scissors
    return None


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    # Detect up to 2 hands so showing both hands can start/restart the game
    detector = HandDetector(maxHands=2, detectionCon=0.7)

    # Window setup: Fullscreen mode
    window_name = "Rock Paper Scissors"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    scores = [0, 0]  # [AI, Player]
    state = "WAITING"  # "WAITING", "COUNTDOWN", "RESULT"
    initialTime = 0
    resultTime = 0
    resultText = ""
    resultColor = (255, 255, 255)
    imgAI = None
    is_fullscreen = True

    while True:
        imgBG = cv2.imread("Resources/BG.png")
        if imgBG is None:
            print("Error: Could not load Resources/BG.png")
            break

        success, img = cap.read()
        if not success or img is None:
            continue

        # Resize and center-crop camera feed to fit 400x420 slot at imgBG[234:654, 795:1195]
        h, w, _ = img.shape
        scale = 420.0 / h
        new_w = int(w * scale)
        img_resized = cv2.resize(img, (new_w, 420))
        if new_w >= 400:
            start_x = (new_w - 400) // 2
            imgScaled = img_resized[:, start_x:start_x + 400]
        else:
            imgScaled = cv2.resize(img, (400, 420))

        # Detect hands (supports up to 2 hands)
        hands, imgScaled = detector.findHands(imgScaled, draw=True)

        current_time = time.time()

        # Handle Game States
        if state == "WAITING":
            # Display prompt in center box
            draw_centered_text(imgBG, "SHOW 2", 639, 390, scale=2.3, thickness=3, color=(255, 255, 255))
            draw_centered_text(imgBG, "HANDS", 639, 425, scale=2.3, thickness=3, color=(255, 255, 255))
            draw_centered_text(imgBG, "Show 2 hands (or press 'S') to start!", 640, 685, scale=1.6, thickness=2, color=(50, 50, 50))

            # Trigger start if 2 hands are detected
            if len(hands) >= 2:
                state = "COUNTDOWN"
                initialTime = current_time
                imgAI = None

        elif state == "COUNTDOWN":
            elapsed = current_time - initialTime
            timeLeft = 3 - int(elapsed)

            if timeLeft > 0:
                # Countdown: 3, 2, 1
                draw_centered_text(imgBG, str(timeLeft), 639, 410, scale=6, thickness=6, color=(255, 0, 255))
                draw_centered_text(imgBG, "Get Ready! Show Rock, Paper, or Scissors!", 640, 685, scale=1.6, thickness=2, color=(50, 50, 50))
            else:
                # 3 seconds elapsed: evaluate player's move and AI's move
                randomNumber = random.randint(1, 3)
                imgAI = cv2.imread(f"Resources/{randomNumber}.png", cv2.IMREAD_UNCHANGED)

                playerMove = None
                if hands:
                    # Look through detected hands for a valid gesture
                    for hand in hands:
                        fingers = detector.fingersUp(hand)
                        move = get_player_move(fingers)
                        if move is not None:
                            playerMove = move
                            break

                # Determine Winner and update scores
                if playerMove is None:
                    resultText = "RETRY!"
                    resultColor = (0, 165, 255)
                elif playerMove == randomNumber:
                    resultText = "TIE!"
                    resultColor = (0, 215, 255)
                elif (playerMove == 1 and randomNumber == 3) or \
                     (playerMove == 2 and randomNumber == 1) or \
                     (playerMove == 3 and randomNumber == 2):
                    scores[1] += 1
                    resultText = "YOU WIN!"
                    resultColor = (0, 200, 0)
                else:
                    scores[0] += 1
                    resultText = "AI WINS!"
                    resultColor = (0, 0, 255)

                state = "RESULT"
                resultTime = current_time

        elif state == "RESULT":
            # Display result in the center box
            draw_centered_text(imgBG, resultText, 639, 406, scale=2.3, thickness=3, color=resultColor)

            res_elapsed = current_time - resultTime
            if res_elapsed >= 2.0:
                # After 2 seconds cooldown, allow restarting with 2 hands
                draw_centered_text(imgBG, "Show 2 hands (or press 'S') to play again!", 640, 685, scale=1.6, thickness=2, color=(50, 50, 50))
                if len(hands) >= 2:
                    state = "COUNTDOWN"
                    initialTime = current_time
                    imgAI = None
            else:
                draw_centered_text(imgBG, "Round finished!", 640, 685, scale=1.6, thickness=2, color=(80, 80, 80))

        # Paste webcam frame into Player section
        imgBG[234:654, 795:1195] = imgScaled

        # Overlay AI move image if available
        if imgAI is not None and state == "RESULT":
            imgBG = cvzone.overlayPNG(imgBG, imgAI, (149, 310))

        # Draw AI and Player scores centered inside their respective scoreboard boxes
        # AI score box center: (433, 193), Player score box center: (1137, 193)
        draw_centered_text(imgBG, str(scores[0]), 433, 193, scale=3, thickness=4, color=(255, 255, 255))
        draw_centered_text(imgBG, str(scores[1]), 1137, 193, scale=3, thickness=4, color=(255, 255, 255))

        # Show window
        cv2.imshow(window_name, imgBG)

        key = cv2.waitKey(1) & 0xFF

        # Keyboard shortcuts
        if key == ord('s'):
            state = "COUNTDOWN"
            initialTime = time.time()
            imgAI = None
        elif key == ord('r'):
            # Reset scores
            scores = [0, 0]
        elif key == ord('f'):
            # Toggle fullscreen
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        elif key == ord('q') or key == 27:  # 'q' or ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
