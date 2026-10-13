import os
import time
from collections import deque

import cv2
from ultralytics import YOLO

# ---- Settings you can tune ----
CONF = 0.3            # minimum confidence to accept a detection
IMG_SIZE = 960        # inference size (use 640 if it's laggy)
CONFIRM_FRAMES = 2    # frames in a row a card must be seen before it counts
HOLD_SECONDS = 1.0    # keep a card "on the table" this long after last seen

FONT = cv2.FONT_HERSHEY_SIMPLEX

model = YOLO("best.pt")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

os.makedirs("samples", exist_ok=True)

hits = {}               # card name -> consecutive frames seen
last_seen = {}          # card name -> time it was last detected
active = set()          # cards currently counted as on the table
log = deque(maxlen=5)   # most recent events

prev = time.time()


def hand_value(cards):
    total, aces = 0, 0
    for c in cards:
        rank = c[:-1]                 # "10H" -> "10", "KS" -> "K"
        if rank == "A":
            total += 11
            aces += 1
        elif rank in ("J", "Q", "K", "10"):
            total += 10
        else:
            total += int(rank)
    while total > 21 and aces:        # count an ace as 1 instead of 11 if we'd bust
        total -= 10
        aces -= 1
    return total


def add_event(text):
    line = f"{time.strftime('%H:%M:%S')} {text}"
    log.append(line)
    print(line)


while True:
    ok, frame = cap.read()
    if not ok:
        break
    now = time.time()
    fps = 1.0 / max(now - prev, 1e-6)
    prev = now

    results = model(frame, conf=CONF, imgsz=IMG_SIZE, verbose=False)[0]
    annotated = results.plot()

    # Each card has two corner indices, so one card can be reported twice.
    # A deck has one of each card, so the SET of labels = the unique cards.
    detected = {model.names[int(c)] for c in results.boxes.cls}

    for name in detected:
        hits[name] = hits.get(name, 0) + 1
        last_seen[name] = now
        if hits[name] >= CONFIRM_FRAMES and name not in active:
            active.add(name)
            add_event(f"Card added: {name}")

    for name in list(hits):
        if name not in detected:
            hits[name] = 0

    for name in list(active):
        if now - last_seen[name] > HOLD_SECONDS:
            active.remove(name)
            add_event(f"Card removed: {name}")

    # Blackjack total
    total = hand_value(active)
    status = ""
    if total == 21:
        status = "  BLACKJACK!" if len(active) == 2 else "  21!"
    elif total > 21:
        status = "  BUST"

    if active:
        line1 = f"Cards: {len(active)}  ({', '.join(sorted(active))})"
        line2 = f"Blackjack total: {total}{status}"
    else:
        line1 = "No cards detected"
        line2 = ""

    cv2.putText(annotated, line1, (10, 30), FONT, 0.8, (0, 255, 0), 2)
    if line2:
        cv2.putText(annotated, line2, (10, 65), FONT, 0.8, (0, 255, 255), 2)

    # FPS counter, top right
    cv2.putText(annotated, f"FPS: {fps:.1f}", (annotated.shape[1] - 160, 30),
                FONT, 0.7, (255, 255, 0), 2)

    # Event log, newest at the bottom
    h = annotated.shape[0]
    for i, entry in enumerate(log):
        y = h - 15 - 25 * (len(log) - 1 - i)
        cv2.putText(annotated, entry, (10, y), FONT, 0.55, (255, 255, 255), 1)

    cv2.imshow("Card detector", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key in (ord("s"), ord("S")):
        path = f"samples/{time.strftime('%H%M%S')}.jpg"
        cv2.imwrite(path, annotated)
        print("Saved", path)

cap.release()
cv2.destroyAllWindows()