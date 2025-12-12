import threading
import cv2
import os
import subprocess
from deepface import DeepFace

# ================== PATH SETUP ==================
# I'm getting the folder where this script is located,
# and then I build paths to everything inside the AR project.
# This makes the script portable and avoids hardcoding paths.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AR_DIR = os.path.join(BASE_DIR, "AR project")

REF_PATH = os.path.join(AR_DIR, "reference.jpg")      # Reference image used for identity check
EMOTION_SCRIPT = os.path.join(AR_DIR, "emotionDetection.py")  # The script that runs next after unlock

print("[INFO] Using BASE_DIR     =", BASE_DIR)
print("[INFO] Using AR_DIR       =", AR_DIR)
print("[INFO] Reference image    =", REF_PATH)
print("[INFO] Emotion script     =", EMOTION_SCRIPT)

# ================== LOAD & PREPARE THE REFERENCE FACE ==================
# I load the reference image, extract the face from it using DeepFace,
# and store that cropped face for comparisons.

reference_img = cv2.imread(REF_PATH)
if reference_img is None:
    print("ERROR: could not read reference.jpg at:", REF_PATH)
    quit()

ref_faces = DeepFace.extract_faces(
    reference_img,
    detector_backend="opencv",
    enforce_detection=False
)

if len(ref_faces) == 0:
    print("ERROR: no face detected in reference.jpg")
    quit()

ref_face = ref_faces[0]["face"]  # Cropped face region
print("[INFO] Reference face loaded OK.")

# ================== CAMERA INITIALIZATION ==================
# I open the webcam here. If the camera can't open, there's no point continuing.

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: cannot open camera")
    quit()

counter = 0
face_match = False
last_distance = None

# I set these values to control how strict and how often the face verification runs.
THRESHOLD = 0.55              # Lower = stricter match
VERIFY_EVERY_N_FRAMES = 15    # How often I run the heavy AI check

# Variables to track unlock status
unlocked = False
best_distance = None          # The closest match we’ve ever seen (just for info)

# A safeguard to prevent infinite loops if the camera starts failing repeatedly
fail_streak = 0
MAX_FAIL_STREAK = 50

# ================== FACE MATCHING FUNCTION ==================
# This function runs in a background thread so the main camera loop never freezes.
# It compares the live face with the reference face using DeepFace.

def check_face(frame):
    global face_match, last_distance, unlocked, best_distance

    if unlocked:
        return  # Once unlocked, I stop verifying every frame.

    try:
        # Extract face from the camera frame
        faces = DeepFace.extract_faces(
            frame,
            detector_backend="opencv",
            enforce_detection=False
        )

        if len(faces) == 0:
            print("No face detected in frame.")
            face_match = False
            last_distance = None
            return

        live_face = faces[0]["face"]

        # Compare the live face with the stored reference face
        result = DeepFace.verify(
            ref_face,
            live_face,
            model_name="VGG-Face",
            distance_metric="cosine",
            enforce_detection=False
        )

        dist = float(result["distance"])
        last_distance = dist
        face_match = dist <= THRESHOLD

        print(f"Distance: {dist:.4f} | Threshold: {THRESHOLD:.4f}")

        # Track best match distance just for debugging
        if best_distance is None or dist < best_distance:
            best_distance = dist

        # If the distance is good enough, consider the user verified
        if face_match and not unlocked:
            unlocked = True
            print(f"[INFO] >>> UNLOCKED (distance={dist:.4f})")

    except Exception as e:
        print("DeepFace error:", e)
        face_match = False
        last_distance = None

# ================== MAIN LOOP (CAMERA + DISPLAY) ==================
# This loop keeps reading frames from the camera, displaying text,
# and occasionally running the face verification in the background.

HOLD_FRAMES_AFTER_UNLOCK = 60  # Display "UNLOCKED" briefly before closing
hold_counter = 0

while True:
    ret, frame = cap.read()

    # If the camera fails too many times in a row, exit safely.
    if not ret:
        fail_streak += 1
        print("[WARN] Failed to grab frame, skipping. Streak =", fail_streak)
        if fail_streak > MAX_FAIL_STREAK:
            print("[ERROR] Too many camera failures, exiting.")
            break
        continue
    else:
        fail_streak = 0

    # Run identity check only occasionally (to stay fast)
    if (not unlocked) and (counter % VERIFY_EVERY_N_FRAMES == 0):
        threading.Thread(
            target=check_face,
            args=(frame.copy(),),
            daemon=True
        ).start()
    counter += 1

    # ---- Update text overlay ----
    if unlocked:
        text = "MATCH ✅ UNLOCKED"
        color = (0, 255, 0)
        hold_counter += 1
    else:
        if face_match:
            text = "MATCH ✅"
            color = (0, 255, 0)
        else:
            text = "NO MATCH"
            color = (0, 0, 255)

    # Draw match status
    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display last distance value (for debugging)
    if last_distance is not None:
        cv2.putText(frame, f"d={last_distance:.3f}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

    cv2.imshow("Face Detection", frame)

    # After unlocking, keep the window open briefly before moving on
    if unlocked and hold_counter >= HOLD_FRAMES_AFTER_UNLOCK:
        print("[INFO] Hold time after unlock reached, closing window.")
        break

    # Allow user to quit manually
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("[INFO] 'q' pressed, exiting.")
        unlocked = False
        break

# ================== CLEAN UP CAMERA ==================

cap.release()
cv2.destroyAllWindows()

# ================== LAUNCH EMOTION DETECTION IF UNLOCKED ==================
# Once the identity is confirmed, I run the next script (emotionDetection.py)
# automatically. This happens in the background so the window doesn't freeze.

if unlocked:
    if os.path.exists(EMOTION_SCRIPT):
        print("[INFO] Face verified. Launching emotionDetection.py ...")
        try:
            subprocess.Popen(
                ["/usr/bin/python3", EMOTION_SCRIPT]
            )
            print("[INFO] emotionDetection.py started; exiting faceDetection.py.")
        except Exception as e:
            print("[ERROR] Could not run emotionDetection.py:", e)
    else:
        print("[WARN] emotionDetection.py not found at:", EMOTION_SCRIPT)

    raise SystemExit(0)

else:
    print("[INFO] Not unlocked; emotionDetection.py will NOT run.")
