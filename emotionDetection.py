import cv2
from deepface import DeepFace
from collections import deque
import os
import numpy as np

print(">>> Emotion detection script started")

# ---------- WHERE I SAVE THE HAPPY FACE ----------
# This is the folder where I keep the latest "happy" face image.
# Unity also looks here to use that image as a texture.
UNITY_PROJECT_DIR = "/Users/dhiraj/Emotion3D"
OUTPUT_FILENAME = "happy_capture.jpg"
OUTPUT_PATH = os.path.join(UNITY_PROJECT_DIR, OUTPUT_FILENAME)

# ---------- OPENCV FACE DETECTOR ----------
# I use a standard Haar Cascade model from OpenCV to find faces in each frame.
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------- GLOBAL FLAGS FOR THE BUTTON ----------
# happy_saved -> becomes True once I have at least one happy face saved.
# button_pressed -> becomes True when the on-screen button is clicked.
button_pressed = False
happy_saved = False

# Button position in the "Emotion Detection" window
BTN_X1, BTN_Y1 = 20, 110
BTN_X2, BTN_Y2 = 260, 150   # bottom-right corner of the button


def mouse_callback(event, x, y, flags, param):
    """
    Mouse handler for the Emotion Detection window.

    If I click inside the button area and a happy face has already been saved,
    then I flag button_pressed = True so that the 3D viewer can be opened.
    """
    global button_pressed, happy_saved

    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if the click was inside the button rectangle
        if BTN_X1 <= x <= BTN_X2 and BTN_Y1 <= y <= BTN_Y2:
            if happy_saved and os.path.exists(OUTPUT_PATH):
                print("[INFO] Button clicked: opening 3D cube face viewer...")
                button_pressed = True
            else:
                print("[WARN] No happy face saved yet. Make a happy face first :)")


# ---------- 3D CUBE FACE VIEWER ----------
def launch_face_viewer(img_path: str):
    """
    This function opens a separate window that shows my face on a fake 3D cube.

    Controls in the 3D window:
        W / A / S / D  -> move the cube up/left/down/right
        J / L          -> rotate the cube (yaw)
        U / I          -> zoom out / zoom in
        R              -> reset everything
        Q              -> close the 3D viewer
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Could not read {img_path}")
        return

    # --- Crop a square from the middle of the face and resize it as texture ---
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    face_tex = img[y0:y0 + side, x0:x0 + side]

    tex_size = 400
    face_tex = cv2.resize(face_tex, (tex_size, tex_size))

    # This is the size of the window where I draw the cube
    canvas_size = 800
    viewer_name = "3D Face Viewer"

    # Starting parameters for the cube
    yaw_deg = 0.0       # rotation angle
    scale = 1.0         # zoom level
    offset_x = 0        # horizontal shift
    offset_y = 0        # vertical shift
    dist = 3.0          # fake distance from camera

    cv2.namedWindow(viewer_name)

    def project_square(yaw_deg, scale, offset_x, offset_y):
        """
        This function takes a square in 3D space, rotates it (yaw),
        and projects it into 2D so OpenCV can draw it.
        """
        theta = np.deg2rad(yaw_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        s = scale
        # 3D coordinates of the corners of a square
        pts_3d = [
            (-s, -s, 0),
            ( s, -s, 0),
            ( s,  s, 0),
            (-s,  s, 0),
        ]

        # Simple pinhole camera model for projection
        fx = fy = canvas_size * 0.6
        cx = canvas_size / 2 + offset_x
        cy = canvas_size / 2 + offset_y

        dst = []
        for x, y, z in pts_3d:
            # Rotate around the Y axis (yaw)
            x_r = x * cos_t + z * sin_t
            z_r = -x * sin_t + z * cos_t + dist
            y_r = y

            # Perspective projection into 2D
            u = fx * x_r / z_r + cx
            v = fy * y_r / z_r + cy
            dst.append([u, v])

        return np.float32(dst)

    while True:
        # Blank canvas to draw the cube frame
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

        # Where the corners of the face texture should map to after rotation
        dst_pts = project_square(yaw_deg, scale, offset_x, offset_y)

        # Source points (the corners of the texture image)
        src_pts = np.float32([
            [0, 0],
            [tex_size, 0],
            [tex_size, tex_size],
            [0, tex_size],
        ])

        # Compute perspective transform and warp the face onto the canvas
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        cv2.warpPerspective(
            face_tex, H, (canvas_size, canvas_size),
            dst=canvas,
            borderMode=cv2.BORDER_TRANSPARENT
        )

        # Fake shading effect based on rotation angle
        shade = 0.4 + 0.6 * abs(np.cos(np.deg2rad(yaw_deg)))
        canvas = (canvas.astype(np.float32) * shade).clip(0, 255).astype(np.uint8)

        # Helper text drawn on the 3D viewer
        help_lines = [
            "3D Cube Face Viewer",
            "W/A/S/D: move | J/L: rotate",
            "U/I: zoom out/in | R: reset | Q: quit",
        ]
        y_text = 25
        for line in help_lines:
            cv2.putText(canvas, line, (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
            y_text += 25

        cv2.imshow(viewer_name, canvas)
        key = cv2.waitKey(30) & 0xFF

        # Keyboard controls for the cube
        if key == ord('q'):
            break
        elif key == ord('w'):
            offset_y -= 10
        elif key == ord('s'):
            offset_y += 10
        elif key == ord('a'):
            offset_x -= 10
        elif key == ord('d'):
            offset_x += 10
        elif key == ord('j'):
            yaw_deg -= 5
        elif key == ord('l'):
            yaw_deg += 5
        elif key == ord('u'):
            scale = max(0.4, scale - 0.05)
        elif key == ord('i'):
            scale = min(1.8, scale + 0.05)
        elif key == ord('r'):
            # Reset everything
            yaw_deg = 0.0
            scale = 1.0
            offset_x = 0
            offset_y = 0
            dist = 3.0

    cv2.destroyWindow(viewer_name)


# ---------- CAMERA SETUP ----------
# I open the webcam and start reading frames.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: cannot open camera")
    exit()

counter = 0
emotion_text = "Detecting..."
smoothed_emotion = "unknown"

# I use a small history of emotions to make the result less jumpy.
emotion_history = deque(maxlen=7)

# Colors for different emotions (BGR)
emotion_colors = {
    "happy":   (0, 255, 0),
    "angry":   (0, 0, 255),
    "surprise":(0, 255, 255),
    "sad":     (255, 0, 0),
    "neutral": (42, 42, 165),
}

# Create the main window and plug in the mouse callback for the button
cv2.namedWindow("Emotion Detection")
cv2.setMouseCallback("Emotion Detection", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # ---- EMOTION DETECTION ----
    # I don't run DeepFace on every single frame to keep things smoother.
    if counter % 12 == 0 and len(faces) > 0:
        try:
            # Use the largest detected face if there are multiple
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            face_roi = frame[y:y+h, x:x+w]

            analysis = DeepFace.analyze(
                img_path=face_roi,
                actions=["emotion"],
                enforce_detection=False
            )

            # DeepFace sometimes returns a list, so I normalise it to a dict
            if isinstance(analysis, list):
                analysis = analysis[0]

            emo_dict = analysis.get("emotion", {})

            # Make "sad" a bit more sensitive by boosting its score
            if "sad" in emo_dict:
                emo_dict["sad"] *= 1.3

            # Pick the emotion with the highest confidence
            dominant = max(emo_dict, key=emo_dict.get)

            # Add to history and compute the most common recent emotion
            emotion_history.append(dominant)
            smoothed_emotion = max(set(emotion_history), key=emotion_history.count)
            emotion_text = f"Emotion: {smoothed_emotion}"

            print("Detected:", smoothed_emotion)

            # Every time I see "happy", I save the latest cropped face.
            if smoothed_emotion == "happy":
                os.makedirs(UNITY_PROJECT_DIR, exist_ok=True)
                cv2.imwrite(OUTPUT_PATH, face_roi)
                happy_saved = True
                print(f"[INFO] Saved happy face to: {OUTPUT_PATH}")
                print("[INFO] Click the button or press 'B' to open 3D viewer.")

        except Exception as e:
            print("DeepFace error:", e)

    counter += 1

    # ---- DRAW FACE BOXES ----
    for (x, y, w, h) in faces:
        # Green rectangle around all detected faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # Pick colour for text based on emotion
    bgr = emotion_colors.get(smoothed_emotion, (255, 255, 255))

    # Show the current emotion on screen
    cv2.putText(frame, emotion_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, bgr, 2)

    # Small help text
    cv2.putText(frame, "Q: quit | B: 3D viewer", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # ---- DRAW THE ON-SCREEN BUTTON ----
    if happy_saved:
        btn_color = (0, 255, 0)      # Green when it's usable
        text_color = (0, 0, 0)
    else:
        btn_color = (100, 100, 100)  # Grey when disabled
        text_color = (200, 200, 200)

    cv2.rectangle(frame, (BTN_X1, BTN_Y1), (BTN_X2, BTN_Y2), btn_color, -1)
    cv2.putText(frame, "Create 3D Face", (BTN_X1 + 10, BTN_Y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    cv2.imshow("Emotion Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    # ---------- KEYBOARD CONTROLS ----------
    if key == ord('q'):
        # Quit the emotion detection app
        break     
    elif key == ord('b'):
        # Open 3D viewer using the latest happy face (if available)
        if happy_saved and os.path.exists(OUTPUT_PATH):
            print("[INFO] Opening 3D cube face viewer (via B key)...")
            launch_face_viewer(OUTPUT_PATH)
            print("[INFO] Back to emotion detection.")
        else:
            print("[WARN] No happy face saved yet. Make a happy face first :)")

    # ---------- MOUSE BUTTON CLICK HANDLING ----------
    # If the on-screen button was clicked, open the 3D viewer.
    if button_pressed:
        button_pressed = False
        launch_face_viewer(OUTPUT_PATH)
        print("[INFO] Back to emotion detection.")

cap.release()
cv2.destroyAllWindows()
print(">>> Emotion detection script finished")
