import cv2
import numpy as np


def order_points(pts):
    # Order points as: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# Open the default camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === Detect outer (polaroid) rectangle in the original frame ===
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # [:, :, 0]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(
        edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    outerContour = None
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        outerContour = approx
        # if len(approx) == 4:
        #     outerContour = approx
        #     break

    if outerContour is not None:
        # Draw outer rectangle and its corner points on the original frame
        cv2.drawContours(frame, [outerContour], -1, (0, 255, 0), 2)

        # for point in outerContour:
        #     cv2.circle(frame, tuple(point[0]), 5, (0, 0, 255), -1)

    # cv2.imshow("Camera Feed", frame)
    cv2.imshow("DEBUG", gray)
    # cv2.imshow("DEBUG", edged)
    # cv2.imshow("DEBUG", blurred)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
