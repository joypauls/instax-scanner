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
    frame_annotated = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # [:, :, 0]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 100, 150)

    contours, _ = cv2.findContours(
        edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # filter through contours to find the outside of the instax film
    largest_contour = None
    # largest_contour_approx = None
    max_area = 0
    for contour in contours:
        # cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
        area = cv2.contourArea(contour)
        if area > max_area and area > 10000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
            # cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            if len(approx) == 4:
                max_area = area
                largest_contour = contour
                # largest_contour_approx = approx
            # max_area = area
            # largest_contour = contour

    if largest_contour is not None:
        cv2.drawContours(frame_annotated, [largest_contour], -1, (0, 255, 0), 3)
        # cv2.drawContours(frame_annotated, [largest_contour_approx], -1, (0, 255, 0), 3)

    else:
        print("didn't find the outside instax film")
        # cv2.imwrite("no_contour.jpg", frame_annotated)

    cv2.imshow("Camera Feed", frame_annotated)
    # cv2.imshow("DEBUG", gray)
    # cv2.imshow("DEBUG", edged)
    # cv2.imshow("DEBUG", blurred)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
