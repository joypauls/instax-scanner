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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        if len(approx) == 4:
            outerContour = approx
            break

    if outerContour is not None:
        # Draw outer rectangle and its corner points on the original frame
        cv2.drawContours(frame, [outerContour], -1, (0, 255, 0), 2)
        for point in outerContour:
            cv2.circle(frame, tuple(point[0]), 5, (0, 0, 255), -1)

        # Warp the outer rectangle to a top-down view.
        pts = outerContour.reshape(4, 2)
        rect = order_points(pts)
        widthA = np.linalg.norm(rect[2] - rect[3])
        widthB = np.linalg.norm(rect[1] - rect[0])
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(rect[1] - rect[2])
        heightB = np.linalg.norm(rect[0] - rect[3])
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array(
            [
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1],
            ],
            dtype="float32",
        )
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
        # Save a clean copy before drawing inner markers.
        warped_clean = warped.copy()

        # === In the warped image, detect the inner rectangle ===
        # Convert to grayscale and equalize to handle varying brightness
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        warped_eq = clahe.apply(warped_gray)
        warped_blurred = cv2.GaussianBlur(warped_eq, (5, 5), 0)
        warped_edged = cv2.Canny(warped_blurred, 50, 150)

        warped_contours, _ = cv2.findContours(
            warped_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        warped_contours = sorted(warped_contours, key=cv2.contourArea, reverse=True)

        # Define the warped outer boundary as a quadrilateral (the full image)
        warped_outer = np.array(
            [
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1],
            ],
            dtype="float32",
        )
        warped_outer = warped_outer.reshape(-1, 1, 2).astype(int)
        warped_area = maxWidth * maxHeight

        innerContour = None
        bestInnerArea = 0
        for cnt in warped_contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                # Exclude contours that nearly cover the entire warped image.
                if area < warped_area * 0.9:
                    allInside = True
                    for point in approx:
                        pt = (int(point[0][0]), int(point[0][1]))
                        if cv2.pointPolygonTest(warped_outer, pt, False) < 0:
                            allInside = False
                            break
                    if allInside and area > bestInnerArea:
                        bestInnerArea = area
                        innerContour = approx

        # If an inner rectangle is found in the warped image, draw it on the warped feed
        if innerContour is not None:
            cv2.drawContours(warped, [innerContour], -1, (255, 0, 0), 2)
            for point in innerContour:
                cv2.circle(warped, tuple(point[0]), 5, (0, 255, 255), -1)

            # --- Extract the innermost rectangle from the clean warped image ---
            inner_pts = innerContour.reshape(4, 2)
            inner_rect = order_points(inner_pts)
            widthA_inner = np.linalg.norm(inner_rect[2] - inner_rect[3])
            widthB_inner = np.linalg.norm(inner_rect[1] - inner_rect[0])
            innerWidth = max(int(widthA_inner), int(widthB_inner))
            heightA_inner = np.linalg.norm(inner_rect[1] - inner_rect[2])
            heightB_inner = np.linalg.norm(inner_rect[0] - inner_rect[3])
            innerHeight = max(int(heightA_inner), int(heightB_inner))

            dst_inner = np.array(
                [
                    [0, 0],
                    [innerWidth - 1, 0],
                    [innerWidth - 1, innerHeight - 1],
                    [0, innerHeight - 1],
                ],
                dtype="float32",
            )
            M_inner = cv2.getPerspectiveTransform(inner_rect, dst_inner)
            innerWarp = cv2.warpPerspective(
                warped_clean, M_inner, (innerWidth, innerHeight)
            )
            # Show the innermost rectangle without any annotations.
            cv2.imshow("Inner Rectangle", innerWarp)

        cv2.imshow("Warped Feed", warped)

    cv2.imshow("Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
