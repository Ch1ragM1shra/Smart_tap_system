

import cv2
import numpy as np
import argparse


def nothing(x):
    pass


def tune(image_path: str):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not read {image_path}")
        return

    frame = cv2.resize(frame, (640, 480))
    cv2.namedWindow("Tuner", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask",  cv2.WINDOW_NORMAL)

    # Create trackbars
    for name, default in [
        ("H_low", 0), ("H_high", 180),
        ("S_low", 100), ("S_high", 255),
        ("V_low", 70),  ("V_high", 255),
    ]:
        cv2.createTrackbar(name, "Tuner", default, 255 if name != "H_high" else 180, nothing)
    cv2.createTrackbar("H_high", "Tuner", 180, 180, nothing)

    print("Adjust sliders until the rainbow tag is WHITE in the mask.")
    print("Press S to print the values, Q to quit.")

    while True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lo = np.array([
            cv2.getTrackbarPos("H_low", "Tuner"),
            cv2.getTrackbarPos("S_low", "Tuner"),
            cv2.getTrackbarPos("V_low", "Tuner"),
        ])
        hi = np.array([
            cv2.getTrackbarPos("H_high", "Tuner"),
            cv2.getTrackbarPos("S_high", "Tuner"),
            cv2.getTrackbarPos("V_high", "Tuner"),
        ])

        mask   = cv2.inRange(hsv, lo, hi)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow("Tuner", frame)
        cv2.imshow("Mask",  mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            print(f"\nHSV range:")
            print(f"  lower = {list(lo)}")
            print(f"  upper = {list(hi)}")
            print(f"  tag_pixels = {int(np.sum(mask > 0))}")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    tune(args.image)
