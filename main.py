from dt_apriltags import Detector
import os
import cv2
import numpy as np
import time

# AprilTag Options
param = [329.8729619143081, 332.94611303946357, 528.0, 396.0]
at_detector = Detector(searchpath=['apriltags'],
                       families='tagStandard41h12',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

# Visualization Options
tag_w = 50  # pixel unit
crop_ratio = 8

if __name__ == '__main__':
    print("Testing SVM...q10")

    cap = cv2.VideoCapture(0)
    print("Press q to quit")
    time.sleep(2)

    while True:
        if cv2.waitKey(1) == ord('q'):
            break

        ret, image_rgb = cap.read()
        if ret:
            h, w, channel = image_rgb.shape
            image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

            tags = at_detector.detect(
                image_gray, estimate_tag_pose=True, camera_params=param, tag_size=0.065)

            if len(tags) == 0:
                SVM = np.zeros((h, w, 3), np.uint8)
                img_detected = image_rgb.copy()
            else:
                center = tags[0].center
                corners = tags[0].corners
                center_err = corners - center
                new_box = corners + center_err * (crop_ratio - 1)

                show_w, show_h = tag_w * crop_ratio, tag_w * crop_ratio

                srcPoint = new_box.astype(np.float32)
                dstPoint = np.array([[0, show_w], [show_h, show_w], [show_h, 0], [0, 0]], dtype=np.float32)

                matrix = cv2.getPerspectiveTransform(srcPoint, dstPoint)
                SVM = cv2.warpPerspective(image_rgb, matrix, (show_h, show_w))

                img_detected = image_rgb.copy()
                img_detected = cv2.polylines(img_detected, [corners.astype(np.int64)], True, (0, 0, 255), 10)

            combined_show = cv2.hconcat([cv2.resize(SVM, (w, h)), img_detected])
            cv2.imshow("svm", combined_show)

    cap.release()
    cv2.destroyAllWindows()


else:
    print("SVM module imported...q10")
