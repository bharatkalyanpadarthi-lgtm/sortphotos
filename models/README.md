# Video Face Detection Model

`face_detection_yunet_2023mar.onnx` is the OpenCV Zoo YuNet face detector.
It is used only as a fallback when InsightFace finds no usable face in a video
frame. Identity recognition still uses the existing local ArcFace model.
The configured one-recognized-frame video policy is separate from image consensus.

- Source: [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- License: MIT, [upstream notice](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE), reproduced in `LICENSE-YuNet.txt`.
- SHA-256: `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`

Only this small fallback detector is bundled. Primary and secondary InsightFace
weights remain external and are checked by `verify_runtime.py`.
