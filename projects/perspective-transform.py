import cv2
import numpy as np

video = cv2.VideoCapture("../data/empty-road.mp4")
# points = []

# def mouse_callback(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(f"Point: ({x}, {y})")
#         points.append([x, y])

#         # 클릭한 위치에 점 찍기
#         cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
#         cv2.imshow("image", img)

# # 이미지 불러오기 (영상이면 frame 하나 저장해서 사용)
# ret, frame = video.read()

# if ret:
#     cv2.imwrite("frame.jpg", frame)
#     print("frame.jpg 저장 완료")
# else:
#     print("프레임 못 읽음")

# video.release()

# img = cv2.imread("frame.jpg")
# cv2.imshow("image", img)
# cv2.setMouseCallback("image", mouse_callback)

# print("👉 차선 기준으로 4점 클릭하세요 (왼아 → 오아 → 오위 → 왼위)")

# while True:
#     cv2.imshow("image", img)
#     if cv2.waitKey(1) & 0xFF == 27:  # ESC 누르면 종료
#         break

# cv2.destroyAllWindows()

# print("\n선택된 좌표:")
# print(np.float32(points))     

#첫 프레임 가져오기 -> 좌표 설정
ret, frame = video.read()
if not ret:
    print("비디오를 읽을 수 없습니다.")
    video.release()
    exit()

h, w = frame.shape[:2]

src = np.float32([
    [35, 715],
    [902, 715],
    [677,580],
    [655,579]
])
dst = np.float32([
    [0, h],
    [w, h],
    [w,0],
    [0,0]
])

M = cv2.getPerspectiveTransform(src, dst)

# 다시 영상 처음부터
video.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    bev = cv2.warpPerspective(frame, M, (w, h))

    cv2.imshow("BEV", bev)

    if cv2.waitKey(1) == 27:  # ESC
        break

video.release()
cv2.destroyAllWindows()
