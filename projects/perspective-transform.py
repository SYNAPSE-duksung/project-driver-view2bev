import cv2
import numpy as np

video = cv2.VideoCapture("../data/forest-driving.mp4")

#첫 프레임 가져오기 -> 좌표 설정
ret, frame = video.read()
if not ret:
    print("비디오를 읽을 수 없습니다.")
    video.release()
    exit()

points = []

# 마우스 클릭
def mouse_callback(event, x, y, flags, param):
    global frame

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(points) < 4:
            points.append([x, y])

            print(f"{len(points)}번째 점: ({x}, {y})")

            # 점 표시
            cv2.circle(frame, (x, y), 7, (0, 0, 255), -1)

            # 선 연결
            if len(points) > 1:
                cv2.line(
                    frame,
                    tuple(points[-2]),
                    tuple(points[-1]),
                    (0, 255, 0),
                    2
                )
                if len(points) == 4:
                    pts = np.array(points, np.int32)
                    cv2.polylines(
                        frame,
                        [pts],
                        True,
                        (0, 255, 0),
                        3
                    )

            cv2.imshow("Select ROI", frame)

# ===== 창 =====
cv2.imshow("Select ROI", frame)
cv2.setMouseCallback("Select ROI", mouse_callback)

print("도로 영역 4점 클릭")
print("순서:")
print("1. 왼아래")
print("2. 오른아래")
print("3. 오른위")
print("4. 왼위")

# ===== 4점 선택 대기 =====
while True:

    cv2.imshow("Select ROI", frame)

    key = cv2.waitKey(1)

    # ESC 종료
    if key == 27:
        break

    # 4점 선택 완료
    if len(points) == 4:
        break

cv2.destroyAllWindows()

# ===== 좌표 출력 =====
src = np.float32(points)

print("\n선택된 좌표:")
print(src)

# ===== BEV =====
h, w = frame.shape[:2]

dst_w = 500
dst_h = 700

dst = np.float32([
    [0, dst_h],
    [dst_w, dst_h],
    [dst_w, 0],
    [0, 0]
])

M = cv2.getPerspectiveTransform(src, dst)

# 영상 처음부터
video.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:

    ret, frame = video.read()

    if not ret:
        break

    bev = cv2.warpPerspective(frame, M, (dst_w, dst_h))

    cv2.imshow("Original", frame)
    cv2.imshow("BEV", bev)

    if cv2.waitKey(30) == 27:
        break

video.release()
cv2.destroyAllWindows()