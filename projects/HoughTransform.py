import cv2
import numpy as np

video = cv2.VideoCapture("../data/empty-road.mp4")

while True:
    ret, frame = video.read()
    if not ret:
        print("비디오를 읽을 수 없습니다.")
        break
    h, w = frame.shape[:2]

    # ===== ROI 설정 =====
    mask = np.zeros((h, w), dtype=np.uint8)
    roi = np.array([[
        (0, h),
        (w, h),
        (int(w * 0.6), int(h * 0.8)),
        (int(w * 0.4), int(h * 0.8))
    ]], dtype=np.int32)

    cv2.fillPoly(mask, roi, 255)

    # Edge 검출 (전처리)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)    # 노이즈 제거
    edges = cv2.Canny(blur, 50, 120)
    # ROI 적용
    edges = cv2.bitwise_and(edges, mask)

    # Hough Transform을 사용하여 선 검출
    lines = cv2.HoughLinesP(
        edges,  # 입력 Edge 이미지
        1,      # 거리 해상도(1픽셀 단위)
        np.pi / 180,    # 각도 해상도(1도 단위)
        threshold=7,   # 클수록 강한 직선만 검출
        minLineLength=100, #최소 선 길이
        maxLineGap=5   # 선 사이 최대 간격
    )

    # 시각화용 이미지
    debug = frame.copy()
    # ROI 영역 표시
    cv2.polylines(
        debug,
        roi,
        True,
        (0, 255, 255),
        2
    )

    # 좌우 차선 분리
    left_lines = []
    right_lines = []

    # 직선 검출 결과 처리
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(
                debug,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )
            
            # 수직선 방지
            if x2 - x1 == 0:
                continue

            # 기울기 계산
            slope = (y2 - y1) / (x2 - x1) 
            if abs(slope) < 0.3 or abs(slope) > 5:   # 너무 완만하거나 너무 가파른 선은 무시
                continue
            if slope < 0:   # 음수 기울기는 왼쪽 차선
                left_lines.append(line)
            else:   # 양수 기울기는 오른쪽 차선
                right_lines.append(line)

        # 대표 차선 계산 (좌우 차선 평균 내기 -> 하나의 선으로 표현)
        def average_line(lines):
            x_points = []
            y_points = []

            for line in lines:
                x1, y1, x2, y2 = line[0]
                x_points.extend([x1, x2])
                y_points.extend([y1, y2])
            # 선이 없으면 None 반환
            if len(x_points) == 0:
                return None
        
            # 1차 다항식으로 선형 회귀 (y = mx + b 형태로 직선 찾기)
            poly = np.polyfit(x_points, y_points, 1)
            slope = poly[0]
            intercept = poly[1]

            return slope, intercept

        # 선 좌표 생성 (특정 y 위치에서 x 좌표 계산)
        def make_line_points(y1, y2, slope, intercept):
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
        
            return (x1, int(y1)), (x2, int(y2))

        # 평균 차선 생성
        left_avg = average_line(left_lines) 
        right_avg = average_line(right_lines)

        y_bottom = h
        y_top = int(h * 0.6)   # 화면 위쪽 60% 지점

        def draw_representative_line(avg_line):
            if avg_line is None:
                return 

            line = make_line_points(
                y_bottom,
                y_top,
                avg_line[0],
                avg_line[1]
            )

            cv2.line(
                debug,
                line[0],
                line[1],
                (0, 255, 0),
                5
            )
        draw_representative_line(left_avg)
        draw_representative_line(right_avg)

        # 결과 출력
        print(len(left_lines), len(right_lines))
        results = np.hstack((frame, debug))
        results = cv2.resize(results, (0, 0), fx=0.5, fy=0.5)

        cv2.imshow("Lane Detection", results)
        if cv2.waitKey(30) == 27:
            break

video.release()
cv2.destroyAllWindows()