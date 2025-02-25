import cv2
import numpy as np
import time


def find_aim(img, team, aim, start_time):

    # 将RGB图像转换为HSV图像
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 设置红色的HSV阈值
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([160, 100, 100])
    upper_red = np.array([179, 255, 255])
    mask_red2 = cv2.inRange(hsv, lower_red, upper_red)

    # 将两个红色掩码合并为一个
    mask_red = mask_red1 + mask_red2

    # 设置蓝色的HSV阈值
    lower_blue = np.array([80, 80, 80])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # 设置黄色的HSV阈值
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 设置绿色的HSV阈值
    lower_green = np.array([50, 70, 70])
    upper_green = np.array([70, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # 对红色掩码进行形态学操作
    kernel = np.ones((5,5), np.uint8)
    # mask_red = cv2.erode(mask_red, kernel)
    mask_red = cv2.dilate(mask_red, kernel)

    # 对蓝色掩码进行形态学操作
    # mask_blue = cv2.erode(mask_blue, kernel)
    mask_blue = cv2.dilate(mask_blue, kernel)

    # 对黄色掩码进行形态学操作
    # mask_yellow = cv2.erode(mask_yellow, kernel)
    mask_yellow = cv2.dilate(mask_yellow, kernel)

    # 对绿色掩码进行形态学操作
    # mask_green = cv2.erode(mask_green, kernel)
    mask_green = cv2.dilate(mask_green, kernel)

    # cv2.imshow('Red Objects', mask_red)
    # cv2.imshow('Blue Objects', mask_blue)
    # cv2.imshow('Yellow Objects', mask_yellow)
    # cv2.imshow('Green Objects', mask_green)

    red = []
    red_contours, red_hierarchy = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in range(len(red_contours)):
        approx = cv2.approxPolyDP(red_contours[cnt], cv2.arcLength(red_contours[cnt], True) * 0.03, True)
        area = cv2.contourArea(red_contours[cnt])
        x, y, w, h = cv2.boundingRect(red_contours[cnt])
        wh_ratio = w / h
        area_ratio = area / (w * h)
        is_rectangle = True if len(approx) == 4 else False
        is_area = True if 100 < area < 30000 else False
        is_square = True if 0.4 < wh_ratio < 0.7 else False
        is_shape = True if 0.7 < area_ratio < 1 else False
        if is_rectangle and is_area and is_square and is_shape:
            # print(len(approx), area, wh_ratio, area_ratio)
            center = (x + w / 2, y + h / 2)
            red.append(center)
            cv2.drawContours(img, red_contours, cnt, (255, 0, 128), 2)

    blue = []
    blue_contours, blue_hierarchy = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in range(len(blue_contours)):
        approx = cv2.approxPolyDP(blue_contours[cnt], cv2.arcLength(blue_contours[cnt], True) * 0.03, True)
        area = cv2.contourArea(blue_contours[cnt])
        x, y, w, h = cv2.boundingRect(blue_contours[cnt])
        wh_ratio = w / h
        area_ratio = area / (w * h)
        is_rectangle = True if len(approx) == 4 else False
        is_area = True if 100 < area < 30000 else False
        is_square = True if 0.4 < wh_ratio < 0.7 else False
        is_shape = True if 0.7 < area_ratio < 1 else False
        if is_rectangle and is_area and is_square and is_shape:
            # print(len(approx), area, wh_ratio, area_ratio)
            center = (x + w / 2, y + h / 2)
            blue.append(center)
            cv2.drawContours(img, blue_contours, cnt, (255, 0, 128), 2)

    yellow = []
    yellow_contours, yellow_hierarchy = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in range(len(yellow_contours)):
        approx = cv2.approxPolyDP(yellow_contours[cnt], cv2.arcLength(yellow_contours[cnt], True) * 0.02, True)
        area = cv2.contourArea(yellow_contours[cnt])
        x, y, w, h = cv2.boundingRect(yellow_contours[cnt])
        wh_ratio = w / h
        area_ratio = area / (w * h)
        is_circle = True if len(approx) > 4 else False
        is_area = True if 100 < area < 30000 else False
        is_square = True if 0.8 < wh_ratio < 1.2 else False
        is_shape = True if 0.6 < area_ratio < 1 else False

        if 1 and is_area and is_square and is_shape:
            # print(len(approx), area, wh_ratio, area_ratio)
            center = (x + w / 2, y + h / 2)
            yellow.append(center)
            cv2.drawContours(img, yellow_contours, cnt, (255, 0, 128), 2)

    green = []
    green_contours, green_hierarchy = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in range(len(green_contours)):
        approx = cv2.approxPolyDP(green_contours[cnt], cv2.arcLength(green_contours[cnt], True) * 0.05, True)
        area = cv2.contourArea(green_contours[cnt])
        x, y, w, h = cv2.boundingRect(green_contours[cnt])
        wh_ratio = w / h
        area_ratio = area / (w * h)
        is_triangle = True if len(approx) == 3 else False
        is_area = True if 100 < area < 30000 else False
        is_square = True if 0.8 < wh_ratio < 1.2 else False
        is_shape = True if 0.3 < area_ratio < 0.7 else False

        if is_triangle and is_area and is_square and is_shape:
            center = (x + w / 2, y + h / 2)
            green.append(center)
            cv2.drawContours(img, green_contours, cnt, (255, 0, 128), 2)

    for r in red:
        for g in green:
            distance = ((r[0] - g[0]) ** 2 + (r[1] - g[1]) ** 2) ** 0.5
            if 0 < distance < 100:
                x = int((r[0] + g[0]) / 2)
                y = int((r[1] + g[1]) / 2)
                center = (x, y)
                cv2.circle(img, center, 20, (255, 0, 128), -1)
                aim["red_true"] += 1
    for r in red:
        for y in yellow:
            distance = ((r[0] - y[0]) ** 2 + (r[1] - y[1]) ** 2) ** 0.5
            if 0 < distance < 100:
                x = int((r[0] + y[0]) / 2)
                y = int((r[1] + y[1]) / 2)
                center = (x, y)
                cv2.circle(img, center, 20, (255, 0, 128), -1)
                aim["red_false"] += 1
    for b in blue:
        for y in yellow:
            distance = ((b[0] - y[0]) ** 2 + (b[1] - y[1]) ** 2) ** 0.5
            if 0 < distance < 100:
                x = int((b[0] + y[0]) / 2)
                y = int((b[1] + y[1]) / 2)
                center = (x, y)
                cv2.circle(img, center, 20, (255, 0, 128), -1)
                aim["blue_true"] += 1
    for b in blue:
        for g in green:
            distance = ((b[0] - g[0]) ** 2 + (b[1] - g[1]) ** 2) ** 0.5
            if 0 < distance < 100:
                x = int((b[0] + g[0]) / 2)
                y = int((b[1] + g[1]) / 2)
                center = (x, y)
                cv2.circle(img, center, 20, (255, 0, 128), -1)
                aim["blue_false"] += 1

    print(aim)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print("程序运行时间：", elapsed_time, "秒")
    if elapsed_time > 60:
        count = 0
        for value in aim.values():
            if value == 0:
                count += 1
        if count == 4:
            print("count == 4")
            return 1
        else:
            max_key = max(aim.items(), key=lambda x: x[1])[0]
            if aim[max_key] < 5:
                print("aim[max_key] < 5")
                return 1
            else:
                if max_key == 'red_false' and max_key == 'blue_false':
                    return 2
                if max_key == 'red_true' and team == "red":
                    return 3
                if max_key == 'blue_true' and team == "blue":
                    return 3
                if max_key == 'blue_true' and team == "red":
                    return 4
                if max_key == 'red_true' and team == "blue":
                    return 4
    else:
        max_key = max(aim.items(), key=lambda x: x[1])[0]
        if aim[max_key] >= 1:
            if max_key == 'red_false' and max_key == 'blue_false':
                return 2
            if max_key == 'red_true' and team == "red":
                return 3
            if max_key == 'blue_true' and team == "blue":
                return 3
            if max_key == 'blue_true' and team == "red":
                return 4
            if max_key == 'red_true' and team == "blue":
                return 4
        else:
            return -1

    print(aim)
    y = 30
    for key, value in aim.items():
        cv2.putText(img, key + ': ' + str(value), (0, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        y += 30
    cv2.imshow('img', img)


if __name__ == "__main__":

    # cap = cv2.VideoCapture(1)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    aim = {"red_true": 0, "red_false": 0, "blue_true": 0, "blue_false": 0}
    team = "blue"
    while True:
        start_time = time.perf_counter()

        # ret, frame = cap.read()
        # find_aim(frame, team, aim, start_time)

        src = cv2.imread("blue_false.jpg")
        find_aim(src, team, aim, start_time)

        # cv2.imshow("input", src)
        cv2.waitKey(1)
