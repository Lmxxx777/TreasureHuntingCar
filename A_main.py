"""主要逻辑代码
需要与底层控制通讯，树莓派发送控制指令到动作执行机构
需要完善以下部分 ，参考注释编写代码
car_move
move_with_one_line
attactk
recive_msg
"""

import time
import cv2
from itertools import permutations
import heapq
import numpy as np
import A_detect_map
import A_detect_treasure
import A_plan_path
import math
import serial
import binascii
import yaml



uart2 = serial.Serial(port="/dev/ttyAMA0", baudrate=115200)
end_flag = b'\n'

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
aim = {"red_true": 0, "red_false": 0, "blue_true": 0, "blue_false": 0}
team = "blue"


# 最基本的A*搜索算法
def A_star(map, start, end):
    """
    使用A*算法寻找起点到终点的最短路径
    :param map: 二维列表，表示地图。0表示可以通过的点，1表示障碍物。
    :param start: 元组，表示起点坐标。
    :param end: 元组，表示终点坐标。
    :return: 列表，表示从起点到终点的最短路径，其中每个元素是一个坐标元组。
    """

    # 定义启发式函数（曼哈顿距离）
    def heuristic(node1, node2):
        return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])

    # 初始化open_list、closed_list、g_score、came_from
    open_list = [(0, start)]
    closed_list = set()
    g_score = {start: 0}
    came_from = {}

    # 开始搜索
    while open_list:
        # 取出f值最小的节点
        current = heapq.heappop(open_list)[1]
        if current == end:
            # 找到终点，返回路径
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        # 将当前节点加入closed_list
        closed_list.add(current)

        # 遍历相邻节点
        for neighbor in [(current[0] - 1, current[1]),
                         (current[0] + 1, current[1]),
                         (current[0], current[1] - 1),
                         (current[0], current[1] + 1)]:
            if 0 <= neighbor[0] < len(map) and 0 <= neighbor[1] < len(map[0]) and map[neighbor[0]][neighbor[1]] == 0:
                # 相邻节点是可通过的节点
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # 如果相邻节点不在g_score中，或者新的g值更优，则更新g_score和came_from
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, end)
                    heapq.heappush(open_list, (f_score, neighbor))
                    came_from[neighbor] = current

    # 没有找到可行路径，返回空列表
    return []


# 坐标变换函数，将10*10的坐标映射到地图矩阵上，方便用来可视化
def pose2map(x, y):
    return 21 - 2 * y, x * 2 - 1


def map2pose(x, y):
    return (21 - y) / 2, (x + 1) / 2


# 地图上两点之间的最短路径
def A_star_length(map1, start_x, start_y, end_x, end_y):
    # 定义起点和终点
    start = (start_x, start_y)
    end = (end_x, end_y)

    # 计算最短路径
    start = pose2map(*start)
    end = pose2map(*end)
    path = A_star(map1, start, end)
    path_length = int((len(path) - 1) / 2)
    return path_length


# 预计算
def precomputation(map1, start, end, mid_points):
    permutations_list = list(permutations(mid_points, 2))
    length_dict = {}
    length_dict[start] = {}
    for pt1 in mid_points:
        length_dict[pt1] = {}
        length_dict[start][pt1] = A_star_length(map1, start[0], start[1], pt1[0], pt1[1])
    for pt1 in mid_points:
        length_dict[pt1][end] = A_star_length(map1, pt1[0], pt1[1], end[0], end[1])
    for pt1, pt2 in permutations_list:
        length_dict[pt1][pt2] = A_star_length(map1, pt1[0], pt1[1], pt2[0], pt2[1])
    length_dict[start][end] = A_star_length(map1, start[0], start[1], end[0], end[1])
    return length_dict


# 计算最短距离的路线
def get_min_path(map1, start, end, mid_points, length_dict=None):
    # 穷举法8！=40320
    # 计算1000个路径需要3s，全部计算需要2分钟计算太慢,但是使用路径查询后大大减少了计算量40320组数据在0.2s完成计算获得最优路径
    permutations_list = list(permutations(mid_points))
    min_path_length = float("inf")
    min_path = None
    for mid_points in permutations_list:
        mid_points = list(mid_points)
        mid_points.append(end)
        mid_points.insert(0, start)

        all_length = 0
        for i in range(len(mid_points) - 1):
            if length_dict:  # 如果没有预计算则采用现场计算，很费时
                length = length_dict[mid_points[i]][mid_points[i + 1]]
            else:
                length = A_star_length(map1, mid_points[i][0], mid_points[i][1], mid_points[i + 1][0],
                                       mid_points[i + 1][1])
            all_length += length
        if all_length < min_path_length:
            min_path_length = all_length
            min_path = mid_points

    return min_path, min_path_length


# 将10*10pose坐标映射到21*21的地图坐标上
def gennerate_all_path(map1, min_path):
    path = []
    for i in range(len(min_path) - 1):
        # start=pose2map(*start)
        # end=pose2map(*end)
        base_path = A_star(map1, pose2map(*min_path[i]), pose2map(*min_path[i + 1]))
        path += base_path[1:]
    path.insert(0, pose2map(*min_path[0]))
    return path


def multi_goal_Astar(map1, start, end, mid_points):
    '''
    含有中间位置的最短路径规划算法
    '''
    yujisuan = precomputation(map1, start, end, mid_points)
    min_path, min_path_length = get_min_path(map1, start, end, mid_points, yujisuan)

    # print(real)

    return min_path, min_path_length


def multi_Astar(map1, start, end, mid_points):
    min_path, min_path_length = multi_goal_Astar(map1, start, end, mid_points)
    all_points = []
    for i in range(len(min_path) - 1):
        temp = A_star(map1, pose2map(*min_path[i]), pose2map(*min_path[i + 1]))[:-1]
        for j in temp:
            all_points.append(j)
    all_points.append(pose2map(*min_path[-1]))
    real = []
    for point in all_points:
        real.append(map2pose(point[0], point[1]))
    # print(all_points)
    real = real[::-1]
    return real


def find_turning_points(path):
    turning_points = []
    for i in range(1, len(path) - 1):
        current = path[i]
        previous = path[i - 1]
        next = path[i + 1]
        if ((current[0] - previous[0]) * (next[1] - current[1]) != (current[1] - previous[1]) * (
                next[0] - current[0])) or (current[0] - previous[0]) * (next[0] - current[0]) + (
                current[1] - previous[1]) * (next[1] - current[1]) < 0:
            # or (current[0]-previous[0]) * (next[1]-current[1])==(current[1]-previous[1]) * (next[0]-current[0]) ==0
            # #(3,1) (3,3) (3,1) (3-3)*(1-3) (3-1)*(1-3)
            turning_points.append(current)

    return turning_points


def turn_direction(v1, pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    # 计算下一个方向向量
    v2 = np.array([x2, y2]) - np.array([x1, y1])
    # 计算叉积
    cross = np.cross(v1, v2)
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    norm_product = ((v1[0] ** 2 + v1[1] ** 2) * (v2[0] ** 2 + v2[1] ** 2)) ** 0.5
    if cross > 0:
        return 'left'
    elif cross < 0:
        return 'right'
    elif dot_product == norm_product:
        return "straight"
    elif dot_product == (-1 * norm_product):
        return 'Reverse direction'


def car_move(action):
    """控制小车改变运行方向
    action:"right" "left" "straight" "Reverse direction"
    """
    pass
    # 在这里添加控制车辆转向的代码，使用串口通讯向stm32发送指令，必须执行完动作才能返回，阻塞！

    return


def recive_msg(action):
    """recive_msg 接收stm32发送过来识别到岔路口的信号
    如果识别到岔路口，返回True否则返回False
    在这里添加你的代码，接收消息
    """
    # pass

    data = uart2.readline()  # 循环读取一行数据
    qwe = data.decode().strip()
    print(action,"---",qwe,"+++")  # 打印接收到的数据（去除行尾的换行符）
    msg = "0"
    if "straight" == action:
        msg = "1"
    if "left" == action:
        msg = "2"
    if "right" == action:
        msg = "3"
    if "Reverse direction" == action:
        msg = "4"
    if "slow" == action:
        msg = "5"

    print("judge")
    print(type(qwe))
    print(type(msg))

    if qwe == msg:
        print("555")
        return 1
    if qwe == msg:
        print("111")
        return 0
    if qwe == msg:
        print("222")
        return 0
    if qwe == msg:
        print("333")
        return 0
    if qwe == msg:
        print("555")
        return 0
    if qwe == msg:
        print("999")
        return -1


def move_with_one_line(action):
    '''
    控制小车寻线行驶，
    #在这里添加控制小车寻线行驶的代码,使用串口通讯向stm32发送指令，直到识别出道路多叉口或者识别到宝藏，才能结束这个函数，这个函数必须阻塞！
    #这里你需要做两个事情，1小车寻线行驶，2如果遇到宝藏，识别宝藏的颜色,将maze_color变量赋值：int 0无宝藏，1可触碰宝藏，2不可触碰宝藏
    '''

    """
    添加控制小车寻线行驶的代码
    """

    for key in aim:
        aim[key] = 0

    find_cross = recive_msg(action)  ##判断是否进入岔路口

    if find_cross == 0:
        print("find_cross")
        return 0
    elif find_cross == -1:
        return -1
    elif find_cross == 1:
        ret, frame = cap.read()
        color = 0
        start_time = time.perf_counter()
        """在这里添加识别宝藏类型的代码如果需要碰撞的宝藏则返回1，不可触碰返回0，赋值变量color"""
        color = A_detect_treasure.find_aim(frame, team, aim, start_time)
        return color


def attactk():
    """attactk 撞击宝藏
    """
    pass
    # 在这里添加碰撞宝藏的代码，不能转动车的方向，建议使用舵机插上小棍，在车前面摆动，执行完动作之后返回，一定要阻塞
    # 控制舵机直行动作可以使用树莓派，也可以使用stm32,会产生pwm波就可以
    return

def gogogo(maze_location, map1, start, end, turn_points, corner_points, now_dir):

    print("准备动作执行列表")

    action_list = []
    all_action_list = []
    all_message_list = []
    for i in range(len(turn_points) - 1):
        action = turn_direction(now_dir, turn_points[i], turn_points[i + 1])
        now_dir = turn_points[i + 1][0] - turn_points[i][0], turn_points[i + 1][1] - turn_points[i][1]

        if turn_points[i + 1] in maze_location:
            all_action_list.append("slow")
            print(i, turn_points[i + 1], "slow")
            all_message_list.append([turn_points[i + 1], "slow", i])
        if turn_points[i] in corner_points:
            all_action_list.append(action)
            print(i, turn_points[i], action)
            all_message_list.append([turn_points[i], action, i])
        if turn_points[i] in maze_location and turn_points[i] not in corner_points:
            print(i, turn_points[i], action)
            if action == "Reverse direction":
                all_action_list.append(action)
                all_message_list.append([turn_points[i], action, i])

    print("全部路径规划")
    print(all_action_list)

    # 把规划成功的列表转成数组发给控制
    # TODO: 帧头协商
    run_list = []
    run_list.append(33)
    run_list.append(len(all_action_list))
    for i in all_action_list:
        if "straight" == i:
            run_list.append(1)
        if "left" == i:
            run_list.append(2)
        if "right" == i:
            run_list.append(3)
        if "Reverse direction" == i:
            run_list.append(4)
        if "slow" == i:
            run_list.append(5)
    print(run_list)

    byte_data = bytearray(run_list)
    uart2.write(byte_data)

    # step4开始寻线行驶
    print("开始寻线行驶")

    last_message = []
    for message in all_message_list:  # 从遇到第一个岔路口开始依次执行动作

        print("到达岔路口选择动作", message)
        # car_move(action)  # 多叉口路口选择方向 不需要了！！！！！！！！！！！！！！！！！！！！

        # 把减速和掉头都规划进去
        # 0 接收到它使用这个元素的信息
        # 1 无宝藏， 离开、保留规划
        # 2 假宝藏， 离开、删除对称点重新规划
        # 3 己方真宝藏， 撞击、删除对称的点重新规划、计数器++
        # 4 对方真宝藏， 离开、保留规划、
        # 发一个还是发多个：发一个就好
        # 发一个确认一个，如果遇到slow就开始识别，
        # color = move_with_one_line(message[1])

        # find_cross = recive_msg(action)  ##判断是否进入岔路口

        data = uart2.readline()  # 循环读取一行数据
        qwe = data.decode().strip()
        if qwe[-1] == "d":
            print("again!!!")
            maze_location = [(6, 3)]
            return True

        print(message[1],"---",qwe,"+++")  # 打印接收到的数据（去除行尾的换行符）

        find_cross = 99
        if qwe[-1] == "5":
            find_cross = 1
        if qwe[-1] == "1":
            find_cross = 0
        if qwe[-1] == "2":
            find_cross = 0
        if qwe[-1] == "3":
            find_cross = 0
        if qwe[-1] == "4":
            find_cross = 0

        color = 99
        if find_cross == 0:
            print("find_cross")
            color = 0
        elif find_cross == 1:
            start_time = time.perf_counter()
            for key in aim:
                aim[key] = 0
            while True:
                ret, frame = cap.read()
                if ret:
                    color = A_detect_treasure.find_aim(frame, team, aim, start_time)
                    if color != -1:
                        cv2.destroyAllWindows()
                        break



        if color == 0:
            last_message = message
            print("receive true message", message[1])
        elif color == 1 or color == 4:
            # TODO：协商帧头
            xiayige = [66, -1]
            xiayige_byte_data = bytearray(xiayige)
            uart2.write(xiayige_byte_data)
            last_message = message
        elif color == 2:

            other = (int(2 * 5.5 - message[0][0]), int(2 * 5.5 - message[0][1]))
            maze_location.remove(other)
            maze_location.remove(message[0])

            with open('remember.yml', 'r') as file2:
                remember2 = yaml.load(file2, Loader=yaml.FullLoader)
            if other in remember2['location']:
                remember2['location'].remove(other)
            if message[0] in remember2['location']:
                remember2['location'].remove(message[0])
            with open('remember.yml', 'w') as file2:
                yaml.dump(remember2, file2)

            # 利用上一个点作为起点，遇到拐角宝藏，只有这样才有可能规划成掉头，当然也有不是规划成掉头躲开
            # 删除当前和对称的两个假宝藏的点
            new_path = A_plan_path.path_plan(map1, last_message[0], end, maze_location)
            # 利用原来的路径规划，计算当前的朝向
            now_dir = turn_points[message[2]][0] - turn_points[message[2] - 1][0], turn_points[message[2]][1] - \
                      turn_points[message[2] - 1][1]

            # 清空旧列表，然后做新规划
            action_list.clear()
            all_action_list.clear()
            all_message_list.clear()
            for i in range(len(new_path) - 1):
                action = turn_direction(now_dir, new_path[i], new_path[i + 1])
                now_dir = new_path[i + 1][0] - new_path[i][0], new_path[i + 1][1] - new_path[i][1]

                if new_path[i + 1] in maze_location:
                    all_action_list.append("slow")
                    print(i, new_path[i + 1], "slow")
                    all_message_list.append([new_path[i + 1], "slow", i])
                if new_path[i] in corner_points:
                    all_action_list.append(action)
                    print(i, new_path[i], action)
                    all_message_list.append([new_path[i], action, i])
                if new_path[i] in maze_location and new_path[i] not in corner_points:
                    print(i, new_path[i], action)
                    if action == "Reverse direction":
                        all_action_list.append(action)
                        all_message_list.append([new_path[i], action, i])
                if i == 0:
                    all_action_list.append(action)
                    all_message_list.append([turn_points[i], action, i])
                    print(i, turn_points[i], action, "**")


            run_list = []
            run_list.append(33)
            run_list.append(len(all_action_list))
            for i in all_action_list:
                if "straight" == i:
                    run_list.append(1)
                if "left" == i:
                    run_list.append(2)
                if "right" == i:
                    run_list.append(3)
                if "Reverse direction" == i:
                    run_list.append(4)
                if "slow" == i:
                    run_list.append(5)
            print(run_list)

            # TODO：发送run_list，协商帧头规则
            jiade = [66, -1]
            jiade_byte_data = bytearray(jiade)
            uart2.write(jiade_byte_data)

            byte_data = bytearray(run_list)
            uart2.write(byte_data)

            last_message = message
        elif color == 3:

            with open('remember.yml', 'r') as file1:
                remember1 = yaml.load(file1, Loader=yaml.FullLoader)
            remember1['aim_count'] += 1
            with open('remember.yml', 'w') as file1:
                yaml.dump(remember1, file1)

            # 这里要注意下顺序，先发送撞击指令，然后重新路径规划发送，重新规划会规划掉头的（如果在死胡同里）
            # 先重新规划再撞击吧，可能规划太久危险了，可以分两次发送，规划的数组和撞击的指令
            # 撞击之后不一定掉头，可能是右转，像那个抽象点

            if remember1['aim_count'] == 3:
                maze_location.clear()

                with open('remember.yml', 'r') as file3:
                    remember3 = yaml.load(file3, Loader=yaml.FullLoader)
                remember3['location'].clear()
                with open('remember.yml', 'w') as file3:
                    yaml.dump(remember3, file3)

            other = (int(2 * 5.5 - message[0][0]), int(2 * 5.5 - message[0][1]))
            maze_location.remove(other)
            maze_location.remove(message[0])
            with open('remember.yml', 'r') as file4:
                remember4 = yaml.load(file4, Loader=yaml.FullLoader)
            if other in remember4['location']:
                remember4['location'].remove(other)
            if message[0] in remember4['location']:
                remember4['location'].remove(message[0])
            with open('remember.yml', 'w') as file4:
                yaml.dump(remember4, file4)

            # 利用上一个点作为起点，遇到拐角宝藏，只有这样才有可能规划成掉头，当然也有不是规划成掉头躲开
            # 删除当前和对称的两个假宝藏的点
            new_path = A_plan_path.path_plan(map1, last_message[0], end, maze_location)
            # 利用原来的路径规划，计算当前的朝向
            now_dir = turn_points[message[2]][0] - turn_points[message[2] - 1][0], turn_points[message[2]][1] - \
                      turn_points[message[2] - 1][1]

            # 清空旧列表，然后做新规划
            action_list.clear()
            all_action_list.clear()
            all_message_list.clear()
            for i in range(len(new_path) - 1):
                action = turn_direction(now_dir, new_path[i], new_path[i + 1])
                now_dir = new_path[i + 1][0] - new_path[i][0], new_path[i + 1][1] - new_path[i][1]

                if new_path[i + 1] in maze_location:
                    all_action_list.append("slow")
                    print(i, new_path[i + 1], "slow")
                    all_message_list.append([new_path[i + 1], "slow", i])
                if new_path[i] in corner_points:
                    all_action_list.append(action)
                    print(i, new_path[i], action)
                    all_message_list.append([new_path[i], action, i])
                if new_path[i] in maze_location and new_path[i] not in corner_points:
                    print(i, new_path[i], action)
                    if action == "Reverse direction":
                        all_action_list.append(action)
                        all_message_list.append([new_path[i], action, i])
                if i == 0:
                    all_action_list.append(action)
                    all_message_list.append([turn_points[i], action, i])
                    print(i, turn_points[i], action, "**")


            run_list = []
            run_list.append(33)
            run_list.append(len(all_action_list))
            for i in all_action_list:
                if "straight" == i:
                    run_list.append(1)
                if "left" == i:
                    run_list.append(2)
                if "right" == i:
                    run_list.append(3)
                if "Reverse direction" == i:
                    run_list.append(4)
                if "slow" == i:
                    run_list.append(5)
            print(run_list)

            # TODO：发送run_list，协商帧头规则
            zhende = [66, 1]
            zhende_byte_data = bytearray(zhende)
            uart2.write(zhende_byte_data)
            # 先撞击发送，再发送规划
            run_list_byte_data = bytearray(run_list)
            uart2.write(run_list_byte_data)

            last_message = message



if __name__ == "__main__":
    # 初始化串口，设置波特率为9600

    start = (1, 1)
    end = (10, 10)

    map1 = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1],
        [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
        [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]


    location = []
    map_time = time.perf_counter()
    # 输入合法并弥补不足
    draw_location = []
    tmp_location = []
    center_point = (5.5, 5.5)  # 对称中心点

    print("begin!!!")

    with open('remember.yml', 'r') as file:
        remember = yaml.load(file, Loader=yaml.FullLoader)

    if not remember['look_map']:
        while True:
            ret, frame = cap.read()
            if ret:
                bg = np.zeros((720,200,3),np.uint8)
                bg.fill(0)
                total_img = np.zeros((720,1480,3),np.uint8)

                data = uart2.readline()
                xinhao = data.decode().strip()
                print(xinhao)
                if xinhao[-1] == "a":
                # if 1:
                    # print(tmp_location)
                    map_end_time = time.perf_counter()
                    map_elapsed_time = map_end_time - map_time
                    print("程序运行时间：", map_elapsed_time, "秒")
                    if map_elapsed_time > 600:
                        remember['look_map'] = True
                        remember['location'] = list(draw_location)
                        cv2.destroyAllWindows()
                        break

                    location, img = A_detect_map.get_maze_map_pose(frame, 0)

                    legal_location = []
                    for i in location:
                        tmp = pose2map(i[0], i[1])
                        if map1[tmp[0]][tmp[1]] == 0:
                            legal_location.append(i)
                            symmetric_point = (2 * center_point[0] - i[0], 2 * center_point[1] - i[1])
                            if symmetric_point not in location:
                                legal_location.append(symmetric_point)
                    # print(legal_location)

                    for i in legal_location:
                        tmp_location.append(i)

                    draw_location = list(set(tmp_location))
                    for i in range(len(draw_location)):
                        text = f"({draw_location[i][0]}, {draw_location[i][1]})"
                        cv2.putText(bg,text,(0,30+i*30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
                    total_img[:, :1280, :] = img
                    total_img[:, 1280:, :] = bg
                elif xinhao[-1] == "b":
                    tmp_location.clear()
                elif xinhao[-1] == "c":
                    remember['look_map'] = True
                    remember['location'] = list(draw_location)
                    location = list(set(draw_location))
                    cv2.destroyAllWindows()
                    break
                cv2.imshow('result', total_img)
                cv2.waitKey(1)
            else:
                print("no picture")
    else:
        location = remember['location']

    with open('remember.yml', 'w') as file:
        yaml.dump(remember, file)

    # step1 识别宝藏
    # maze_location = [(8, 8), (5, 8), (9, 2), (10, 7), (1, 4), (2, 9)]
    maze_location = list(location)


    # step2路径规划

    print("开始路径规划")
    min_path = A_plan_path.path_plan(map1, start, end, maze_location)
    turn_points = min_path
    print("路径规划已完成")
    print(turn_points)


    # step3计算岔路口的动作
    now_dir = (1, 0)
    print("设定初始方向", now_dir)
    cross_points = [(3, 4),
                    (4, 3),
                    (5, 2),
                    (6, 2),
                    (7, 1),
                    (9, 1),
                    (8, 3),
                    (7, 4),
                    (7, 5),
                    (7, 6),
                    (8, 6),
                    (10, 6),
                    (8, 7),
                    (7, 8),
                    (6, 9),
                    (5, 9),
                    (3, 8),
                    (4, 7),
                    (4, 6),
                    (3, 5),
                    (1, 5),
                    (4, 5),
                    (4, 10),
                    (2, 10)]

    corner_points = [(2, 1), (3, 1), (5, 1), (6, 1), (7, 1), (9, 1), (10, 1),
                     (1, 2), (2, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2),
                     (1, 3), (2, 3), (4, 3), (6, 3), (7, 3), (8, 3), (9, 3),
                     (2, 4), (3, 4), (4, 4), (7, 4), (9, 4), (10, 4),
                     (1, 5), (3, 5), (4, 5), (7, 5), (8, 5),
                     (3, 6), (4, 6), (7, 6), (8, 6), (10, 6),
                     (1, 7), (2, 7), (4, 7), (7, 7), (8, 7), (9, 7),
                     (2, 8), (3, 8), (4, 8), (5, 8), (7, 8), (9, 8), (10, 8),
                     (3, 9), (4, 9), (5, 9), (6, 9), (7, 9), (9, 9), (10, 9),
                     (1, 10), (2, 10), (4, 10), (5, 10), (6, 10), (8, 10), (9, 10),
                     ]


    while 1:
        again = gogogo(maze_location,map1,start,end,turn_points,corner_points,now_dir)
        if again:
            print("*******")
            with open('remember.yml', 'r') as file9:
                remember9 = yaml.load(file9, Loader=yaml.FullLoader)
            maze_location = remember9['location']

            print("开始路径规划")
            min_path = A_plan_path.path_plan(map1, start, end, maze_location)
            turn_points = min_path
            print("路径规划已完成")
            print(turn_points)
            print("*******")
            continue

    # 到达终点
    print("到达终点结束运行")
