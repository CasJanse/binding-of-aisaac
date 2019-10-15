from ScreenshotMachine import ScreenshotMachine
import cv2
import time
import win32gui
import re
import random
import numpy as np
from NeuralNetwork import NeuralNetwork
from InputHandler import InputHandler

show_windows = True

# Key codes
W = 0x11
A = 0x1E
S = 0x1F
D = 0x20
Up = 0xC8
Left = 0xCB
Down = 0xD0
Right = 0xCD
key_codes = [W, A, S, D, Up, Left, Down, Right]

# Neural network
input_nodes_amount = 612
hidden_nodes_amount = 100
output_nodes_amount = 4
learning_rate = 0.15
hidden_layers = 1

# Input Handler
input_handler = InputHandler()

# Door data
top_open = np.array([88, 80, 75, 78, 75, 77, 80, 80, 81, 75, 75, 75, 77, 74, 74, 75, 80, 89,
 65, 58, 56, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 46, 58, 58, 70,
 77, 76, 59, 47, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 49, 56, 66, 84,
 64, 72, 62, 52, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 51, 74, 66, 60,
 80, 82, 78, 54, 48, 48, 48, 48, 48, 48, 48, 48, 48, 47, 57, 71, 79, 78,
 78, 78, 81, 59, 47, 48, 47, 47, 48, 47, 47, 47, 48, 49, 70, 75, 78, 78,
 79, 78, 75, 66, 51, 52, 51, 57, 52, 51, 56, 51, 51, 55, 74, 74, 77, 77,
 71, 71, 71, 75, 57, 58, 58, 58, 59, 58, 58, 58, 58, 64, 64, 68, 68, 67,
 59, 59, 58, 57, 61, 61, 62, 62, 62, 61, 61, 61, 61, 62, 59, 59, 59, 59])

top_closed = np.array([ 89,  89,  88,  89,  89, 112,  65,  74,  84,  89,  89,  89,  89,  89,
  89,  89,  89,  89,  90,  82,  88,  89,  89,  72,  69,  71,  76,  89,
  89,  85,  86,  89,  90,  90,  90,  88,  75,  78,  78,  78,  75,  73,
  67,  67,  73,  77,  78,  78,  78,  72,  72,  72,  78,  83,  62,  63,
  63,  63,  63,  64,  66,  67,  64,  63,  63,  63,  63,  62,  63,  66,
  66,  59,  77,  79,  77,  78,  80,  80,  81,  79,  73,  81,  81,  81,
  79,  76,  77,  77,  76,  75,  76,  76,  76,  76,  76,  76,  76,  74,
  73,  76,  76,  80,  76,  73,  76,  76,  75,  75,  76,  76,  77,  81,
  70,  75,  92,  83,  73,  77,  76,  68,  70,  73,  70,  72,  73,  73,
  69,  69,  69,  69,  64,  75, 196, 195,  79,  83,  72,  64,  64,  66,
  67,  66,  66,  65,  59,  59,  58,  58,  58,  54, 200,  53,  56,  70,
  59,  59,  59,  58,  58,  58,  58,  58])

left_open = np.array([76, 58, 83, 64, 78, 79, 75, 72, 63, 76, 55, 57, 61, 79, 80, 80, 67, 62,
 76, 47, 47, 53, 51, 63, 73, 75, 61, 76, 47, 48, 48, 48, 47, 51, 58, 65,
 75, 47, 48, 48, 48, 48, 51, 58, 65, 75, 47, 48, 48, 48, 48, 55, 58, 65,
 76, 47, 48, 48, 48, 48, 55, 58, 56, 76, 47, 48, 48, 48, 48, 51, 55, 65,
 81, 47, 48, 48, 48, 48, 53, 58, 64, 79, 47, 48, 48, 48, 48, 55, 58, 64,
 80, 47, 48, 48, 48, 48, 55, 58, 66, 62, 47, 48, 48, 48, 48, 51, 58, 66,
 81, 47, 48, 47, 50, 52, 59, 62, 66, 75, 49, 52, 56, 58, 67, 73, 66, 62,
 76, 58, 55, 79, 83, 80, 84, 66, 63, 80, 58, 61, 67, 76, 79, 79, 67, 66])

left_closed = np.array([99, 93, 79, 67, 78, 79, 75, 72, 63, 93, 93, 74, 74, 77, 79, 79, 67, 62,
 93, 88, 89, 70, 79, 79, 76, 66, 62, 93, 93, 89, 70, 81, 79, 68, 66, 65,
 93, 93, 89, 70, 80, 79, 76, 65, 62, 93, 92, 90, 70, 79, 79, 76, 60, 62,
 93, 89, 94, 72, 69, 68, 61, 60, 63, 93, 86, 89, 71, 60, 60, 66, 59, 62,
 93, 84, 90, 70, 65, 62, 65, 62, 62, 93, 90, 94, 71, 65, 66, 60, 60, 63,
 93, 91, 90, 70, 79, 79, 73, 60, 66, 93, 93, 89, 70, 80, 79, 77, 65, 65,
 93, 93, 89, 70, 81, 79, 68, 66, 62, 93, 89, 89, 70, 80, 79, 75, 66, 62,
 96, 89, 86, 72, 75, 79, 83, 66, 63, 93, 93, 79, 72, 76, 79, 79, 67, 66])

right_open = np.array([73, 62, 75, 77, 74, 85, 77, 58, 73, 73, 61, 72, 79, 68, 58, 57, 52, 73,
 62, 63, 57, 57, 53, 51, 48, 48, 73, 63, 58, 54, 48, 48, 48, 48, 48, 64,
 61, 58, 59, 51, 48, 48, 48, 48, 80, 61, 57, 59, 50, 48, 48, 48, 48, 79,
 64, 57, 59, 50, 48, 48, 48, 48, 79, 61, 53, 51, 48, 48, 48, 48, 48, 73,
 62, 57, 59, 50, 48, 48, 48, 48, 73, 60, 57, 59, 50, 48, 48, 48, 48, 73,
 60, 57, 59, 51, 48, 48, 48, 48, 75, 61, 57, 53, 48, 48, 48, 48, 48, 65,
 71, 66, 67, 56, 53, 51, 48, 48, 74, 73, 62, 70, 82, 80, 57, 56, 53, 73,
 71, 63, 77, 77, 75, 76, 81, 57, 73, 66, 64, 77, 76, 77, 58, 75, 72, 79])

right_closed = np.array([ 71,  62,  78,  70,  81,  59,  75,  95,  86,  72,  62,  71,  71,  80,
  60,  76,  91,  93,  73,  63,  64,  78,  79,  60,  77,  87,  93,  74,
  61,  66,  79,  80,  59,  78,  93,  93,  74,  60,  66,  79,  82,  59,
  79,  90,  93,  70,  60,  66,  80,  82,  59,  80,  98,  93,  68,  60,
  60,  68,  85,  59,  81,  99,  93,  66,  60,  67,  62,  60,  60,  75,
  84,  93,  64,  60,  60,  66,  66,  61,  67,  74,  76,  64,  60,  65,
  63,  64,  59,  81, 101,  74,  66,  60,  65,  78,  82,  59,  81, 102,
  93,  69,  60,  66,  80,  82,  59,  79,  90,  93,  69,  62,  66,  79,
  81,  59,  78,  90,  93,  71,  63,  77,  79,  81,  59,  78,  88,  93,
  70,  64,  79,  79,  79,  59,  77,  85,  93,  65,  64,  80,  78,  79,
  59,  77,  94,  93])

bottom_open = np.array([64, 64, 64, 62, 75, 74, 74, 68, 67, 74, 74, 74, 74, 73, 59, 64, 64, 64,
 65, 65, 66, 63, 64, 58, 58, 58, 58, 58, 58, 58, 58, 57, 74, 65, 65, 65,
 75, 75, 75, 72, 55, 51, 51, 51, 51, 52, 51, 51, 52, 51, 66, 75, 75, 75,
 76, 75, 70, 75, 49, 48, 48, 48, 48, 48, 48, 48, 48, 47, 59, 82, 75, 75,
 75, 75, 71, 56, 47, 48, 48, 48, 48, 48, 48, 48, 48, 48, 54, 77, 76, 75,
 58, 56, 82, 54, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 53, 63, 72, 58,
 72, 67, 55, 49, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 47, 59, 82, 74,
 66, 57, 58, 49, 51, 51, 50, 50, 50, 50, 50, 50, 51, 51, 50, 55, 57, 55,
 87, 82, 84, 75, 80, 77, 87, 81, 82, 81, 81, 85, 81, 76, 79, 82, 84, 87])

bottom_closed = np.array([64, 64, 64, 64, 64, 64, 64, 64, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64,
 67, 67, 67, 67, 66, 68, 68, 68, 67, 67, 68, 66, 66, 67, 66, 66, 66, 67,
 78, 78, 78, 79, 75, 69, 79, 80, 78, 78, 77, 69, 68, 69, 77, 77, 78, 72,
 79, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 85, 78, 78, 78, 78, 78, 78,
 78, 78, 78, 78, 78, 78, 79, 79, 78, 78, 78, 78, 79, 79, 78, 78, 78, 78,
 59, 59, 58, 58, 57, 58, 59, 60, 60, 59, 58, 59, 59, 58, 59, 60, 60, 59,
 75, 76, 76, 76, 75, 75, 66, 65, 75, 75, 76, 77, 76, 75, 75, 75, 76, 76,
 85, 82, 94, 93, 88, 74, 66, 70, 75, 86, 94, 93, 80, 89, 87, 84, 82, 93,
 93, 93, 93, 93, 93, 84, 67, 71, 82, 93, 93, 93, 93, 93, 93, 93, 93, 93])

top_open = top_open / 255 + 0.01
left_open = left_open / 255 + 0.01
right_open = right_open / 255 + 0.01
bottom_open = bottom_open / 255 + 0.01
top_closed = top_closed / 255 + 0.01
left_closed = left_closed / 255 + 0.01
right_closed = right_closed / 255 + 0.01
bottom_closed = bottom_closed / 255 + 0.01

open_doors = [top_open, left_open, right_open, bottom_open]
closed_doors = [top_closed, left_closed, right_closed, bottom_closed]
all_doors = [closed_doors, open_doors]


def main():
    time.sleep(1)

    # Screen positioning
    window_rect = get_window_position()

    # Screenshot
    screenshot_machine = create_screenshot_machine(window_rect)

    # Neural network
    # neural_network = NeuralNetwork(screenshot_machine.image_size, hidden_nodes_amount, output_nodes_amount, learning_rate,
    #                                hidden_layers)
    # neural_network = NeuralNetwork(input_nodes_amount, hidden_nodes_amount, output_nodes_amount,
    #                                learning_rate,
    #                                hidden_layers)

    neural_network_door_recognition = NeuralNetwork("door_recognition", 612, 100, 4, 0.15, 1)

    neural_network_door_movement = NeuralNetwork("door_movement", 6, 10, 4, 0.2, 1)

    # train_door_recognition_network(neural_network_door_recognition)

    if show_windows:
        cv2.namedWindow("Tracking")
        cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
        cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
        cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
        cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
        cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
        cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

    start_door_network(screenshot_machine, neural_network_door_recognition, neural_network_door_movement)
    pass


def get_window_position():
    window = win32gui.GetForegroundWindow()
    window_rect = win32gui.GetWindowRect(window)
    win32gui.SetForegroundWindow(window)
    return window_rect


def create_screenshot_machine(window_rect):
    x = window_rect[0]
    y = window_rect[1]
    width = window_rect[2]
    height = window_rect[3]
    screenshot_machine = ScreenshotMachine(x, int(y + ((height - y) * 0.226)), width - x, int((height - y) * 0.774))
    # screenshot_machine.set_coordinates(68, 153, 662, 410)
    return screenshot_machine


def train_door_recognition_network(network):
    # Set training data
    door_data = []
    targets = []
    print('tets')
    network.load_weights()
    # for i in range(1000):
    #     current_doors = np.array([])
    #     choices = np.random.randint(2, size=4)
    #     current_doors = np.append(current_doors, all_doors[choices[0]][0])
    #     current_doors = np.append(current_doors, all_doors[choices[1]][1])
    #     current_doors = np.append(current_doors, all_doors[choices[2]][2])
    #     current_doors = np.append(current_doors, all_doors[choices[3]][3])
    #     door_data.append(current_doors)
    #     targets.append(choices)
    #
    # # Train the network
    # for i in range(len(door_data)):
    #     network.train(door_data[i], targets[i])

    test_data = np.array([])
    test_data = np.append(test_data, top_open)
    test_data = np.append(test_data, left_open)
    test_data = np.append(test_data, right_closed)
    test_data = np.append(test_data, bottom_closed)

    test = network.query(test_data)
    print(test)
    save = input("save weights?")
    if save.lower()[0] == "y":
        network.save_weights()


def start_door_network(screenshot_machine, neural_network_door_recognition, neural_network_door_movement):
    isaac_x = 0
    isaac_y = 0
    neural_network_door_recognition.load_weights()

    while True:
        # Take screenshot
        time.sleep(1 / 20)
        sct_img = screenshot_machine.take_screenshot()

        # Set the position of isaac gained from the screenshot
        isaac_coords = find_isaac(sct_img)
        if isaac_coords is not None:
            isaac_x = isaac_coords[0]
            isaac_y = isaac_coords[1]

        if show_windows:
            cv2.imshow("OpenCV/Numpy grayscale", cv2.cvtColor(sct_img, cv2.COLOR_BGRA2GRAY))

        sct_img = cv2.cvtColor(sct_img, cv2.COLOR_BGRA2GRAY)
        sct_img = sct_img.flatten()

        # Get door pixels from screenshot
        top_door = get_img_values_in_square(sct_img, screenshot_machine.image_width, {"x": 71, "y": 4},
                                            {"width": 17, "height": 9})
        left_door = get_img_values_in_square(sct_img, screenshot_machine.image_width, {"x": 7, "y": 40},
                                             {"width": 8, "height": 16})
        right_door = get_img_values_in_square(sct_img, screenshot_machine.image_width, {"x": 144, "y": 40},
                                              {"width": 8, "height": 16})
        down_door = get_img_values_in_square(sct_img, screenshot_machine.image_width, {"x": 71, "y": 82},
                                             {"width": 17, "height": 9})

        cv2.imshow("bottomdoor", down_door)

        # Prepare data into flat array
        doors = np.array([left_door, right_door, down_door])
        inputs = np.array(top_door)
        for door in doors:
            for value in door:
                inputs = np.append(inputs, value)
        inputs = ((inputs / 255) * 0.99 + 0.01)
        output_door_recognition = neural_network_door_recognition.query(inputs)

        output_door_recognition = np.append(output_door_recognition, isaac_x)
        output_door_recognition = np.append(output_door_recognition, isaac_y)

        print(output_door_recognition)

        output_door_movement = neural_network_door_movement.query(output_door_recognition)

        # press_keys(output_door_movement)

        # # Prepare inputs by normalizing them
        # inputs = sct_img
        # inputs = ((inputs / 255) * 0.99) + 0.01
        # output = neural_network.query(inputs)
        #
        # for key_code in key_index_array:
        #     input_handler.release_key(key_codes[key_code])
        #
        # key_index_array = []
        #
        # for i in range(len(output)):
        #     if output[i] >= 0.5:
        #         key_index_array.append(i)
        #         input_handler.press_key(key_codes[i])

        # Stop loop when done
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break


def find_isaac(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # # Set upper and lower bound of Hue, Saturation and Value using the tracking window
    # l_h = cv2.getTrackbarPos("LH", "Tracking")
    # l_s = cv2.getTrackbarPos("LS", "Tracking")
    # l_v = cv2.getTrackbarPos("LV", "Tracking")
    # u_h = cv2.getTrackbarPos("UH", "Tracking")
    # u_s = cv2.getTrackbarPos("US", "Tracking")
    # u_v = cv2.getTrackbarPos("UV", "Tracking")
    # l_c = np.array([l_h, l_s, l_v])
    # u_c = np.array([u_h, u_s, u_v])

    l_c = np.array([0, 63, 167])
    u_c = np.array([0, 65, 208])

    # Create a mask using the upper and lower color values
    mask = cv2.inRange(hsv, l_c, u_c)

    # Put the mask over the original image to show only isaac
    res_img = cv2.bitwise_and(img, img, mask=mask)

    # Convert image to so everything expect isaac becomes white
    res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2GRAY)
    # res_img = 255-res_img

    # Find isaac in image by checking non-zero pixels
    isaac_img = cv2.findNonZero(res_img)
    if isaac_img is not None:
        # Find top, bottom, left and right edges of isaac
        left = isaac_img[:, 0, 0].min()
        right = isaac_img[:, 0, 0].max()
        top = isaac_img[:, 0, 1].min()
        bottom = isaac_img[:, 0, 1].max()

        # Find center of isaac by averaging edges
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2

        # print("Isaac center: {}, {}".format(center_x, center_y))

        if show_windows:
            cv2.imshow("OpenCV/Numpy HSV", hsv)
            cv2.imshow("OpenCV/Numpy Result", res_img)

        return [center_x, center_y]
    else:
        return None
    pass


def press_keys(key_indices):
    # Release all relevant keys
    for key_code in key_codes:
        input_handler.release_key(key_code)

    key_index_array = []

    # Press keys
    for i in range(len(key_indices)):
        if key_indices[i] >= 0.5:
            key_index_array.append(i)
            input_handler.press_key(key_codes[i])
    pass


def get_img_values_in_square(image, image_width, top_left, dimensions):
    values = np.array([])
    start_point = image_width * top_left["y"] + top_left["x"]

    # Get all pixels from the square by looping and getting each row
    for i in range(dimensions["height"]):
        temp_list = image[start_point + (image_width * i): start_point + (image_width * i) + dimensions["width"] + 1]
        for value in temp_list:
            values = np.append(values, value)
    return values
    pass


# Dummy callback for tracking window
def nothing(x):
    pass


main()
