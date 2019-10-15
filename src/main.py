from ScreenshotMachine import ScreenshotMachine
import cv2
import time
import win32gui
import re
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

input_handler = InputHandler()

# Isaac position
isaac_x = 0
isaac_y = 0


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
    door_data = [["Door Pixel Data"], ["Door Pixel Data"]]
    targets = [["Outputs"], ["Outputs"]]

    # Train the network
    for i in range(len(door_data)):
        network.train(door_data[i], targets[i])
    network.save_weights()


def start_door_network(screenshot_machine, neural_network_door_recognition, neural_network_door_movement):
    while True:
        # Take screenshot
        time.sleep(1 / 20)
        sct_img = screenshot_machine.take_screenshot()

        # Set the position of isaac gained from the screenshot
        find_isaac(sct_img)

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

        # Prepare data into flat array
        doors = np.array([left_door, right_door, down_door])
        inputs = np.array(top_door)
        for door in doors:
            for value in door:
                inputs = np.append(inputs, value)
        inputs = ((inputs / 255) * 0.99 + 0.01)
        output_door_recognition = neural_network_door_recognition.query(inputs)
        print(output_door_recognition)
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

        global isaac_x
        global isaac_y
        isaac_x = center_x
        isaac_y = center_y

        # print("Isaac center: {}, {}".format(center_x, center_y))

        if show_windows:
            cv2.imshow("OpenCV/Numpy HSV", hsv)
            cv2.imshow("OpenCV/Numpy Result", res_img)
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
