from ScreenshotMachine import ScreenshotMachine
import cv2
import time
from NeuralNetwork import NeuralNetwork
from InputHandler import InputHandler

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
input_nodes_amount = 7480
hidden_nodes_amount = 50
output_nodes_amount = 8
learning_rate = 0.3
hidden_layers = 2

input_handler = InputHandler()


def main():
    # Neural network
    neural_network = NeuralNetwork(input_nodes_amount, hidden_nodes_amount, output_nodes_amount, learning_rate, hidden_layers)
    key_index_array = [0]

    # Screenshot
    screenshot_machine = ScreenshotMachine()
    screenshot_machine.set_coordinates(68, 153, 662, 410)
    time.sleep(1)

    while True:
        time.sleep(1 / 20)
        sct_img = screenshot_machine.take_screenshot()

        cv2.imshow("OpenCV/Numpy grayscale", cv2.cvtColor(sct_img, cv2.COLOR_BGRA2GRAY))

        sct_img = cv2.cvtColor(sct_img, cv2.COLOR_BGRA2GRAY)
        sct_img = sct_img.flatten()

        inputs = sct_img
        output = neural_network.query(inputs)

        for key_code in key_index_array:
            input_handler.release_key(key_codes[key_code])

        key_index_array = []

        for i in range(len(output)):
            if output[i] >= 0.5:
                key_index_array.append(i)
                input_handler.press_key(key_codes[i])

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
    pass


main()


