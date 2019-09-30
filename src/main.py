from ScreenshotMachine import ScreenshotMachine


def main():
    screenshot_machine = ScreenshotMachine()
    screenshot_machine.set_coordinates(18, 153, 762, 467)
    screenshot_machine.take_screenshots(15)


main()
