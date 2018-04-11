from r12.arm import *
import time

SPEED = 3000
USER_ANSWERS = ['y','n', 'Y', 'N','']
MODES = ['s', 'e', 'S', 'E']

is_calibrated = None
modality = None


def calibrate_initial():
    robot.write('DE-ENERGIZE')
    print("\nPlease manually move the robot in the upright position.")
    raw_input("\nWhen done, press enter to continue...")
    robot.write('ENERGIZE')
    time.sleep(2)
    robot.write('CALIBRATE')
    print(robot.read())
    print("Robot ready to operate\n")


def experiment_set_up():
    time.sleep(2)
    robot.write('TELL SHOULDER 4000 MOVE')
    time.sleep(2)
    robot.write('TELL ELBOW 8500 MOVE')
    # time.sleep(2)
    # robot.write('POINT PICKING_POS')
    # time.sleep(2)
    # robot.write('TELL ELBOW -1000 MOVE')
    # time.sleep(2)
    # robot.write('TELL SHOULDER 1300 MOVE')
    # time.sleep(2)
    # robot.write('POINT PICKED')
    # time.sleep(2)
    # robot.write('PICKING_POS GOTO')

def pick():
    time.sleep(2)
    robot.write('CARTESIAN 0 231.5 -162.9 MOVETO')

def back_up():
    time.sleep(2)
    robot.write('CARTESIAN 0 211.8 -63.1 MOVETO')


if __name__ == "__main__":

    # Start r12 and set initial speed
    robot = Arm()
    robot.connect()
    robot.write('START')
    print("Robot starting...")
    time.sleep(2)
    robot.write('{} SPEED !'.format(SPEED))
    print("Setting speed value to {}".format(SPEED))
    print(robot.read())

    #define useful robot variables
    # time.sleep(2)
    # robot.write(': PICK_POS CARTESIAN 0.0 211.8 -63.1 MOVETO ;')  # position to start picking from
    # time.sleep(2)
    # robot.write(': PICK TELL SHOULDER ? ELBOW ? MOVETO ;')  # change position to pick


    try:
        # Calibrate the robot to ensure start from the HOME position
        while is_calibrated not in USER_ANSWERS:
            is_calibrated = raw_input("Is the robot in the upright (home) position? [y/n]")
            if is_calibrated not in USER_ANSWERS:
                print('Please respond with [y/n]')
            elif is_calibrated in ['n','N']:
                calibrate_initial()


        while modality not in MODES:
            modality = raw_input("Enter the modality you would like to run the script in:"
                                      "\n    s(shell)\n    e(xperiment)\n\nAnwer: ")
            if modality not in MODES:
                print('Please respond with [s/e]')

        # ----------------- SHELL MODE -----------------
        if modality in ['s','S']:
            while True:
                cmd = raw_input("write command: ")
                robot.write(cmd)
                time.sleep(.5)
                response = robot.read()
                print(response)
        else:
            # ----------------- EXPERIMENT MODE -----------------
            print("\nSetting up the experiment...")
            experiment_set_up()
            while True:

                raw_input("\nPress enter to pick.")
                pick()
                time.sleep(2)
                back_up()

                # :
                #     cmd = raw_input("write command: ")
                #     robot.write(cmd)
                #     time.sleep(.3)
                #     response = robot.read()
                #     print(response)

        print("\n\nDisconnecting and exiting...")
        robot.disconnect()


    except:
        time.sleep(2)
        robot.write('HOME')
        time.sleep(2)
        print("\n\nDisconnecting and exiting...")
        robot.disconnect()