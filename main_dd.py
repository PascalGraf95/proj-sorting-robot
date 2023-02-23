"""
Autor: Dominic Doetterer (Mat.- Nr. 201974)
Date: 19.08.2022 - 19.12.2022
Information: Bachelor thesis "Clustering of image features of unknown objects and their sorting by means of a roboter"
MAIN PROGRAM
"""

########################################################################################################
# Libraries
########################################################################################################
# Import default libraries
import os
import glob
import pickle
import shutil
import serial.tools.list_ports
import time
import numpy as np
import imutils

# Import own libraries
import GUI
import Setup
import Conveyor
import DoBot_Robot
import IDS_Camera
import imageProcessing
import imageProcessing as ImPr
import clustering as clu
import neural_network as nn

# Import additional libraries
import multiprocessing
import continuous_threading
import cv2
from tensorflow import keras
from keras import layers

# Change to True to calibrate the experimental setup
Calibration = False
# Change to True to get all available information in the console output
verbose = False
# Change to False for no reset of data/Clustering/Collected Data/Scanned Items by starting scanning after reboot
deleteDataSet = True

########################################################################################################
# Generate global objects and variables
########################################################################################################
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
home = GUI.Home()
mode = home.getMode()
ModeControl = GUI.Mode()


###########################################################################################################
# PROCESSING PROCESS
###########################################################################################################
# Capture and edit Data
# takes on the task of data processing
# Provides the main functions: Scanning, Clustering and Sorting
def Processing_Proc():
    print('[Info] Process for Processing created')
    # Destructors for objets inside the process
    IDScam = IDS_Camera.IDS()
    Robot = DoBot_Robot.Robot()

    # Set Variable for Process
    Proc_Mode = None
    Mode_Change = False
    Processing_Proc_ShutDown = False
    startSorting = False
    reloadImage = True
    NewScan = True

    stopLine = 200
    X = []

    # Extract bottleneck features rom the pretrained model
    PretrainedModel = nn.Get_Model()
    input_layer = PretrainedModel.input
    flattened = PretrainedModel.get_layer("Bottleneck").output
    flattened = layers.Flatten()(flattened)
    bottleneck = keras.Model(input_layer, flattened)

    # auto homing of robot
    Robot.homing()

    while not Processing_Proc_ShutDown:
        # Read mode after change of a parameter in the GUI
        while os.path.exists('Control/Proc_Control.pkl') and not os.path.exists(Setup.PCF_Path):
            try:
                # Opens new Mode
                shutil.copy2('Control/Proc_Control.pkl', 'Control/Cache_Proc_Control.pkl')
                file = open('Control/Cache_Proc_Control.pkl', 'rb')
                Proc_Mode = pickle.load(file)
                file.close()
                os.remove('Control/Proc_Control.pkl')
            except:
                pass

        # The process just works if the mode is set
        if Proc_Mode is not None:
            # Check each lap if the Process must be shut down
            if Proc_Mode.getShutDown():
                Processing_Proc_ShutDown = True
                cv2.destroyAllWindows()
                Robot.disconnect()
                print('[Info] Process for Processing Shutdown {}'.format(Processing_Proc_ShutDown))
                break

            ###########################################################################################################
            # ROBOT HOMING ALGORITHMS
            ###########################################################################################################
            # Send the robot to its home position
            if Proc_Mode.getRoboter():
                Robot.homing()
                Proc_Mode.setRoboter(mode=False)
                Mode_Change = True

            ###########################################################################################################
            # SCANNING ALGORITHMS
            ###########################################################################################################
            # Scan items on conveyor and save to Path data/Clustering/Collected Data/Scanned Items
            if Proc_Mode.getScanning():
                if NewScan:
                    if deleteDataSet:
                        ImPr.CLearPathFolder()
                    NewScan = False

                else:
                    print('[Info] Scanning Items')
                    frame = IDScam.getimage()
                    ImPr.ProcessFrame(frame)

            ###########################################################################################################
            # Clustering ALGORITHMS
            ###########################################################################################################
            # Create cluster and dataset
            elif Proc_Mode.getClustering():
                print('[Info] Start Clustering')

                # Clear existing data of the destinations folder
                files = glob.glob(Setup.Clu_X_Data + '/*')
                for file in files:
                    os.remove(file)

                # generate dataset of objects from existing folder
                for img in os.listdir(Setup.ImageBlackBG_Path):
                    img_array = cv2.imread(os.path.join(Setup.ImageBlackBG_Path, img))
                    small_img_array = cv2.resize(img_array, (Setup.NN_IMG_SIZE, Setup.NN_IMG_SIZE))
                    X.append(small_img_array)
                    if verbose:
                        print("[Info] Picture {} imported".format(img))

                # Save Dataset with all pictures in (80x80x3) format
                pickle_out = open(Setup.Clu_X_Data, "wb")
                pickle.dump(X, pickle_out)
                pickle_out.close()

                # Preprocessing dataset to generate the cluster
                # A neural Network works better with values of {0-1}
                CompareData = X.copy()
                X = np.array(X)
                X = X / 255

                # Extract the features of each object
                Data2Cluster = bottleneck.predict(X)

                # Generate and save a kMeans model of the dataset
                # The optimal amount of clusters for the unsupervised sorting will be determined
                clu.GenerateFitModel(Data2Cluster)
                kMeans = clu.GetFittedModel()

                # Generate, save and display a montage of all items to get clue, what will be in one cluster
                nCLuster = clu.get_best_num_of_Cluster()
                y_clusters = kMeans.predict(Data2Cluster)
                clu.CreateItemMontage(nCLuster, y_clusters, Data=CompareData)
                cv2.waitKey()
                cv2.destroyAllWindows()

                Proc_Mode.setClustering(False)
                Mode_Change = True
                print('[Info] Cluster created')

            ###########################################################################################################
            # SORTING ALGORITHMS
            ###########################################################################################################
            # Sort the items from the conveyor with the roboter
            # Drop the object to an individual station for each cluster
            elif Proc_Mode.getSorting():
                print('[Info] Sorting in Process')
                frame = IDScam.getimage()
                (medianBlurThresh, image) = ImPr.getThresh(frame)
                output = image.copy()

                contours = cv2.findContours(medianBlurThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
                contours = imutils.grab_contours(contours)

                if len(contours) > 0:
                    print('[INFO] {} different contours found'.format(len(contours)))
                    allX = []
                    allY = []
                    allWidth = []
                    allHeight = []
                    allAngle = []
                    nContour = 0

                    while nContour < len(contours):
                        # generate most small rectangular contour of an Item
                        rect = cv2.minAreaRect(contours[nContour])
                        (x, y), (width, height), angle = rect

                        angle = imageProcessing.getItemAngle(width, height, angle)
                        # Check if object is not on the boundary or too big to grasp
                        if x - (width / 2) > 600 or min(width, height) > 70:
                            # Edit item only if it is not on the right boundary
                            # Make sure that have enough information about the item are visible
                            pass
                        elif 20 < width < 500 and 20 < height < 500:
                            box = cv2.boxPoints(rect)
                            box = np.int0(box)
                            cv2.drawContours(output, [box], 0, (0, 0, 255), 2)
                            cv2.line(output, (stopLine, 0), (stopLine, Setup.Track_belt_y_height), color=(255, 0, 0))
                            cv2.putText(output, '#{}'.format(nContour + 1), (int(x) + 10, int(y) + 10),
                                        fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0))

                            allX.append(x)
                            allY.append(y)
                            allWidth.append(width)
                            allHeight.append(height)
                            allAngle.append(angle)
                            cv2.imshow('output', output)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                cv2.destroyWindow('output')
                        nContour = nContour + 1

                    cv2.imshow('output', output)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyWindow('output')

                    if len(allX) > 0:
                        var = 0
                        listNr = 0

                        for i in allX:
                            if i == min(allX):
                                var = listNr
                            listNr = listNr + 1

                        if abs(allX[var] - (allWidth[var] / 2)) < stopLine:
                            if not startSorting:
                                Proc_Mode.setConveyor(Setup.ConveyorMode_Forward, mode=False)
                                Proc_Mode.setConveyor(Setup.ConveyorMode_Stop, mode=True)
                                if verbose:
                                    print('[INFO] A item hit on the Scanline')
                                Mode_Change = True
                                startSorting = True

                            else:
                                print('Start Sorting Frame')
                                # wait that the conveyor stops and get a new frame
                                if reloadImage:
                                    if verbose:
                                        print('[INFO] Reload new image for accurate position')
                                    time.sleep(5)
                                    reloadImage = False

                                else:
                                    # Sorting all items from frame
                                    for obj in range(len(allX)):
                                        print(obj)
                                        ((x, y), (width, height), angle) = (
                                            (allX[obj], allY[obj]), (allWidth[obj], allHeight[obj]), allAngle[obj])
                                        rect = (x, y), (width, height), angle

                                        box = cv2.boxPoints(rect)
                                        box = np.int0(box)
                                        src_pts = box.astype("float32")
                                        # coordinate of the points in box points after the rectangle has been
                                        # straightened
                                        dst_pts = np.array([[0, int(height) - 1],
                                                            [0, 0],
                                                            [int(width) - 1, 0],
                                                            [int(width) - 1, int(height) - 1]], dtype="float32")
                                        # the perspective transformation matrix
                                        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                                        # directly warp the rotated rectangle to get the straightened rectangle
                                        warped = cv2.warpPerspective(image, M, (int(width), int(height)))
                                        if width >= height:
                                            warped = cv2.rotate(warped, cv2.cv2.ROTATE_90_CLOCKWISE)
                                        item = []
                                        BlackBG = ImPr.ItemOnBlackBG(warped)

                                        # Prepare the items do get sorted into its corresponding cluster
                                        cv2.imwrite('data/Clustering/Collected Data/Cache/Item.png', BlackBG)
                                        img_array = cv2.imread('data/Clustering/Collected Data/Cache/Item.png')
                                        small_img_array = cv2.resize(img_array, (Setup.NN_IMG_SIZE, Setup.NN_IMG_SIZE))
                                        item.append(small_img_array)

                                        # Prepare item to extract the features
                                        PictureOfItem = np.array(item)
                                        PictureOfItem = PictureOfItem / 255
                                        PictureOfItem = np.array(PictureOfItem).reshape(-1, 80, 80, 3)

                                        # Extract features and assign the item to a cluster
                                        FeatureOfItem = bottleneck.predict(PictureOfItem)
                                        kMeans = clu.GetFittedModel()
                                        item_yCluster = kMeans.predict(FeatureOfItem)
                                        cv2.putText(output, 'Cluster{}'.format(item_yCluster + 1),
                                                    (int(x) + 10, int(y) + 25),
                                                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0))

                                        cv2.imshow('output', output)
                                        if cv2.waitKey(1) & 0xFF == ord('q'):
                                            cv2.destroyWindow('output')

                                        (x_roboter, y_roboter) = DoBot_Robot.CalcRobotCoord(x, y)
                                        if verbose:
                                            print('x_rob {}'.format(obj + 1), x_roboter)
                                            print('y_rob {}'.format(obj + 1), y_roboter)
                                            print('angle {}'.format(obj + 1), angle)

                                        # Picking and sorting routine
                                        Robot.gotoPos(x_roboter, y_roboter, Setup.DoBot_belt_z, angle)
                                        Robot.pick_Item()
                                        Robot.GoToStorage(item_yCluster + 1)
                                        Robot.release_Item()

                                    # All object of this frame are sorted
                                    startSorting = False
                                    reloadImage = True
                                    Proc_Mode.setConveyor(Setup.ConveyorMode_Forward, mode=True)
                                    Proc_Mode.setConveyor(Setup.ConveyorMode_Stop, mode=False)
                                    Mode_Change = True

            if Proc_Mode.getStop():
                cv2.destroyAllWindows()
                Proc_Mode.setStop(mode=False)
                Mode_Change = False

            ###########################################################################################################
            # PUBLISH CHANGES
            ###########################################################################################################
            # Release the new mode the administrative Thread to make it globally available
            if Mode_Change:
                while Mode_Change:
                    try:
                        os.remove(Setup.PCC_Path)
                        cache = open(Setup.PCC_Path, 'wb')
                        pickle.dump(Proc_Mode, cache)
                        cache.close()

                        shutil.copy2(Setup.PCC_Path, Setup.PCF_Path)
                        os.remove(Setup.PCC_Path)
                        Mode_Change = False
                        print('Mode Process change done')
                    except:
                        pass


###########################################################################################################
# CONVEYOR PROCESS
###########################################################################################################
# Control and administration of the Conveyor
# Enable parallel processing
def Conveyor_Proc():
    print('[Info] Process for Conveyor created')
    Conveyor_Belt = Conveyor.Belt()
    Conveyor_Proc_ShutDown = False
    Con_Mode = None
    while not Conveyor_Proc_ShutDown:
        if Con_Mode is not None:
            if Con_Mode.getShutDown():
                Conveyor_Proc_ShutDown = True
                Conveyor_Belt.shutdown()
                print('[Info] Process for Conveyor Shutdown {}'.format(Conveyor_Proc_ShutDown))
                break

        # Stars the conveyor when there is a change of the mode
        while os.path.exists(Setup.CC_Path) and not os.path.exists(Setup.CCF_Path):
            try:
                # Opens new Mode
                shutil.copy2(Setup.CC_Path, Setup.CCC_Path)
                file = open(Setup.CCC_Path, 'rb')
                Con_Mode = pickle.load(file)
                file.close()
                os.remove(Setup.CCC_Path)
                if Con_Mode.getConveyor(Setup.ConveyorMode_Forward):
                    Conveyor_Belt.forward()
                    Con_Mode.setConveyor(Setup.ConveyorMode_Forward, False)
                elif Con_Mode.getConveyor(Setup.ConveyorMode_Backward):
                    Conveyor_Belt.backward()
                    Con_Mode.setConveyor(Setup.ConveyorMode_Backward, False)
                elif Con_Mode.getConveyor(Setup.ConveyorMode_Slow):
                    Conveyor_Belt.slowdown()
                    Con_Mode.setConveyor(Setup.ConveyorMode_Slow, False)
                elif Con_Mode.getConveyor(Setup.ConveyorMode_Fast):
                    Conveyor_Belt.speedup()
                    Con_Mode.setConveyor(Setup.ConveyorMode_Fast, False)
                elif Con_Mode.getConveyor(Setup.ConveyorMode_Stop):
                    Conveyor_Belt.stop()
                    Con_Mode.setConveyor(Setup.ConveyorMode_Stop, False)
                os.remove(Setup.CC_Path)

                write = True
                while write:
                    try:
                        cache = open(Setup.CCC_Path, 'wb')
                        pickle.dump(Con_Mode, cache)
                        cache.close()
                        shutil.copy2(Setup.CCC_Path, Setup.CCF_Path)
                        os.remove(Setup.CCC_Path)
                        write = False
                    except:
                        pass
            except:
                pass


###########################################################################################################
# MODE THREAD
###########################################################################################################
# This is a workaround, because "mode" is an object and a child of the GUI object. The mode object of the GUI have to
# be in the same process as the GUI. But each process needs the information of the actual GUI state.
# There a no classic pointer in Python to provide the object for another process.
# The thread, started in the same process as the GUI has the opportunity to read the parameters of the mode object.
# This thread provides the mode of the GUI for all the other processes by writing them into cache data.
# By reading the cache data in the process the changes are noticed
def Thread_Mode(ModeControl, mode):
    print('[Info] Thread for ModeControl created')

    def Init(ModeControl, mode):
        ModeControl.setConveyor(Setup.ConveyorMode_Forward, mode.getConveyor(Setup.ConveyorMode_Forward))
        ModeControl.setConveyor(Setup.ConveyorMode_Backward, mode.getConveyor(Setup.ConveyorMode_Backward))
        ModeControl.setConveyor(Setup.ConveyorMode_Fast, mode.getConveyor(Setup.ConveyorMode_Fast))
        ModeControl.setConveyor(Setup.ConveyorMode_Slow, mode.getConveyor(Setup.ConveyorMode_Slow))
        ModeControl.setConveyor(Setup.ConveyorMode_Stop, mode.getConveyor(Setup.ConveyorMode_Stop))

        ModeControl.setCamera(mode.getCamera())
        ModeControl.setShutDown(mode.getShutDown())
        ModeControl.setThreshold(mode.getThreshold())
        ModeControl.setClustering(mode.getClustering())
        ModeControl.setScanning(mode.getScanning())
        ModeControl.setStop(mode.getStop())
        ModeControl.setSorting(mode.getSorting())
        ModeControl.setRoboter(mode.getRoboter())

        write = True
        while write:
            try:
                cache = open('Control/Cache_Control.pkl', 'wb')
                pickle.dump(ModeControl, cache)
                cache.close()
                shutil.copy2('Control/Cache_Control.pkl', Setup.PC_Path)
                shutil.copy2('Control/Cache_Control.pkl', Setup.CC_Path)
                os.remove('Control/Cache_Control.pkl')
                mode.set_is_changed(mode=False)
                write = False
            except:
                pass

    if mode:
        Init(ModeControl, mode)
        print('[INFO] Mode Init done')
        Thread_Mode_ShutDown = False
        while not Thread_Mode_ShutDown:
            if mode.is_changed():
                Init(ModeControl, mode)
                if mode.getShutDown():
                    Thread_Mode_ShutDown = True

            else:
                if os.path.exists(Setup.CCF_Path):
                    mode.setConveyor(Setup.ConveyorMode_Forward, mode=False)
                    mode.setConveyor(Setup.ConveyorMode_Backward, mode=False)
                    mode.setConveyor(Setup.ConveyorMode_Fast, mode=False)
                    mode.setConveyor(Setup.ConveyorMode_Slow, mode=False)
                    mode.setConveyor(Setup.ConveyorMode_Stop, mode=False)
                    mode.set_is_changed(False)
                    while os.path.exists(Setup.CCF_Path):
                        try:
                            os.remove(Setup.CCF_Path)
                            # print('Con Flag removed')
                            break
                        except:
                            pass

                if os.path.exists(Setup.PCF_Path) and not os.path.exists(Setup.PCC_Path):
                    while os.path.exists(Setup.PCF_Path):
                        try:
                            file = open(Setup.PCF_Path, 'rb')
                            Flag_Mode = pickle.load(file)
                            file.close()
                            mode.setConveyor(Setup.ConveyorMode_Forward,
                                             Flag_Mode.getConveyor(Setup.ConveyorMode_Forward))
                            mode.setConveyor(Setup.ConveyorMode_Stop, Flag_Mode.getConveyor(Setup.ConveyorMode_Stop))
                            mode.setCamera(Flag_Mode.getCamera())
                            mode.setClustering(Flag_Mode.getClustering())
                            mode.setRoboter(Flag_Mode.getRoboter())

                            os.remove(Setup.PCF_Path)
                            break

                        except:
                            pass


###########################################################################################################
# DEVICES CONNECT ALGORITHMS
###########################################################################################################
# Check if the robot and conveyor are connected
# Set the com ports automatically to the ports where they belong to
# Set the com ports automatically to the ports where they belong to
def AllDevicesConnected():
    print('[INFO] Connect the Dobot and Arduino')
    Dobot = None
    Arduino = None
    while True:
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            desc = p.description
            name = []
            for letters in desc:
                name.append(letters)

            # Arduino
            if name[0:7] == ['A', 'r', 'd', 'u', 'i', 'n', 'o']:
                if Arduino is None:
                    print(p.description)
                    Port_Arduino = p.name

                    # Write the comport to a data to make it global
                    pickle_out = open(Setup.Cal_Arduino_Port_Path, "wb")
                    pickle.dump(Port_Arduino, pickle_out)
                    pickle_out.close()

                    Arduino = True
                else:
                    pass

            # Dobot
            elif name[0:7] == ['S', 'i', 'l', 'i', 'c', 'o', 'n']:
                if Dobot is None:
                    print(p.description)
                    Port_Dobot = p.name

                    # Write the comport to a data to make it global
                    pickle_out = open(Setup.Cal_Dobot_Port_Path, "wb")
                    pickle.dump(Port_Dobot, pickle_out)
                    pickle_out.close()

                    Dobot = True
                else:
                    pass
        cv2.waitKey()
        if Dobot and Arduino:
            break


###########################################################################################################
# DEVICES CONNECT ALGORITHMS
###########################################################################################################
# the main function generate and start each process and thread
def main():
    # Calibrate the needed robot coordinates and the threshold of the image processing
    if Calibration:
        IDScam = IDS_Camera.IDS()
        ImPr.CalibrateThreshold(IDScam)
        DoBot_Robot.CalibrateCoordinates(IDScam)
        print('[INFO] Calibration Done')
        print('[INFO] Change "Calibration" back to False')

    else:
        for file in glob.glob('Control/*'):
            os.remove(file)
        AllDevicesConnected()

        # Define the thread
        thread_Mode = continuous_threading.Thread(target=Thread_Mode, args=(ModeControl, mode))
        thread_Mode.start()

        # Define the Processes
        process_Conveyor = multiprocessing.Process(target=Conveyor_Proc)
        process_Processing = multiprocessing.Process(target=Processing_Proc)

        # Start the Processes
        process_Conveyor.start()
        process_Processing.start()

        # create GUI
        home.createGUI()

        # waiting for Processes and Threads to be done
        thread_Mode.join()
        process_Conveyor.join()
        process_Processing.join()


if __name__ == "__main__":
    main()
