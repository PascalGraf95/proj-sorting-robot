from pyueye import ueye
import numpy as np
import cv2


class IDSCameraController:
    def __init__(self):
        # Variables
        self.h_cam = ueye.HIDS(0)  # 0: first available camera;  1-254: The camera with the specified camera ID
        self.pcImageMemory = ueye.c_mem_p()
        self.mem_id = ueye.int()
        rect_aoi = ueye.IS_RECT()
        self.pitch = ueye.INT()
        self.nBitsPerPixel = ueye.INT(24)  # 24: bits per pixel for color mode; take 8 bits per pixel for monochrome
        m_ncolormode = ueye.INT()  # Y8/RGB16/RGB24/REG32
        # ---------------------------------------------------------------------------------------------------------------------

        # Starts the driver and establishes the connection to the camera
        ueye.is_InitCamera(self.h_cam, None)

        pParam = ueye.wchar_p()
        pParam.value = r"modules\configs\camera_parameters_230224.ini"
        ueye.is_ParameterSet(self.h_cam, ueye.IS_PARAMETERSET_CMD_LOAD_FILE, pParam, 0)

        # Set display mode to DIB
        ueye.is_SetDisplayMode(self.h_cam, ueye.IS_SET_DM_DIB)

        # Set the right color mode
        ueye.is_GetColorDepth(self.h_cam, self.nBitsPerPixel, m_ncolormode)

        # Can be used to set the size and position of an "area of interest"(AOI) within an image
        ueye.is_AOI(self.h_cam, ueye.IS_AOI_IMAGE_GET_AOI, rect_aoi, ueye.sizeof(rect_aoi))

        self.width = rect_aoi.s32Width
        self.height = rect_aoi.s32Height

        # Allocates an image memory for an image having its dimensions defined by
        # self.width and self.height and its color depth defined by self.nBitsPerPixel
        ueye.is_AllocImageMem(self.h_cam, self.width, self.height, self.nBitsPerPixel, self.pcImageMemory, self.mem_id)
        ueye.is_SetImageMem(self.h_cam, self.pcImageMemory, self.mem_id)
        ueye.is_SetColorMode(self.h_cam, m_ncolormode)

        # Activates the camera's live video mode (free run mode)
        ueye.is_CaptureVideo(self.h_cam, ueye.IS_DONT_WAIT)
        # Enables the queue mode for existing image memory sequences
        ueye.is_InquireImageMem(self.h_cam, self.pcImageMemory, self.mem_id, self.width, self.height,
                                self.nBitsPerPixel, self.pitch)
        
    def capture_image(self):
        # Read image from camera
        array = ueye.get_data(self.pcImageMemory, self.width, self.height, self.nBitsPerPixel, self.pitch, copy=False)
        bytes_per_pixel = int(self.nBitsPerPixel / 8)
    
        # Reshape data into numpy array
        frame = np.reshape(array, (self.height.value, self.width.value, bytes_per_pixel))
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    
    def close_camera_connection(self):
        # Releases an image memory that was allocated using is_AllocImageMem() and removes it from the driver management
        ueye.is_FreeImageMem(self.h_cam, self.pcImageMemory, self.mem_id)

        # Disables the hCam camera handle and releases the data structures and memory areas taken up by the uEye camera
        ueye.is_ExitCamera(self.h_cam)


class WebcamCameraController:
    def __init__(self, cam_idx=0):
        self.vid = cv2.VideoCapture(cam_idx)

    def capture_image(self):
        ret, frame = self.vid.read()
        return frame

    def close_camera_connection(self):
        self.vid.release()


# Mouse callback function to update cursor position
def mouse_callback(event, x, y, flags, param):
    global cursor_x, cursor_y
    if event == cv2.EVENT_MOUSEMOVE:
        cursor_x, cursor_y = x, y


def main():
    # Initialize the camera
    cam = IDSCameraController()
    '''
    import image_processing
    # Global variables to store cursor position
    global cursor_x, cursor_y
    cursor_x, cursor_y = -1, -1

    # Connect to the camera and capture images
    for i in range(10000):
        frame = cam.capture_image()
        frame = image_processing.image_preprocessing(frame)

        # Add the cursor coordinates to the image
        cursor_position = f"({cursor_x}, {cursor_y})"
        cv2.putText(frame, cursor_position, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Show the image
        cv2.imshow("Test", frame)
        cv2.setMouseCallback("Test", mouse_callback)

        # Check for key press to exit
        key = cv2.waitKey(1)
        if key == 27:  # Press 'Esc' to exit
            break
    '''
    # Close the camera connection and destroy all windows
    cam.close_camera_connection()
    cv2.destroyAllWindows()





if __name__ == '__main__':
    main()

