import cv2
import numpy as np
from skimage import io
from django.conf import settings

# from tutorial https://subscription.packtpub.com/book/application-development/9781785282690/1/ch01lvl1sec11/generating-a-warming-cooling-filter
from scipy.interpolate import UnivariateSpline

def _create_LUT_8UC1(x, y):
  spl = UnivariateSpline(x, y)
  return spl(range(256))

incr_ch_lut = _create_LUT_8UC1([0, 64, 128, 192, 256], [0, 70, 140, 210, 256])
incr_ch_lut_by_5 = _create_LUT_8UC1([0, 64, 128, 192, 256], [0, 69, 133, 198, 256])
decr_ch_lut = _create_LUT_8UC1([0, 64, 128, 192, 256], [0, 30, 80, 120, 192])
decr_ch_lut_by_5 = _create_LUT_8UC1([0, 64, 128, 192, 256], [0, 59, 123, 187, 192])

class Camera(object):
    def __init__(self, animal):
        self.videoStream = cv2.VideoCapture(1)
        self.animal = animal
        self.endStream = False

    def __del__(self):
        # release webcam
        self.videoStream.release()
        # closes windows from imshow()
        cv2.destroyAllWindows()

    def startStream(self):
        self.endStream = False

    def endStream(self):
        self.endStream = True

    def getVideoFrame(self):
        while(True):
            isReturn, frame = self.videoStream.read()
            if (self.animal == "whale"):
                #increase contrast/brightness?

                img_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                filtered = cv2.blur(src=img_grey, ksize=(10, 10))

            elif (self.animal == "human"):
                filtered = frame
            
            elif (self.animal == "cat"):
                # convert to hsv colorspace
                img_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                # change datatype to float
                img_hsv = np.array(img_hsv, dtype=np.float64)
                # scale pixel values for red hue
                img_hsv[:, :, 0][ np.logical_and(img_hsv[:, :, 0] > 48, img_hsv[:, :, 0] < 287) ] = 85
                # change datatype back to int
                img_hsv = np.array(img_hsv, dtype=np.uint8)

                # split hsv channels
                hue, sat, val = cv2.split(img_hsv)
                # make lighter by increasing value channel
                val = cv2.LUT(val, _create_LUT_8UC1([0, 64, 128, 192, 256], [80, 130, 180, 220, 256])).astype(np.uint8)
                # make less saturated by decreasing saturation channel
                sat = cv2.LUT(sat, _create_LUT_8UC1([0, 64, 128, 192, 256], [0, 20, 40, 50, 60])).astype(np.uint8)
                # merge hsv channels
                img_hsv = cv2.merge((hue, sat, val))
                # convert to RGB colorspace
                image_colored = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

                # blur 
                filtered = cv2.blur(src=image_colored, ksize=(5, 3))

            elif (self.animal == "cow"):
                img_rgb = np.array(frame, dtype=np.float64)
                val = 1.8
                # scale pixel values up or down for red(?)
                img_rgb[:, :, 2] = img_rgb[:, :, 2] * val
                # setting values > 255 to 255.
                img_rgb[:, :, 2][img_rgb[:, :, 2] > 255] = 255 
                img_rgb = np.array(img_rgb, dtype=np.uint8)

                # split channels
                red, blue, green = cv2.split(img_rgb)
                # increase red
                red = cv2.LUT(red, incr_ch_lut).astype(np.uint8)
                img_rgb = cv2.merge((red, blue, green))

                # zoom in
                filtered = cv2.resize(img_rgb, None, fx= 1.2, fy= 1.2, interpolation= cv2.INTER_LINEAR)
            
            elif (self.animal == "rattlesnake"):
                filtered = cv2.applyColorMap(frame, 4)

            elif (self.animal == "alien"):
                lut = np.zeros((256, 1, 3), dtype=np.uint8)
                # read file path of original image
                filename = settings.MEDIA_ROOT[0:-6] + "\\media\\cat_map.png"
                camera = io.imread(filename)

                for idx, pixelArr in enumerate(camera[0]):
                    lut[idx] = pixelArr[0:3]

                filtered = cv2.LUT(frame, lut)

            else:
                # from https://medium.com/featurepreneur/colour-filtering-and-colour-pop-effects-using-opencv-python-3ce7d4576140
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                #obtain the grayscale image of the original image
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                #set the bounds for the red hue
                lower_red = np.array([160,100,50])
                upper_red = np.array([180,255,255])

                #create a mask using the bounds set
                mask = cv2.inRange(hsv, lower_red, upper_red)
                #create an inverse of the mask
                mask_inv = cv2.bitwise_not(mask)
                #Filter only the red colour from the original image using the mask(foreground)
                res = cv2.bitwise_and(frame, frame, mask=mask)
                #Filter the regions containing colours other than red from the grayscale image(background)
                background = cv2.bitwise_and(gray, gray, mask = mask_inv)
                #convert the one channelled grayscale background to a three channelled image
                background = np.stack((background,)*3, axis=-1)
                #add the foreground and the background
                added_img = cv2.add(res, background)
                filtered = added_img
        
            isSuccessEncode, jpeg = cv2.imencode('.jpg', filtered)
            
            if self.endStream:
                break

            return jpeg.tobytes()
 
