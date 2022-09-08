import numba
import cv2
from numba import jit
import time
import numpy as np
from numpy import int16, uint8, log2
import cv2
import pydvs.generate_spikes as gs
#-------------------------------------------------------------------------------------------------------------
width = 1080
height = 720

shape = (height, width)


history_weight = 1.0
threshold = 12 # ~ 0.05*255
# max_threshold = 180 # 12*15 ~ 0.7*255

# scale_width = 0
# scale_height = 0
# col_from = 0
# col_to = 0

curr     = np.zeros(shape,     dtype=int16) 
ref      = 128*np.ones(shape,  dtype=int16) 
spikes   = np.zeros(shape,     dtype=int16) 
diff     = np.zeros(shape,     dtype=int16) 
abs_diff = np.zeros(shape,     dtype=int16) 

# just to see things in a window
spk_img  = np.zeros((height, width, 3), np.uint8)

num_bits = 6   # how many bits are used to represent exceeded thresholds
num_active_bits = 2 # how many of bits are active
log2_table = gs.generate_log2_table(num_active_bits, num_bits)[num_active_bits - 1]
# spike_lists = None
# pos_spks = None
# neg_spks = None
# max_diff = 0
#-------------------------------------------------------------------------------------------------------------
@jit( nopython=True)
def thresholded_difference(curr_frame,ref_frame,threshold):
  """
    :param curr_frame: Latest image captured by the camera
    :param ref_frame:  Saves value when the pixel was marked as "spiking"
    :param threshold:  How big the difference between current and reference
                       frames needs to be to mark a pixel as spiking

    :return diff:     Signed difference between current and reference frames
    :return abs_diff: Absolute value of the difference
    :return spikes:   Signed pixels marked as spiking (-1 means change from higher
                      to lower brightness value, 1 means change from lower to
                      higher brightness value.)
  """
  # cdef np.ndarray[DTYPE_t, ndim=2] diff, abs_diff, spikes
  # cdef np.ndarray[DTYPE_IDX_t, ndim=1] neg_r, neg_c

  diff = curr_frame - ref_frame
  abs_diff = np.abs(diff)

  spikes = (abs_diff > threshold).astype(np.int16)

  abs_diff = (abs_diff*spikes)

  neg_r, neg_c = np.where(diff < -threshold)
  # spikes[neg_r, neg_c] = -1
  for f, b in zip(neg_r, neg_c):
    spikes[f,b] = -1


  return diff, abs_diff, spikes
#----------------------------------------------------------------------------------------------------------------------------
@jit(nopython=True)
def update_reference_time_binary_thresh(abs_diff,spikes,ref_frame,threshold,history_weight,log2_table):

  """
    Time based spike transmission.
    :param abs_diff:        Absolute value of the difference of current frame and
                            reference frame (computed in *thresholded_difference*)
    :param spikes:          Pixels marked as spiking
    :param ref_frame:       Previous reference frame
    :param threshold:       How much brightness has to change to mark a pixel as spiking
    :param num_spikes:      Number of active bits to use in the encoding
    :param history_weight:  How much does the previous reference frame weighs in the
                            update equation
    :param log2_table:      Precomputed versions of the raw difference using only num_spikes
                            active bits
    :returns ref_frame:     Updated reference frame

    Binary encoding of the number of thresholds rebased by the brightness difference,
    using only num_spikes active bits

    TH <- Threshold value
    T  <- Inter frame time (1/frames per second)

   t=0 _______________|_____|________ t=T
       |     |     |     |     |     |
   bit    5     4     3     2     1
   val                8     2
   original = 125, threshold = 12 => ~10 thresholds
   encoded with only 2 active bits = 10*12 = 120

  """
  logTab = abs_diff//threshold
  mult = log2_table.take(logTab)
  # mult = (log2_table[abs_diff//threshold])

  ref_frame = np.clip( (history_weight*ref_frame) + spikes*mult*threshold, 0, 255)

  return ref_frame

# @jit(nopython=True,parallel=True,)
def render_frame(spikes,curr_frame,width,height):
  """
    Overlaps the generated spikes onto the latest image from the video
    source. Red means a negative change in brightness, Green a positive one.

    :param spikes:     Pixels marked as spiking
    :param curr_frame: Latest image from the video source
    :param width:      Image width
    :param height:     Image height
    :returns spikes_frame: Combined spikes/image information in a color image
  """
  spikes_frame = np.zeros((height, width, 3), np.uint8)
  # cdef np.ndarray[Py_ssize_t, ndim=1] rows, cols
#   spikes_frame[:, :, 0] = curr_frame
#   spikes_frame[:, :, 1] = curr_frame
#   spikes_frame[:, :, 2] = curr_frame

  spikes_frame= np.repeat(curr_frame[:, :, np.newaxis], 3, axis=2)

#   spikes_frame = np.tile(curr_frame,(3,1,1))

  rows, cols = np.where(spikes > 0)
  for r,c in zip(rows, cols):
    spikes_frame[r, c, :] = [0, 255, 0]

  rows, cols = np.where(spikes < 0)
  for r,c in zip(rows, cols):
    spikes_frame[r, c, :] = [0, 0, 255]
  # spikes_frame[rows, cols, :] = [0, 0, 255]

  return spikes_frame
#-------------------------------------------------------------------------------------------------
def grab_frame(dev, width, height):
  _, raw = dev.read()
  img = cv2.resize(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(int16),(width, height))
  return img


#---------------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------- #
    # camera/frequency related                                             #

    video_dev = cv2.VideoCapture(0) # webcam
    #video_dev = cv2.VideoCapture('/path/to/video/file') # webcam

    print(video_dev.isOpened())

    #ps3 eyetoy can do 125fps
    try:
        video_dev.set(cv2.CAP_PROP_FPS, 30)
    except:
        pass
    
    fps = video_dev.get(cv2.CAP_PROP_FPS)
    if fps == 0.0:
        fps = 30.0

    print(fps)
    
    #.......................................................
    WINDOW_NAME = 'spikes'
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.startWindowThread()

    is_first_pass = True
    start_time = time.time()
    end_time = 0
    frame_count = 0

    while True:
        curr = grab_frame(video_dev, width, height)

        diff[:], abs_diff[:], spikes[:] = thresholded_difference(curr, ref, threshold)
        ref[:] = update_reference_time_binary_thresh(abs_diff, spikes, ref,threshold,history_weight,log2_table)
        #spk_img[:] = render_frame(spikes, curr, width,height) 
        spk_img[:] =gs.render_frame(spikes, curr, width,height, 2) 
        cv2.imshow (WINDOW_NAME, spk_img.astype(uint8))  
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        end_time = time.time()
        
        if end_time - start_time >= 1.0:
            print("%d frames per second"%(frame_count))
            frame_count = 0
            start_time = time.time()
        else:
            frame_count += 1

    cv2.destroyAllWindows()
    cv2.waitKey(1)



