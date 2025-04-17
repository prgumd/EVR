###############################################################################
#
# Track a patch by HSV color thresholding, with GUI for tuning
#
# History:
# 01-02-25 - Levi Burner - Created file
#
###############################################################################

import numpy as np
import cv2

def on_val_return_trackerbar_int(val, val_ret):
    val_ret[0] = val

def frame_thresh(frame, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi):
    frame_mask = np.ones(shape=frame.shape[:2], dtype=bool)
    # print('hi')
    # print(np.sum(frame_mask))
    # print(h_lo, h_hi, s_lo, s_hi, v_lo, v_hi)
    # print(np.min(frame[:, :, 0]), np.max(frame[:, :, 0]))

    if h_lo < h_hi:
        frame_mask = frame_mask & (frame[:, :, 0] >= h_lo)
        frame_mask = frame_mask & (frame[:, :, 0] <= h_hi)
    else:
        frame_mask = frame_mask & ((frame[:, :, 0] >= h_lo) | (frame[:, :, 0] <= h_hi))
    # print(np.sum(frame_mask))
    frame_mask = frame_mask & (frame[:, :, 1] >= s_lo)
    frame_mask = frame_mask & (frame[:, :, 1] <= s_hi)
    # print(np.sum(frame_mask))
    frame_mask = frame_mask & (frame[:, :, 2] >= v_lo)
    frame_mask = frame_mask & (frame[:, :, 2] <= v_hi)
    # print(np.sum(frame_mask))

    return frame_mask

class ColorTracker:
    def __init__(self, hsv_slices, max_pixels_change=50, name='color tracker', sliders=False):
        self.color_bounds_h = list(hsv_slices[0])
        self.color_bounds_s = list(hsv_slices[1])
        self.color_bounds_v = list(hsv_slices[2])
        self.max_pixels_change = max_pixels_change
        self.name = name
        self.sliders = sliders
        self.last_bounds = None

        self.last_frame_mask_uint8 = None
        self.vis_window_name = self.name + ' selected contour, mask, inliers'

        if self.sliders:
            self.h_lo_slider = [self.color_bounds_h[0],]
            self.h_hi_slider = [self.color_bounds_h[1],]
            self.s_lo_slider = [self.color_bounds_s[0],]
            self.s_hi_slider = [self.color_bounds_s[1],]
            self.v_lo_slider = [self.color_bounds_v[0],]
            self.v_hi_slider = [self.color_bounds_v[1],]

            h_lo_lambda = lambda val, val_return=self.h_lo_slider: on_val_return_trackerbar_int(val, val_return)
            h_hi_lambda = lambda val, val_return=self.h_hi_slider: on_val_return_trackerbar_int(val, val_return)
            s_lo_lambda = lambda val, val_return=self.s_lo_slider: on_val_return_trackerbar_int(val, val_return)
            s_hi_lambda = lambda val, val_return=self.s_hi_slider: on_val_return_trackerbar_int(val, val_return)
            v_lo_lambda = lambda val, val_return=self.v_lo_slider: on_val_return_trackerbar_int(val, val_return)
            v_hi_lambda = lambda val, val_return=self.v_hi_slider: on_val_return_trackerbar_int(val, val_return)

            cv2.namedWindow(self.vis_window_name)
            cv2.createTrackbar('h_low ', self.vis_window_name, self.h_lo_slider[0], 179, h_lo_lambda)
            cv2.createTrackbar('h_high', self.vis_window_name, self.h_hi_slider[0], 179, h_hi_lambda)
            cv2.createTrackbar('s_low ', self.vis_window_name, self.s_lo_slider[0], 255, s_lo_lambda)
            cv2.createTrackbar('s_high', self.vis_window_name, self.s_hi_slider[0], 255, s_hi_lambda)
            cv2.createTrackbar('v_low ', self.vis_window_name, self.v_lo_slider[0], 255, v_lo_lambda)
            cv2.createTrackbar('v_high', self.vis_window_name, self.v_hi_slider[0], 255, v_hi_lambda)

            self.max_pixels_slider = [self.max_pixels_change,]
            max_pixels_lambda = lambda val, val_return=self.max_pixels_slider: on_val_return_trackerbar_int(val, val_return)
            cv2.createTrackbar('max_pixels', self.vis_window_name, self.max_pixels_slider[0], 4*self.max_pixels_change, max_pixels_lambda)

    def update(self, frame, convert_to_hsv=False, visualize=False, debug=False):
        if self.sliders:
            self.color_bounds_h[0] = self.h_lo_slider[0]
            self.color_bounds_h[1] = self.h_hi_slider[0]
            self.color_bounds_s[0] = self.s_lo_slider[0]
            self.color_bounds_s[1] = self.s_hi_slider[0]
            self.color_bounds_v[0] = self.v_lo_slider[0]
            self.color_bounds_v[1] = self.v_hi_slider[0]
            self.max_pixels_change = self.max_pixels_slider[0]

        if convert_to_hsv:
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        else:
            frame_hsv = frame

        frame_mask = frame_thresh(frame_hsv,
                                  self.color_bounds_h[0],
                                  self.color_bounds_h[1],
                                  self.color_bounds_s[0],
                                  self.color_bounds_s[1],
                                  self.color_bounds_v[0],
                                  self.color_bounds_v[1])
        indices = np.argwhere(frame_mask)

        if self.last_bounds is None:
            frame_mask_uint8 = 128*frame_mask.astype(np.uint8)
            contours, hierarchy = cv2.findContours(frame_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # Find the contour with the longest perimeter and assume it is the correct contour
            largest_perim_contour = None
            for contour in contours:
                if largest_perim_contour is None or largest_perim_contour.shape[0] < contour.shape[0]:
                    largest_perim_contour = contour

            if largest_perim_contour is None:
                return None # Early return, we can not find a contour in the frame

            if self.sliders or visualize:
                cv2.drawContours(frame_mask_uint8, (largest_perim_contour,), -1, 255, 3)
                self.last_frame_mask_uint8 = frame_mask_uint8

            largest_perim_contour = largest_perim_contour.reshape((largest_perim_contour.shape[0], largest_perim_contour.shape[2]))
            x_lo = np.min(largest_perim_contour[:, 0])
            x_hi = np.max(largest_perim_contour[:, 0])
            y_lo = np.min(largest_perim_contour[:, 1])
            y_hi = np.max(largest_perim_contour[:, 1])
            self.last_bounds = (x_lo, x_hi, y_lo, y_hi)

            indices_contour = indices        [indices        [:, 1] >= x_lo, :]
            indices_contour = indices_contour[indices_contour[:, 1] <= x_hi, :]
            indices_contour = indices_contour[indices_contour[:, 0] >= y_lo, :]
            indices_contour = indices_contour[indices_contour[:, 0] <= y_hi, :]
        else:
            x_new_lo = max(self.last_bounds[0] - self.max_pixels_change, 0)
            x_new_hi = min(self.last_bounds[1] + self.max_pixels_change, frame.shape[1] - 1)
            y_new_lo = max(self.last_bounds[2] - self.max_pixels_change, 0)
            y_new_hi = min(self.last_bounds[3] + self.max_pixels_change, frame.shape[0] - 1)
            if debug: print(self.last_bounds, x_new_lo, x_new_hi, y_new_lo, y_new_hi)

            self.last_bounds = None
            indices_contour = indices[indices[:, 1] >= x_new_lo, :]
            if indices_contour.shape[0] > 0:
                indices_contour = indices_contour[indices_contour[:, 1] <= x_new_hi, :]
                if indices_contour.shape[0] > 0:
                    indices_contour = indices_contour[indices_contour[:, 0] >= y_new_lo, :]
                    if indices_contour.shape[0] > 0:
                        indices_contour = indices_contour[indices_contour[:, 0] <= y_new_hi, :]
                        if indices_contour.shape[0] > 0:
                            x_lo = np.min(indices_contour[:, 1])
                            x_hi = np.max(indices_contour[:, 1])
                            y_lo = np.min(indices_contour[:, 0])
                            y_hi = np.max(indices_contour[:, 0])
                            self.last_bounds = (x_lo, x_hi, y_lo, y_hi)
                        else:
                            if debug: print(self.name, 'fail y_new_hi')
                    else:
                        if debug: print(self.name, 'fail y_new_low')
                else:
                    if debug: print(self.name, 'fail x_new_hi')
            else:
                if debug:
                    print(self.name, 'fail x_new_lo')
                    print(indices.shape, x_new_lo, indices_contour.shape, np.min(indices[:, 0]), np.max(indices[:, 1]))

            if self.last_bounds is None:
                indices_contour = None
                if debug: print(self.name, 'bounds reset')
                # TODO run find contours instead of waiting till next frame

        if self.sliders or visualize:
            frame_mask_inliers = np.zeros_like(frame_mask)
            if indices_contour is not None:
                frame_mask_inliers[indices_contour[:, 0], indices_contour[:, 1]] = True
            frame_mask_inliers[ 0, :] = True
            frame_mask_inliers[-1, :] = True
            frame_mask_inliers[ :, 0] = True
            frame_mask_inliers[ :,-1] = True
            if self.last_frame_mask_uint8 is None:
                self.last_frame_mask_uint8 = np.zeros_like(frame_mask)
            cv2.imshow(self.vis_window_name, 255*np.hstack((self.last_frame_mask_uint8, frame_mask, frame_mask_inliers)).astype(np.uint8))

        return indices_contour
