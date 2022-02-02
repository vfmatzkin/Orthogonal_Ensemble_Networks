""" normalization.py

This script is part of the build_dataset.py tool for converting the OEN data.
"""

import numpy as np
import math
from scipy.stats import boxcox

def mask_filter_and_z_scores(background_value):
	def mask_filter_and_z_scores_background_set(img, mask):
		#mask = mask.astype('bool')
		#img = img.astype('float32')
		
		#filter from outside mask to nan
		img[mask == 0] = np.nan

		#z-scores
		img = (img - np.nanmean(img)) / np.nanstd(img)

		#filter from nan to the background value (-4 , -1 or another)
		img = np.where(mask, img, background_value)


		return img

	return mask_filter_and_z_scores_background_set

def z_scores_normalization(img):
	print("Normalizing with z-scores...",np.shape(img))
	img = (img - np.mean(img)) / np.std(img)
	return img

def min_max_normalization(img):
	print("Normalizing with Min-Max") # (X = X - Xmin) / (Xmax - Xmin)
	min_img = min(img.flatten())
	max_img = max(img.flatten())
	img = (img - min_img) / (max_flair - min_img)
	return img

class normalization():
	def __init__(self, method, background_value = None):
		if method == "z_scores":
			self.takes_mask = False
			self.method = z_scores_normalization

		elif method == "mask_filter_and_z_scores":
			assert background_value != None
			self.takes_mask = True
			self.method = mask_filter_and_z_scores(background_value)

		elif method == "min_max":
			self.takes_mask = False
			self.method = min_max_normalization

		else:
			raise RuntimeError("Unknown normalization method.")
