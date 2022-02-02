""" dataset.py

This script is part of the build_dataset.py tool for converting the OEN data.
"""

import math
from pdb import set_trace as st

class Dataset:
	def __init__(self, name = None, n_images=math.inf, origin_directory=None, patches_directory=None):
		assert name == "Utrecht" or name == "Amsterdam" or name == "Singapore"  or name == "miccaibrats"
		assert origin_directory != None or patches_directory != None

		self.name = name
		self.n_images = n_images
		self.origin_directory = origin_directory
		self.patches_directory = patches_directory

		
