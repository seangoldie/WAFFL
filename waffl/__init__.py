from . import WAFFL
from .WAFFL import *
""" WAVFL 0.1 - Waveform Audio Fundamental Frequency Learner

		Authors: Sean Goldie, Gregor McWilliam, and Jack Tipper
        This project was originally completed as the final group project for the graduate MIR class at NYU, Fall 2021.
        It is uses a lightweight pre-trained machine learning model to produce pitch contour predictions for hummed/sung vocals.

    Usage of this package:

    Import the model into your Python project:
        >>> import waffl
        >>> model = wavfl.WAFFL()

    Make predictions using its predict methods:
        >>> model.predict_from_array(audio_array, fs, hop_length=320, nfft=4096, win_length=2048)
        >>> model.predict_from_path(audio_file_pathway, hop_length=320, nfft=4096, win_length=2048)

"""