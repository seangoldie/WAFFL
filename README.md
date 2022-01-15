# WAFFL

**Waveform Audio Fundamental Frequency Learner**
**Authors: Sean Goldie, Gregor McWilliam, Jack Tipper**

A novel and efficient machine learning method for estimating the fundamental frequency of monophonic vocal recordings. This work was conducted for our final project in the graduate-level Music Information Retrieval class at NYU Steinhardt.

## Installation

Currently, the best way to install and use the WAFFL package is to clone this repository and move the folder `/WAFFL/waffl` to the desired location. This should be the location your package manager installs new packages, and your Python installation looks for libraries.

### Usage

The model can be used to predict pitch contours for vocal recordings using the `.predict_from_array` and `.predict_from_path` methods. The returned Numpy arrays could be used for a variety of MIR tasks, including constructing feature sets for training other machine learning models. There are four output formats available: `"raw_pitch"`, `"nearest_note"`, `"note_labels"`, and `"midi"`. The return for each format is the following: 
* `"raw_pitch"`: array of frequency values in Hz, with float32 precision. Unvoiced frames will be denoted by `0.0`
* `"nearest_note"`: array of frequency values rounded to the nearest Hz value associated with a musical note
* `"note_labels"`: list of strings representing the note name and register at each frame, e.g. `"C3"` or `"G#3/Ab3"`. Unvoiced frames will be denoted with the string `"--"`
* `"midi"`: array of MIDI note values (0-127)

If no output format string is passed to the prediction method, the default is `"raw_pitch"`.

**Example**

Import the model into your Python project:
```
    >>> import waffl
    >>> model = waffl.WAFFL()
```
Make predictions using its predict methods:
```
    >>> model.predict_from_array(audio_array, fs, hop_length=320, nfft=4096, win_length=2048)
    >>> model.predict_from_path(audio_file_pathway, hop_length=320, nfft=4096, win_length=2048)
```
That's really all there is to it! Call `help()` on a `waffl.WAFFL()` object or either `.predict_from` methods for the full documentation on their usage.

#### More Info
If you'd like to read the whitepaper we wrote about WAFFL, you can find that [here](https://gregormcw.com/waffl/WAFFL_paper_release.pdf).