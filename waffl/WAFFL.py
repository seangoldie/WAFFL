import librosa
import os
import numpy as np
import pickle

class WAFFL():
    """ WAFFL 0.1 - Waveform Audio Fundamental Frequency Learner

        Authors: Sean Goldie, Gregor McWilliam, and Jack Tipper
        This project was originally completed as the final group project for the graduate MIR class at NYU, Fall 2021.
        It is uses a lightweight pre-trained machine learning model to produce pitch contour predictions for hummed/sung vocals.

        Usage of this class:

        Import the model into your Python project:
            >>> import WAFFL
            >>> model = waffl.WAFFL()

        Make predictions using its predict methods:
            >>> model.predict_from_array(audio_array, fs, hop_length=320, nfft=4096, win_length=2048)
            >>> model.predict_from_path(audio_file_pathway, hop_length=320, nfft=4096, win_length=2048)

    """

    def __init__(self):
        """ Return a new WAFFL() model object.

        Parameters:
        -----------
        None

        Returns:
        --------
        A new WAFFL() model object
        
        """
        self.model_path = __file__.split('waffl')[0] + "/waffl/model.pkl"
        self.fs = 16000

        with open(self.model_path, 'rb') as model:
            self.model = pickle.load(model)
    
    def _predict(
        self, 
        x,
        hop_length, 
        nfft, 
        win_length):
        """ Generalized prediction loop
        """

        try: # If it is stereo, make it mono
            _ = x.shape[1]
            x = librosa.to_mono(x)
        except: # It is already mono
            pass

        num_samples_to_chop = x.shape[0] % hop_length

        x = x[:x.shape[0] - num_samples_to_chop]

        x = np.append(
            np.zeros(int(hop_length/2)),
            x
        )

        S = librosa.feature.melspectrogram(
            y=x,
            n_fft=nfft, 
            hop_length=hop_length, 
            win_length=win_length,
            sr=self.fs
        )

        S = S[:, 1:S.shape[1]-1]

        predictions = np.empty(S.shape[1])

        for i in range(S.shape[1]):
            predictions[i] = self.model.predict(S[:,i].reshape(1, -1))
        
        predictions[ predictions < 20 ] = 0 # Rectify below human hearing threshold

        return predictions


    def _freq_to_note(self, freq_array):
        """ Convert array of frequencies to nearest chromatic note values
        Used to align frequencies to nearest note
        """

        # Small dict of note names to facilitate more extensive dict
        num_to_note = {
        0: "C", 1: "C#/Db", 2: "D", 3: "D#/Eb", 4: "E", 5: "F", 6: "F#/Gb", 7: "G", 8: "G#/Ab", 9: "A", 10: "A#/Bb", 11: "B"}

        # Create empty dict
        note_dict = {}

        # Select how many octaves and notes should be represented by a note label
        num_octaves = 10
        num_notes = 12

        # Create dict mapping MIDI value to pitch
        for n in range(num_octaves * num_notes):
            note_dict[n] = num_to_note[n%num_notes] + str(n // num_notes -1)
        
        # Initialize several variables
        a4_midi_val = 69
        a4_freq_val = 440.0

        # Create empty list
        note_labels = []
        
        # Convert frequency to cents
        with np.errstate(divide='ignore'):
            cents = np.round(1200 * np.log2(freq_array / a4_freq_val) + 100 * a4_midi_val)

        cents[np.isneginf(cents)] = 0
        
        # Calculate remainder
        rem = np.mod(cents, 100)

        # Round every value in array to nearest 100th cent
        cents[rem >= (100 - rem)] = 1.0 + cents[rem >= (100 - rem)] // 100
        cents[rem < (100 - rem)] = cents[rem < (100 - rem)] // 100

        # For each note value, convert to note label and append to list
        for n in range(len(cents)):
            if cents[n] < 12:
                note_labels.append("--")
            else:
                note_labels.append(note_dict[cents[n]])

        # Return array of numerical note values and list of note labels
        return cents, note_labels
        

    def _note_to_freq(self, cent_array):
        """ Convert array of chromatic note values to frequencies.
        Can be used to generate array for Auto-Tune effect.

        """

        freq_array = 440 * 2**((cent_array - 69)/12)

        return freq_array


    def _round_raw_freq(self, freq_array):
        """ Simple function that takes raw frequency ref data and returns frequency 
        data rounded to nearest chromatic pitch.

        """

        return self._note_to_freq(self._freq_to_note(freq_array)[0])

    def predict_from_array(
        self,
        audio_array,
        fs,
        output_format="raw_pitch",
        hop_length=320, 
        nfft=4096, 
        win_length=2048
        ):
        """ Predict pitch contours from an array of audio, such as one returned by Librosa.load() or soundfile.read().

        Parameters:
        -----------
        audio_array: np.array
            The array of audio samples.
        
        fs: int
            The sample rate for the audio array.

        output_format: string
            The format of the output data. One of ``raw_pitch``, ``nearest_note``, ``note_labels``, or ``midi``.
            The return for each format is the following: 

                    raw_pitch: array of frequency values in Hz, with float32 precision. 
                            Unvoiced frames will be denoted by 0.0

                    nearest_note: array of frequency values rounded to the nearest Hz value associated with a musical note

                    note_labels: list of strings representing the note name and register at each frame, e.g. "C3" or "G#3/Ab3". 
                            Unvoiced frames will be denoted with the string "--"
                            
                    midi: array of MIDI note values (0-127)

        hop_length: int
            Hop size for the spectrogram data. Default 320. 
            Hop lengths other than multiples of 320 are experimental.

        nfft: int
            FFT size for the spectrogram data

        Returns:
        --------
        predictions: np.array or list
            Pitch predictions in Hz, ordered by bins of hop_length size.
            Number of bins = ceiling(length of the file / hop_length).

        Example:
        ---------
        Predict an audio file array as loaded from Librosa:
            >>> import librosa
            >>> from waffl import WAFFL
            >>> model = WAFFL()
            >>> x, fs = librosa.load(path, sr=None)
            >>> p = model.predict_from_array(x, fs, "raw")

        """

        x = librosa.resample(audio_array, fs, self.fs)

        predictions = self._predict(x, hop_length=hop_length, nfft=nfft, win_length=win_length)

        if output_format == "raw_pitch":
            return predictions

        elif output_format == "nearest_note":
            p = self._round_raw_freq(predictions)
            p[p < 20] = 0 # Rectify less than human hearing - after choosing closest whole pitch
            return p
        
        elif output_format == "note_labels":
            return self._freq_to_note(predictions)[1]
        
        elif output_format == "midi":
            return self._freq_to_note(predictions)[0]
        
        else:
            # Error catch
            raise (Exception("Invalid argument for output_format. Should be one of `raw_pitch`, `nearest_note`, `note_labels`, or `midi`."))


    def predict_from_path(
        self,
        audio_file_pathway,
        output_format="raw_pitch",
        hop_length=320, 
        nfft=4096, 
        win_length=2048,
        ):
        """ Predict pitch contours from a pathway on disk.

        Parameters:
        -----------
        audio_file_pathway: string
            Absolute pathway to the audio file.

        output_format: string
            The format of the output data. One of ``raw_pitch``, ``nearest_note``, ``note_labels``, or ``midi``.
            The return for each format is the following: 

                    raw_pitch: array of frequency values in Hz, with float32 precision. 
                            Unvoiced frames will be denoted by 0.0

                    nearest_note: array of frequency values rounded to the nearest Hz value associated with a musical note

                    note_labels: list of strings representing the note name and register at each frame, e.g. "C3" or "G#3/Ab3". 
                            Unvoiced frames will be denoted with the string "--"
                            
                    midi: array of MIDI note values (0-127)

        hop_length: int
            Hop size for the spectrogram data. Default 320. 
            Hop lengths other than multiples of 320 are experimental.

        nfft: int
            FFT size for the spectrogram data

        Returns:
        --------
        predictions: np.array
            Pitch predictions in Hz, ordered by bins of hop_length size. 
            Number of bins = ceiling(length of the file / hop_length).
        
        Example:
        --------
        Predict an audio file from its path:
            >>> from waffl import WAFFL
            >>> model = WAFFL()
            >>> path = "some_audio_file.wav"
            >>> p = model.predict_from_path(path, "raw_pitch")

        """
        
        x, _ = librosa.load(audio_file_pathway, sr=self.fs, mono=True)

        predictions = self._predict(x, hop_length=hop_length, nfft=nfft, win_length=win_length)

        if output_format == "raw_pitch":
            return predictions

        elif output_format == "nearest_note":
            p = self._round_raw_freq(predictions)
            p[p < 20] = 0 # Rectify less than human hearing - after choosing closest whole pitch
            return p
        
        elif output_format == "note_labels":
            return self._freq_to_note(predictions)[1]
        
        elif output_format == "midi":
            return self._freq_to_note(predictions)[0]
        
        else:
            # Error catch
            raise (Exception("Invalid argument for output_format. Should be one of `raw_pitch`, `nearest_note`, `note_labels`, or `midi`."))


    