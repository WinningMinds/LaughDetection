"""
VGGish embedder
===============
"""
import os

import numpy as np
import resampy
import soundfile as sf
import tensorflow as tf

from . import mel, params, slim
from .postprocess import Postprocessor


class VGGishEmbedder:  # pylint: disable=too-many-instance-attributes
    """VGGish Embedding class.

    Used to convert an audio file to embeddings.

    Parameters
    ----------
    sample_rate
        The audio sampling rate.
    window_length
        The length of the window in seconds.
    step_size
        Advances (in seconds) between windows.
    log_offset
        Offset to add to log functions so that we don't get -inf.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        window_length: float = 0.025,
        step_size: float = 0.010,
        log_offset: float = 0.01,
    ) -> None:
        self.sample_rate = sample_rate
        self.window_length = window_length
        self.step_size = step_size
        self.log_offset = log_offset

        directory = os.path.dirname(os.path.realpath(__file__))
        self.pca_params = os.path.join(directory, "pca_params.npz")
        self.vgg_checkpoint = os.path.join(directory, "model.ckpt")

        self.pproc = Postprocessor(self.pca_params)

        self.graph = tf.Graph()
        self.sess = tf.Session()

        slim.define_vggish_slim(training=False)
        slim.load_vggish_slim_checkpoint(self.sess, self.vgg_checkpoint)

        self.features_tensor = self.sess.graph.get_tensor_by_name(params.INPUT_TENSOR_NAME)
        self.embedding_tensor = self.sess.graph.get_tensor_by_name(params.OUTPUT_TENSOR_NAME)

    def wavform_to_examples(self, data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Convert data to examples VGGish.

        Parameters
        ----------
        data
            One dimension for mono or multiple for multiple channels.
        sample_rate
            Sample rate of the data.
        
        Returns
        -------
        :class:`~numpy.ndarray`
            3-D array of shape ``(num_examples, num_frames, num_bands)`` which
            represents a sequence of examples, each of which contains a patch
            of log mel spectrogram, covering ``num_frames`` frames of audio and
            ``num_bands`` mel frequency bands, where the frame length is
            :ref:`~VGGishEmbedder.step_size`.
        """
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        if sample_rate != self.sample_rate:
            data = resampy.resample(data, sample_rate, self.sample_rate)

        log_mel = mel.log_mel_spectrogram(
            data,
            sample_rate=self.sample_rate,
            log_offset=self.log_offset,
            window_length=self.window_length,
            step_size=self.step_size,
            num_mel_bands=params.NUM_MEL_BINS,
            lower_edge_hertz=params.MEL_MIN_HZ,
            upper_edge_hertz=params.MEL_MAX_HZ,
        )

        features_sample_rate = 1.0 / self.step_size
        example_window_length = int(round(params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
        example_step_size = int(round(params.EXAMPLE_STEP_SECONDS * features_sample_rate))

        return mel.frame(log_mel, window_length=example_window_length, step_size=example_step_size)

    def wavfile_to_examples(self, wav_file: str) -> np.ndarray:
        """Convert wav file to examples for VGGish.

        Parameters
        ----------
        wav_file
            The wav file to process.

        Returns
        -------
        :class:`~numpy.ndarray`
            The examples for VGGish.

        See also
        --------
        :meth:`~VGGishEmbedder.wavform_to_examples`
        """
        data, sample_rate = sf.read(wav_file, dtype="int16")
        samples = data / 32768.0
        return self.wavform_to_examples(samples, sample_rate)

    def convert_audio_to_embedding(self, wav_file: str) -> np.ndarray:
        """Convert audio file to VGGish embeddings.

        Parameters
        ----------
        wav_file
            The file to process and convert.

        Returns
        -------
        :class:`~numpy.ndarray`
            The VGGish embeddings.
        """
        batch = self.wavfile_to_examples(wav_file)
        [embedding_batch] = self.sess.run(
            [self.embedding_tensor], feed_dict={self.features_tensor: batch}
        )
        return self.pproc.postprocess(embedding_batch).tolist()
