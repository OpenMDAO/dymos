from __future__ import print_function, division, absolute_import

from .optimizer_based.gauss_lobatto_phase import GaussLobattoPhase
from .optimizer_based.radau_pseudospectral_phase import RadauPseudospectralPhase


_transcriptions = {'gauss-lobatto': GaussLobattoPhase,
                   'radau-ps': RadauPseudospectralPhase}


def Phase(transcription, **kwargs):
    """
    Instantiates and returns a phase of the given transcription.

    Parameters
    ----------
    transcription : str
        The type of transcription to be used by the new phase.  Valid options are:
        'gauss-lobatto' and 'radau-ps'.
    kwargs
        Keyword arguments needed for the instantiation of the given phase class.

    Returns
    -------
    phase
        The instantiated phase class.

    """
    return _transcriptions[transcription](**kwargs)
