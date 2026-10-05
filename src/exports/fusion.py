"""Load the frozen eleven-token ensemble. No identity or diagnosis input."""

from ._integrity import checkpoint
from ._fusion import GalleryFusionClassifier


class FrozenFusion:
    def __init__(self):
        self._classifier = GalleryFusionClassifier.load(checkpoint("fusion"))

    def predict_proba(self, face, typing, batch_size=256):
        """face[N,42], typing[N,10,128] -> [N,2] control/impaired probabilities.

        Input vectors must be unstandardized. Frozen scaling is applied here.
        Ten embeddings must come from distinct sequences belonging to one user.
        """
        return self._classifier.predict_proba(face, typing, batch_size=batch_size)
