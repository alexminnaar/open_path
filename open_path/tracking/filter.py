

class Filter(object):
    """Abstract filter model class.  A filter must implement a predict and update step."""

    def predict(self):
        """Abstract predict method.  Predicts next time step based on model."""
        pass

    def update(self, z):
        """Abstract update method.  Updates model based on new observation."""
        pass

    def filter(self):
        """Abstract filter method.  Should iteratively call predict and update methods."""