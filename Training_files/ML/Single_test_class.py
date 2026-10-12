import numpy as np
import tensorflow as tf

import Utils as ut
from NN_Class import ErrorRegressor


class ErrorPredictor:
    def __init__(self, model_path, stats_path):

        # compile=False: durante l'inferenza non servono optimizer,
        # loss e metriche del training.
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "ErrorRegressor": ErrorRegressor
            },
            compile=False,
        )

        stats = np.load(stats_path)
        self.mu_y = float(stats["mu"])
        self.sigma_y = float(stats["sigma"])

    def predict(self, image):
        # Deve essere IDENTICA alla funzione usata nel training.
        processed = ut.preprocess(image)

        # (224, 224, 3) -> (1, 224, 224, 3)
        x = tf.convert_to_tensor(processed, dtype=tf.float32)
        x = tf.expand_dims(x, axis=0)

        # Output del modello: target standardizzato
        prediction_normalized = self.model.predict(x, verbose=0)[0, 0]

        # Denormalizzazione nella scala originale di Error
        prediction = (
            prediction_normalized * self.sigma_y
            + self.mu_y
        )

        return float(prediction)

    def evaluate_one(self, image, real_error):
        predicted_error = self.predict(image)
        absolute_error = abs(predicted_error - real_error)

        return {
            "real_error": float(real_error),
            "predicted_error": predicted_error,
            "absolute_error": float(absolute_error),
        }
    