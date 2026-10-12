import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2


@tf.keras.utils.register_keras_serializable()
class ErrorRegressor(tf.keras.Model):
    """MobileNetV2 come feature extractor, testa di regressione su scalare."""

    def __init__(
        self,
        input_shape=(224, 224, 3),
        freeze_backbone=True,
        input_range="none",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_shape_custom = tuple(input_shape)
        self.freeze_backbone = freeze_backbone
        self.input_range = input_range

        self.backbone = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
            pooling="avg",
        )

        # Se freeze_backbone=True, MobileNetV2 non viene addestrata
        # nella prima fase.
        self.backbone.trainable = not freeze_backbone

        self.dropout = tf.keras.layers.Dropout(0.3)
        self.dense = tf.keras.layers.Dense(64, activation="relu")

        # Un solo output lineare: regressione dell'Error.
        # dtype=float32 evita problemi di precisione con mixed_float16.
        self.out = tf.keras.layers.Dense(
            1,
            activation=None,
            dtype="float32",
        )

    def _scale(self, x):
        """
        Porta l'input nel range richiesto da MobileNetV2: [-1, 1].

        Nel tuo caso standard:
        ut.preprocess() produce gia' [-1, 1],
        quindi input_range="none" restituisce x senza modifiche.
        """
        if self.input_range == "0_1":
            return x * 2.0 - 1.0

        if self.input_range == "0_255":
            return x / 127.5 - 1.0

        if self.input_range == "none":
            return x

        raise ValueError(
            f"input_range non valido: {self.input_range}. "
            "Usa 'none', '0_1' oppure '0_255'."
        )

    def call(self, inputs, training=False):
        x = self._scale(inputs)
        x = self.backbone(x, training=training)
        x = self.dropout(x, training=training)
        x = self.dense(x)
        return self.out(x)

    def unfreeze_top(self, n_layers=30):
        """Sblocca gli ultimi n_layers del backbone per il fine tuning."""
        self.backbone.trainable = True

        for layer in self.backbone.layers[:-n_layers]:
            layer.trainable = False

        # I Batch Normalization rimangono congelati nel fine tuning.
        for layer in self.backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    def get_config(self):
        """
        Permette a Keras di salvare e ricaricare questa classe custom
        in modo affidabile.
        """
        config = super().get_config()

        config.update({
            "input_shape": self.input_shape_custom,
            "freeze_backbone": self.freeze_backbone,
            "input_range": self.input_range,
        })

        return config
