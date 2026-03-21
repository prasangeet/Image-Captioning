import tensorflow as tf

class CNNEncoder(tf.keras.Model):
    def __init__(self, embedding_dim=256):
        super(CNNEncoder, self).__init__()
        self.embedding_dim = embedding_dim

        base_model = tf.keras.applications.InceptionV3(
            include_top=False,
            weights='imagenet'
        )
        # Take output from mixed10 — the last layer before pooling
        self.cnn = tf.keras.Model(
            inputs=base_model.input,
            outputs=base_model.output
        )
        self.cnn.trainable = False
        self.fc = tf.keras.layers.Dense(embedding_dim, activation='relu')

    def call(self, inputs, training=False, mask=None):
        x = self.cnn(inputs)                                    # (B, 8, 8, 2048)
        x = tf.reshape(x, (tf.shape(x)[0], -1, x.shape[-1]))  # (B, 64, 2048)
        x = self.fc(x)                                          # (B, 64, 256)
        return x
