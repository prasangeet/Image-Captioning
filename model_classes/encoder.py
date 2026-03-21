import tensorflow as tf 

class CNNEncoder(tf.keras.Model):
    def __init__(self, embedding_dim=256) -> None:
        super(CNNEncoder, self).__init__()
        self.embedding_dim = embedding_dim

        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
        )

        self.cnn = tf.keras.Model(
            inputs = base_model.input,
            outputs=base_model.output
        )

        self.cnn.trainable = False

        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, inputs, training=False, mask=None):
        x = self.cnn(inputs) # (batch, 7, 7, 2048)

        # flatten spacial dims -> (batch, 49, 2048)
        x = tf.reshape(x, (x.shape[0], -1, x.shape[-1]))

        # (batch, 49, 256)
        x = self.fc(x)

        return x

