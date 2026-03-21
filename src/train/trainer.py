from numpy import gradient
import tensorflow as tf 

class Trainer:
    def __init__(self, decoder, tokenizer, lr=1e-4):
        self.decoder = decoder
        self.tokenizer = tokenizer

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction="none"
        )
    

    ## Masked Loss 
    def loss_function(self, real, pred):
        loss = self.loss_object(real, pred)

        mask = tf.cast(tf.not_equal(real, 0), loss.dtype)
        loss *= mask

        return tf.reduce_sum(loss) / tf.reduce_sum(mask)
    
    @tf.function
    def train_step(self, features, captions):
        inp = captions[:, :-1]
        target = captions[:, 1:]

        with tf.GradientTape() as tape:
            predictions = self.decoder(
                (inp, features),
                training=True
            )

            loss = self.loss_function(target, predictions)

        gradients= tape.gradient(
            loss,
            self.decoder.trainable_variables
        )

        self.optimizer.apply_gradients(
            zip(gradients, self.decoder.trainable_variables)
        )

        return loss

    def 
