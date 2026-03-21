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

    def loss_function(self, real, pred):
        """
        real: (B, seq_len)
        pred: (B, seq_len, vocab_size)
        """

        loss = self.loss_object(real, pred)

        mask = tf.cast(tf.not_equal(real, 0), loss.dtype)
        loss *= mask

        return tf.reduce_sum(loss) / tf.reduce_sum(mask)


    @tf.function
    def train_step(self, features, captions):

        # Teacher forcing
        inp = captions[:, :-1]
        target = captions[:, 1:]

        with tf.GradientTape() as tape:

            predictions = self.decoder(
                (inp, features),
                training=True
            )

            loss = self.loss_function(target, predictions)

        gradients = tape.gradient(
            loss,
            self.decoder.trainable_variables
        )

        self.optimizer.apply_gradients(
            zip(gradients, self.decoder.trainable_variables)
        )

        return loss

    # =========================
    # VALIDATION STEP
    # =========================
    @tf.function
    def val_step(self, features, captions):

        inp = captions[:, :-1]
        target = captions[:, 1:]

        predictions = self.decoder(
            (inp, features),
            training=False
        )

        loss = self.loss_function(target, predictions)

        return loss

    # =========================
    # TRAIN LOOP
    # =========================
    def train(self, train_ds, val_ds, epochs):

        for epoch in range(epochs):

            print(f"\nEpoch {epoch+1}")

            total_loss = 0.0
            num_batches = 0

            # ========= TRAIN =========
            for batch, (features, captions) in enumerate(train_ds):

                loss = self.train_step(features, captions)

                total_loss += loss
                num_batches += 1

                if batch % 50 == 0:
                    print(f"Batch {batch}, Loss: {loss:.4f}")

            train_loss = total_loss / num_batches

            # ========= VALIDATION =========
            val_loss = 0.0
            val_batches = 0

            for features, captions in val_ds:
                loss = self.val_step(features, captions)
                val_loss += loss
                val_batches += 1

            val_loss = val_loss / val_batches

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
