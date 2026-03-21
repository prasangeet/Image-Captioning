import os
import tensorflow as tf
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


class Trainer:
    def __init__(self, decoder, tokenizer, lr=1e-3, alpha=1e-5, label_smoothing=0.1):
        self.decoder = decoder
        self.tokenizer = tokenizer

        lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=10000,
            alpha=alpha
        )
        self.optimizer = tf.keras.optimizers.Adam(lr_scheduler)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction="none"
        )
        self.label_smoothing = label_smoothing
        self.index_word = {v: k for k, v in tokenizer.word_index.items()}
        self.end_id = tokenizer.word_index.get("<end>")
        self.start_id = tokenizer.word_index.get("<start>")

    def loss_function(self, real, pred, ohem_ratio=0.7):
        """
        OHEM: only backprop through the hardest ohem_ratio fraction of non-padding tokens.
        """
        vocab_size = tf.cast(tf.shape(pred)[-1], tf.float32)

        # Per-token loss
        loss = self.loss_object(real, pred)  # (B, seq_len)

        # Label smoothing
        if self.label_smoothing > 0:
            smooth_loss = -tf.reduce_sum(
                tf.nn.log_softmax(pred), axis=-1
            ) / vocab_size
            loss = (1.0 - self.label_smoothing) * loss + self.label_smoothing * smooth_loss

        # Padding mask — ignore zero-padded positions
        padding_mask = tf.cast(tf.not_equal(real, 0), loss.dtype)  # (B, seq_len)
        loss *= padding_mask

        # OHEM — keep only the top (hardest) ohem_ratio of non-padding tokens
        # Flatten to select across the whole batch
        loss_flat = tf.reshape(loss, [-1])           # (B * seq_len,)
        mask_flat = tf.reshape(padding_mask, [-1])   # (B * seq_len,)

        # Total non-padding tokens
        num_valid = tf.cast(tf.reduce_sum(mask_flat), tf.int32)

        # How many hard examples to keep
        k = tf.cast(
            tf.math.ceil(tf.cast(num_valid, tf.float32) * ohem_ratio),
            tf.int32
        )
        k = tf.maximum(k, 1)  # always keep at least 1

        # Pick top-k hardest tokens across the batch
        top_k_losses, _ = tf.math.top_k(loss_flat, k=k)

        return tf.reduce_mean(top_k_losses) 

    @tf.function
    def train_step(self, features, captions):
        inp = captions[:, :-1]
        target = captions[:, 1:]
        with tf.GradientTape() as tape:
            predictions = self.decoder((inp, features), training=True)
            loss = self.loss_function(target, predictions)
        gradients = tape.gradient(loss, self.decoder.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.decoder.trainable_variables)
        )
        return loss

    @tf.function
    def val_step(self, features, captions):
        inp = captions[:, :-1]
        target = captions[:, 1:]
        predictions = self.decoder((inp, features), training=False)
        loss = self.loss_function(target, predictions)
        pred_ids = tf.argmax(predictions, axis=-1, output_type=target.dtype)
        return loss, pred_ids, target

    def evaluate_bleu(self, val_ds, sample_size=100):
        # Debug: verify token IDs are correct
        print(f"[DEBUG] start_id={self.start_id}, end_id={self.end_id}")
        print(f"[DEBUG] word at id 2: {self.index_word.get(2)}")
        print(f"[DEBUG] word at id 3: {self.index_word.get(3)}")

        references = []
        hypotheses = []
        smoother = SmoothingFunction().method1
        count = 0

        for features, captions in val_ds:
            if count >= sample_size:
                break

            batch_size = features.shape[0]
            remaining = sample_size - count
            batch_size = min(batch_size, remaining)
            features = features[:batch_size]
            captions = captions[:batch_size]

            # Batch autoregressive generation
            batch_captions = [[self.start_id] for _ in range(batch_size)]
            done = [False] * batch_size

            for _ in range(40):
                if all(done):
                    break

                max_len = max(len(c) for c in batch_captions)
                inp = tf.constant(
                    [c + [0] * (max_len - len(c)) for c in batch_captions],
                    dtype=tf.int32
                )
                predictions = self.decoder((inp, features), training=False)

                for i in range(batch_size):
                    if done[i]:
                        continue
                    pos = len(batch_captions[i]) - 1
                    predicted_id = int(tf.argmax(predictions[i, pos, :]).numpy())
                    if predicted_id == self.end_id:
                        done[i] = True
                    else:
                        batch_captions[i].append(predicted_id)

            # Decode predictions and references
            for i in range(batch_size):
                pred_tokens = [
                    self.index_word.get(t)
                    for t in batch_captions[i][1:]
                    if self.index_word.get(t) not in ("<start>", "<end>", "<unk>", None)
                ]
                ref = captions[i].numpy()
                ref_tokens = [
                    self.index_word.get(t)
                    for t in ref
                    if t != 0 and self.index_word.get(t) not in ("<start>", "<end>", None)
                ]
                if pred_tokens and ref_tokens:
                    hypotheses.append(pred_tokens)
                    references.append([ref_tokens])

            count += batch_size

        # Sample predictions for manual inspection
        print(f"\n[DEBUG] Total hypotheses collected: {len(hypotheses)}")
        print("[SAMPLE PREDICTIONS]")
        for i in range(min(5, len(hypotheses))):
            print(f"  Pred: {' '.join(hypotheses[i]) if hypotheses[i] else '[EMPTY]'}")
            print(f"  Ref:  {' '.join(references[i][0])}")
            print()

        if not hypotheses:
            print("[WARNING] No valid hypotheses — model may be predicting <end> immediately")
            return 0.0, 0.0

        bleu1 = corpus_bleu(references, hypotheses,
                            weights=(1, 0, 0, 0),
                            smoothing_function=smoother)
        bleu4 = corpus_bleu(references, hypotheses,
                            weights=(0.25, 0.25, 0.25, 0.25),
                            smoothing_function=smoother)
        return bleu1, bleu4

    def save_model(self, save_dir, epoch=None):
        os.makedirs(save_dir, exist_ok=True)
        if epoch is not None:
            save_path = os.path.join(save_dir, f"decoder_epoch_{epoch+1}.keras")
        else:
            save_path = os.path.join(save_dir, "decoder_final.keras")
        self.decoder.save(save_path)
        print(f"[INFO] Model saved to {save_path}")

    def train(self, train_ds, val_ds, epochs, save_dir="checkpoints"):
        best_bleu4 = 0.0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            total_loss = 0.0
            num_batches = 0

            # ========= TRAIN =========
            train_bar = tqdm(train_ds, desc="Training", unit="batch")
            for features, captions in train_bar:
                loss = self.train_step(features, captions)
                total_loss += loss
                num_batches += 1
                train_bar.set_postfix(loss=f"{loss:.4f}")

            train_loss = total_loss / num_batches

            # ========= VALIDATION LOSS =========
            val_loss = 0.0
            val_batches = 0

            val_bar = tqdm(val_ds, desc="Validation", unit="batch")
            for features, captions in val_bar:
                loss, _, _ = self.val_step(features, captions)
                val_loss += loss
                val_batches += 1
                val_bar.set_postfix(loss=f"{loss:.4f}")

            val_loss = val_loss / val_batches

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_loss:.4f}")

            # ========= BLEU (every 2 epochs) =========
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print("Computing BLEU scores (100 samples)...")
                bleu1, bleu4 = self.evaluate_bleu(val_ds, sample_size=100)
                print(f"BLEU-1: {bleu1:.4f} | BLEU-4: {bleu4:.4f}")

                if bleu4 > best_bleu4:
                    best_bleu4 = bleu4
                    self.save_model(save_dir, epoch=epoch)
                    print(f"[INFO] Saved at epoch {epoch+1} — best BLEU-4: {bleu4:.4f}")
