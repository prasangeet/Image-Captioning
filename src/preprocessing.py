import os
import random
import tensorflow as tf
import pandas as pd
import numpy as np


class PreprocessingPipeline:
    def __init__(
        self,
        base_path="Flickr8k/flickr8k",
        batch_size=32,
        max_len=40
    ):
        self.base_path = base_path
        self.batch_size = batch_size
        self.max_len = max_len

        self.img_path = os.path.join(base_path, "Images")
        self.captions_path = os.path.join(base_path, "captions.txt")

        self.image_captions = {}
        self.tokenizer = None  # Will be assigned to a Tokenizer instance after build_tokenizer is called
        self.vocab = None

    # =========================
    # LOAD DATA
    # =========================
    def load_data(self):

        df = pd.read_csv(self.captions_path)

        for _, row in df.iterrows():
            img = row["image"]
            caption = "<start> " + str(row["caption"]).lower() + " <end>"

            self.image_captions.setdefault(img, []).append(caption)

        print(f"[INFO] Loaded {len(self.image_captions)} images")

    # =========================
    # TRAIN-VAL SPLIT (IMAGE LEVEL)
    # =========================
    def train_val_split(self, val_ratio=0.1, seed=42):

        random.seed(seed)

        image_keys = list(self.image_captions.keys())
        random.shuffle(image_keys)

        split_idx = int(len(image_keys) * (1 - val_ratio))

        train_keys = image_keys[:split_idx]
        val_keys = image_keys[split_idx:]

        def build_subset(keys):
            features = []
            captions = []

            for img in keys:
                feature_path = os.path.join(
                    "features",
                    img.replace(".jpg", ".npy")
                )

                for cap in self.image_captions[img]:
                    features.append(feature_path)
                    captions.append(cap)

            return features, captions

        train_feat, train_cap = build_subset(train_keys)
        val_feat, val_cap = build_subset(val_keys)

        print(f"[INFO] Train images: {len(train_keys)}")
        print(f"[INFO] Val images: {len(val_keys)}")

        return train_feat, train_cap, val_feat, val_cap

    # =========================
    # TOKENIZER (FREQ FILTER)
    # =========================
    def build_tokenizer(self, captions, min_freq=5):

        temp_tokenizer = tf.keras.preprocessing.text.Tokenizer(
            oov_token="<unk>"
        )
        temp_tokenizer.fit_on_texts(captions)

        word_counts = temp_tokenizer.word_counts

        self.vocab = {
            word for word, count in word_counts.items()
            if count >= min_freq
        }

        print(f"[INFO] Vocab size (filtered): {len(self.vocab)}")

        tokenizer = tf.keras.preprocessing.text.Tokenizer(
            oov_token="<unk>"
        )
        tokenizer.fit_on_texts(captions)

        self.tokenizer = tokenizer

    # =========================
    # TEXT → SEQUENCES
    # =========================
    def text_to_sequences(self, captions):

        if self.tokenizer is None:
            raise ValueError("Tokenizer has not been built. Please call build_tokenizer() before text_to_sequences().")
        sequences = []

        unk_id = self.tokenizer.word_index["<unk>"]

        for cap in captions:
            tokens = cap.split()

            seq = [
                self.tokenizer.word_index.get(word, unk_id)
                if word in self.vocab
                else unk_id
                for word in tokens
            ]

            sequences.append(seq)

        sequences = tf.keras.preprocessing.sequence.pad_sequences(
            sequences,
            maxlen=self.max_len,
            padding="post"
        )

        return sequences

    # =========================
    # LOAD FEATURE (.npy)
    # =========================
    def load_feature(self, path):

        def _load(path_str):
            return np.load(path_str.decode("utf-8"))

        feature = tf.numpy_function(
            _load,
            [path],
            tf.float32
        )

        feature.set_shape((49, 256))  # IMPORTANT

        return feature

    # =========================
    # DATASET (FEATURE-BASED)
    # =========================
    def create_dataset(self, feature_paths, sequences):

        dataset = tf.data.Dataset.from_tensor_slices(
            (feature_paths, sequences)
        )

        def process(path, seq):
            feature = self.load_feature(path)
            return feature, seq

        dataset = dataset.map(
            process,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset
