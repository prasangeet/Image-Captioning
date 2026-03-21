import random
import pandas as pd
import tensorflow as tf
from model_classes.encoder import CNNEncoder
from model_classes.decoder import TransformerDecoder, TransformerDecoderLayer
from src.preprocessing import PreprocessingPipeline

random.seed(42)

def load_and_preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.resnet50.preprocess_input(img)
    img = tf.expand_dims(img, 0)
    return img


def generate_caption(image_path, encoder, decoder, tokenizer, max_len=40, temperature=0.7):
    img = load_and_preprocess_image(image_path)
    features = encoder(img, training=False)  # (1, 49, 256)

    start_id = tokenizer.word_index.get("<start>")
    end_id = tokenizer.word_index.get("<end>")

    assert start_id is not None, f"<start> missing. Special tokens found: {[w for w in tokenizer.word_index if w.startswith('<')]}"
    assert end_id is not None, "<end> missing from tokenizer vocab."

    index_word = {v: k for k, v in tokenizer.word_index.items()}
    caption = [start_id]

    for _ in range(max_len):
        inp = tf.constant([caption], dtype=tf.int32)
        predictions = decoder((inp, features), training=False)
        last_logits = predictions[:, -1, :] / temperature

        predicted_id = int(tf.squeeze(
            tf.random.categorical(last_logits, num_samples=1), axis=-1
        ).numpy()[0])

        if predicted_id == end_id:
            break

        caption.append(predicted_id)

    # Filter out special tokens from output
    special = {"<start>", "<end>", "<unk>"}
    words = [
        index_word.get(i, "<unk>")
        for i in caption[1:]
        if index_word.get(i) not in special
    ]
    return " ".join(words)


def main():
    pipeline = PreprocessingPipeline()
    pipeline.load_data()
    _, train_cap, _, _ = pipeline.train_val_split()
    pipeline.build_tokenizer(train_cap)

    encoder = CNNEncoder(embedding_dim=256)
    encoder(tf.zeros((1, 224, 224, 3)), training=False)

    decoder = tf.keras.models.load_model(
        "checkpoints/decoder_epoch_6.keras",
        custom_objects={
            "TransformerDecoder": TransformerDecoder,
            "TransformerDecoderLayer": TransformerDecoderLayer
        }
    )

    captions_df = pd.read_csv("Flickr8k/flickr8k/captions.txt")
    img_name = random.choice(captions_df['image'].unique())
    image_path = f"Flickr8k/flickr8k/Images/{img_name}"

    caption = generate_caption(image_path, encoder, decoder, pipeline.tokenizer)

    # Also print ground truth for comparison
    ground_truth = captions_df[captions_df['image'] == img_name]['caption'].tolist()
    print(f"\nImage: {img_name}")
    print(f"Generated : {caption}")
    print(f"Ground Truth:")
    for gt in ground_truth:
        print(f"  - {gt}")


if __name__ == "__main__":
    main()
