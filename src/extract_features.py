from src.preprocessing import PreprocessingPipeline
from model_classes.encoder import CNNEncoder
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import os

pipeline = PreprocessingPipeline()
pipeline.load_data()

encoder = CNNEncoder()

image_paths = list(pipeline.image_captions.keys())

save_dir = "features"
os.makedirs(save_dir, exist_ok=True)

for path in tqdm(image_paths):

    full_path = os.path.join(pipeline.img_path, path)

    img = tf.io.read_file(full_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.resnet50.preprocess_input(img)

    img = tf.expand_dims(img, 0)

    feature = encoder(img, training=False)
    feature = tf.squeeze(feature, axis=0).numpy()

    filename = path.replace(".jpg", ".npy")
    np.save(os.path.join(save_dir, filename), filename)
