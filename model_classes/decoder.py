import tensorflow as tf
import numpy as np

## Using positional encodings
def positional_encoding(position, d_model):
    angles = np.arange(position)[:, np.newaxis] / np.power(
        10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / d_model
    )

    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])

    return tf.cast(angles[np.newaxis, ...], dtype=tf.float32)

## Decoder Layer
class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout = 0.1):
        super().__init__()

        self.self_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model
        )

        self.cross_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model
        )

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])

        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.dropout3 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=False, mask=None):

        x, enc_output, look_ahead_mask = inputs

        # 1. Masked self-attention
        attn1 = self.self_attn(
            query=x,
            key=x,
            value=x,
            attention_mask=look_ahead_mask
        )
        if isinstance(attn1, tuple):
            attn1 = attn1[0]

        attn1 = self.dropout1(attn1, training=training)
        out1 = self.norm1(x + attn1)

        # 2. Cross-attention
        attn2 = self.cross_attn(
            query=out1,
            key=enc_output,
            value=enc_output
        )
        if isinstance(attn2, tuple):
            attn2 = attn2[0]

        attn2 = self.dropout2(attn2, training=training)
        out2 = self.norm2(out1 + attn2)

        # 3. Feed forward
        ffn_out = self.ffn(out2)
        ffn_out = self.dropout3(ffn_out, training=training)
        out3 = self.norm3(out2 + ffn_out)

        return out3
        

class TransformerDecoder(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, max_len, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)

        self.pos_encoding = positional_encoding(max_len, d_model)

        self.dec_layers = [
            TransformerDecoderLayer(d_model, num_heads, dff, dropout)
            for _ in range(num_layers)
        ]
        
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq, seq)

    def call(self, inputs, training=False, mask=None):
        """
        inputs = (caption_tokens, encoder_output)
        caption_tokens: (B, seq_len)
        encoder_output: (B, 49, 256)
        """

        x, enc_output = inputs

        seq_len = tf.shape(x)[1]

        x = self.embedding(x)  # (B, seq_len, d_model)

        # 2. Scale
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # 3. Add positional encoding
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=bool(training))

        # 4. Mask
        look_ahead_mask = self.create_look_ahead_mask(seq_len)
        look_ahead_mask = look_ahead_mask[tf.newaxis, tf.newaxis, :, :]

        # 5. Pass through decoder layers
        for layer in self.dec_layers:
            x = layer(
                (x, enc_output, look_ahead_mask),
                training=bool(training)
            )

        # 6. Final projection
        logits = self.final_layer(x)

        return logits
