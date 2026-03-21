from src.preprocessing import PreprocessingPipeline
from model_classes.decoder import TransformerDecoder
from src.train.trainer import Trainer


def main():
    # 1. Initialize PreprocessingPipeline
    pipeline = PreprocessingPipeline()

    # 2. Load data
    pipeline.load_data()
    train_feat, train_cap, val_feat, val_cap = pipeline.train_val_split()

    # 3. Build tokenizer on train only, then encode both splits
    pipeline.build_tokenizer(train_cap)
    if pipeline.tokenizer is None:
        raise ValueError("Tokenizer was not built. Please check build_tokenizer().")
    train_seqs = pipeline.text_to_sequences(train_cap)
    val_seqs = pipeline.text_to_sequences(val_cap)

    # 4. Create datasets
    train_ds = pipeline.create_dataset(train_feat, train_seqs)
    val_ds = pipeline.create_dataset(val_feat, val_seqs)

    # 5. Hyperparameters
    vocab_size = len(pipeline.tokenizer.word_index) + 1
    max_len = pipeline.max_len
    d_model = 256
    num_layers = 2
    num_heads = 4
    dff = 512
    dropout = 0.1
    lr = 3e-4
    alpha_lr_schedule = 1e-6
    label_smoothing = 0.0

    # 6. Initialize decoder
    decoder = TransformerDecoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        vocab_size=vocab_size,
        max_len=max_len,
        dropout=dropout
    )

    # 7. Initialize Trainer and train
    trainer = Trainer(decoder, pipeline.tokenizer, lr, alpha_lr_schedule, label_smoothing)
    trainer.train(train_ds, val_ds, epochs=10, save_dir="checkpoints")


if __name__ == "__main__":
    main()
