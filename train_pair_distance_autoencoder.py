import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class Autoencoder(pl.LightningModule):
    def __init__(self, input_dim, latent_dim=2, hidden_dims=[128, 64, 32]):
        super().__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                ]
            )
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                ]
            )
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.criterion = nn.MSELoss()

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def training_step(self, batch, batch_idx):
        x, _ = batch
        reconstructed = self(x)
        loss = self.criterion(reconstructed, x)
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, _ = batch
        reconstructed = self(x)
        loss = self.criterion(reconstructed, x)
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        reconstructed = self(x)
        loss = self.criterion(reconstructed, x)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


def main():
    # Load your data
    # Replace this with your actual data loading
    data = pd.read_csv(
        "/home/pbuslaev/projects/personal/gpcr/dataset.csv"
    ).values  # Shape: (n_samples, n_features)

    # Convert to PyTorch tensors
    data_tensor = torch.FloatTensor(data)
    dataset = TensorDataset(data_tensor, data_tensor)

    # Split into train and test (90% train, 10% test)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=4
    )

    # Initialize model
    input_dim = data.shape[1]
    model = Autoencoder(input_dim=input_dim, latent_dim=2)

    # Setup CSV logger to save metrics
    csv_logger = pl.loggers.CSVLogger("logs", name="autoencoder")

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
        enable_checkpointing=True,
        logger=csv_logger,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                filename="autoencoder-{epoch:02d}-{val_loss:.4f}",
            ),
            pl.callbacks.EarlyStopping(
                monitor="val_loss", patience=20, mode="min"
            ),
        ],
    )

    # Train the model
    trainer.fit(model, train_loader, test_loader)

    # Test the model
    trainer.test(model, test_loader)

    # Load metrics and plot
    import matplotlib.pyplot as plt

    metrics = pd.read_csv(f"{csv_logger.log_dir}/metrics.csv")

    # Extract train and validation losses
    train_loss = metrics[["epoch", "train_loss"]].dropna()
    val_loss = metrics[["epoch", "val_loss"]].dropna()

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        train_loss["epoch"],
        train_loss["train_loss"],
        label="Train Loss",
        marker="o",
    )
    plt.plot(
        val_loss["epoch"],
        val_loss["val_loss"],
        label="Validation Loss",
        marker="s",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_plot.png", dpi=300, bbox_inches="tight")

    print("Training complete!")
    print(f"Metrics saved to: {csv_logger.log_dir}/metrics.csv")
    print("Plot saved to: training_plot.png")


if __name__ == "__main__":
    main()
