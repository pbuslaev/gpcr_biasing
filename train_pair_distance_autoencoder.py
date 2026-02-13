import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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


def train_and_evaluate_model(
    train_data, val_data, test_data, model_name="autoencoder", normalize=False
):
    """Train and evaluate an autoencoder model."""

    # Normalize data if requested
    if normalize:
        train_mean = train_data.mean(dim=0, keepdim=True)
        train_std = train_data.std(dim=0, keepdim=True)
        train_std[train_std == 0] = 1  # Avoid division by zero

        train_data_normalized = (train_data - train_mean) / train_std
        val_data_normalized = (val_data - train_mean) / train_std
        test_data_normalized = (test_data - train_mean) / train_std
    else:
        train_data_normalized = train_data
        val_data_normalized = val_data
        test_data_normalized = test_data

    # Create datasets
    train_dataset = TensorDataset(train_data_normalized, train_data_normalized)
    val_dataset = TensorDataset(val_data_normalized, val_data_normalized)
    test_dataset = TensorDataset(test_data_normalized, test_data_normalized)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=4
    )

    # Initialize model
    input_dim = train_data_normalized.shape[1]
    model = Autoencoder(input_dim=input_dim, latent_dim=2)

    # Setup CSV logger to save metrics
    csv_logger = pl.loggers.CSVLogger("logs", name=model_name)

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
                filename=f"{model_name}-{{epoch:02d}}-{{val_loss:.4f}}",
            ),
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                mode="min",
            ),
        ],
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Test the model
    trainer.test(model, test_loader)

    # Get latent representations
    model.eval()
    with torch.no_grad():
        train_latent = model.encoder(train_data_normalized)
        val_latent = model.encoder(val_data_normalized)
        test_latent = model.encoder(test_data_normalized)

    # Calculate baseline loss (on normalized data)
    baseline_mean = train_data_normalized.mean(dim=0, keepdim=True)
    baseline_train_loss = nn.MSELoss()(
        train_data_normalized, baseline_mean.expand_as(train_data_normalized)
    ).item()
    baseline_val_loss = nn.MSELoss()(
        val_data_normalized, baseline_mean.expand_as(val_data_normalized)
    ).item()
    baseline_test_loss = nn.MSELoss()(
        test_data_normalized, baseline_mean.expand_as(test_data_normalized)
    ).item()

    return {
        "model": model,
        "csv_logger": csv_logger,
        "train_latent": train_latent,
        "val_latent": val_latent,
        "test_latent": test_latent,
        "baseline_train_loss": baseline_train_loss,
        "baseline_val_loss": baseline_val_loss,
        "baseline_test_loss": baseline_test_loss,
    }


def main():
    # Load your data
    df = pd.read_csv("/home/pbuslaev/projects/personal/gpcr/dataset.csv")

    # Get unique idx values
    unique_idx = df["idx"].unique()

    # Randomly select one idx for validation and one for test
    np.random.seed(42)
    selected_idx = np.random.choice(unique_idx, size=2, replace=False)
    val_idx = selected_idx[0]
    test_idx = selected_idx[1]

    # Split data based on idx
    val_df = df[df["idx"] == val_idx]
    test_df = df[df["idx"] == test_idx]
    train_df = df[~df["idx"].isin([val_idx, test_idx])]

    # Remove idx column and convert to tensors
    val_data = torch.FloatTensor(val_df.drop("idx", axis=1).values)
    test_data = torch.FloatTensor(test_df.drop("idx", axis=1).values)
    train_data = torch.FloatTensor(train_df.drop("idx", axis=1).values)

    # Train both models
    print("\n" + "=" * 60)
    print("Training model WITHOUT normalization...")
    print("=" * 60)
    results_no_norm = train_and_evaluate_model(
        train_data,
        val_data,
        test_data,
        model_name="autoencoder",
        normalize=False,
    )

    print("\n" + "=" * 60)
    print("Training model WITH normalization...")
    print("=" * 60)
    results_norm = train_and_evaluate_model(
        train_data,
        val_data,
        test_data,
        model_name="autoencoder_normalized",
        normalize=True,
    )

    # Create correlation plots for features
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Non-normalized data correlation
    train_corr_no_norm = np.corrcoef(train_data.numpy().T)
    sns.heatmap(
        train_corr_no_norm,
        ax=axes[0],
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"label": "Correlation"},
    )
    axes[0].set_title("Feature Correlation - Without Normalization")
    axes[0].set_xlabel("Feature Index")
    axes[0].set_ylabel("Feature Index")

    # Normalized data correlation
    train_mean = train_data.mean(dim=0, keepdim=True)
    train_std = train_data.std(dim=0, keepdim=True)
    train_std[train_std == 0] = 1
    train_data_norm = (train_data - train_mean) / train_std
    val_data_norm = (val_data - train_mean) / train_std
    test_data_norm = (test_data - train_mean) / train_std
    train_corr_norm = np.corrcoef(train_data_norm.numpy().T)
    sns.heatmap(
        train_corr_norm,
        ax=axes[1],
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"label": "Correlation"},
    )
    axes[1].set_title("Feature Correlation - With Normalization")
    axes[1].set_xlabel("Feature Index")
    axes[1].set_ylabel("Feature Index")

    plt.tight_layout()
    plt.savefig("feature_correlations.png", dpi=300, bbox_inches="tight")
    print("Feature correlation plots saved to: feature_correlations.png")

    # Create feature distribution plots as violin plots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Non-normalized feature distributions
    train_df_plot = pd.DataFrame(train_data.numpy())
    train_df_plot["Dataset"] = "Train"
    val_df_plot = pd.DataFrame(val_data.numpy())
    val_df_plot["Dataset"] = "Val"
    test_df_plot = pd.DataFrame(test_data.numpy())
    test_df_plot["Dataset"] = "Test"
    combined_df = pd.concat(
        [train_df_plot, val_df_plot, test_df_plot], ignore_index=True
    )
    combined_melted = combined_df.melt(
        id_vars=["Dataset"], var_name="Feature", value_name="Value"
    )

    sns.violinplot(
        data=combined_melted,
        x="Feature",
        y="Value",
        hue="Dataset",
        ax=axes[0],
        cut=0,
    )
    axes[0].set_title("Feature Distributions - Without Normalization")
    axes[0].set_xlabel("Feature Index")
    axes[0].set_ylabel("Feature Value")

    # Normalized feature distributions
    train_norm_df_plot = pd.DataFrame(train_data_norm.numpy())
    train_norm_df_plot["Dataset"] = "Train"
    val_norm_df_plot = pd.DataFrame(val_data_norm.numpy())
    val_norm_df_plot["Dataset"] = "Val"
    test_norm_df_plot = pd.DataFrame(test_data_norm.numpy())
    test_norm_df_plot["Dataset"] = "Test"
    combined_norm_df = pd.concat(
        [train_norm_df_plot, val_norm_df_plot, test_norm_df_plot],
        ignore_index=True,
    )
    combined_norm_melted = combined_norm_df.melt(
        id_vars=["Dataset"], var_name="Feature", value_name="Value"
    )

    sns.violinplot(
        data=combined_norm_melted,
        x="Feature",
        y="Value",
        hue="Dataset",
        ax=axes[1],
        cut=0,
    )
    axes[1].set_title("Feature Distributions - With Normalization")
    axes[1].set_xlabel("Feature Index")
    axes[1].set_ylabel("Feature Value")

    plt.tight_layout()
    plt.savefig("feature_distributions.png", dpi=300, bbox_inches="tight")
    print("Feature distribution plots saved to: feature_distributions.png")

    # Create violin plots for latent space
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Without normalization
    latent_df_no_norm = pd.DataFrame(
        {
            "Latent Dim 1": results_no_norm["train_latent"][:, 0].numpy(),
            "Latent Dim 2": results_no_norm["train_latent"][:, 1].numpy(),
            "Dataset": "Train",
        }
    )
    latent_val_no_norm = pd.DataFrame(
        {
            "Latent Dim 1": results_no_norm["val_latent"][:, 0].numpy(),
            "Latent Dim 2": results_no_norm["val_latent"][:, 1].numpy(),
            "Dataset": "Val",
        }
    )
    latent_test_no_norm = pd.DataFrame(
        {
            "Latent Dim 1": results_no_norm["test_latent"][:, 0].numpy(),
            "Latent Dim 2": results_no_norm["test_latent"][:, 1].numpy(),
            "Dataset": "Test",
        }
    )
    all_latent_no_norm = pd.concat(
        [latent_df_no_norm, latent_val_no_norm, latent_test_no_norm],
        ignore_index=True,
    )

    sns.violinplot(
        data=all_latent_no_norm, x="Dataset", y="Latent Dim 1", ax=axes[0, 0]
    )
    axes[0, 0].set_title("Latent Dimension 1 - Without Normalization")
    axes[0, 0].set_ylabel("Value")

    sns.violinplot(
        data=all_latent_no_norm, x="Dataset", y="Latent Dim 2", ax=axes[0, 1]
    )
    axes[0, 1].set_title("Latent Dimension 2 - Without Normalization")
    axes[0, 1].set_ylabel("Value")

    # With normalization
    latent_df_norm = pd.DataFrame(
        {
            "Latent Dim 1": results_norm["train_latent"][:, 0].numpy(),
            "Latent Dim 2": results_norm["train_latent"][:, 1].numpy(),
            "Dataset": "Train",
        }
    )
    latent_val_norm = pd.DataFrame(
        {
            "Latent Dim 1": results_norm["val_latent"][:, 0].numpy(),
            "Latent Dim 2": results_norm["val_latent"][:, 1].numpy(),
            "Dataset": "Val",
        }
    )
    latent_test_norm = pd.DataFrame(
        {
            "Latent Dim 1": results_norm["test_latent"][:, 0].numpy(),
            "Latent Dim 2": results_norm["test_latent"][:, 1].numpy(),
            "Dataset": "Test",
        }
    )
    all_latent_norm = pd.concat(
        [latent_df_norm, latent_val_norm, latent_test_norm], ignore_index=True
    )

    sns.violinplot(
        data=all_latent_norm, x="Dataset", y="Latent Dim 1", ax=axes[1, 0]
    )
    axes[1, 0].set_title("Latent Dimension 1 - With Normalization")
    axes[1, 0].set_ylabel("Value")

    sns.violinplot(
        data=all_latent_norm, x="Dataset", y="Latent Dim 2", ax=axes[1, 1]
    )
    axes[1, 1].set_title("Latent Dimension 2 - With Normalization")
    axes[1, 1].set_ylabel("Value")

    plt.tight_layout()
    plt.savefig("latent_space_violins.png", dpi=300, bbox_inches="tight")
    print("Latent space violin plots saved to: latent_space_violins.png")

    # Plot comparison on separate plots

    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    for idx, (results, title) in enumerate(
        [
            (results_no_norm, "Without Normalization"),
            (results_norm, "With Normalization"),
        ]
    ):
        ax = axes[idx]

        # Load metrics
        metrics = pd.read_csv(f"{results['csv_logger'].log_dir}/metrics.csv")

        train_loss = metrics[["epoch", "train_loss"]].dropna()
        val_loss = metrics[["epoch", "val_loss"]].dropna()
        test_loss = metrics[["epoch", "test_loss"]].dropna()

        ax.plot(
            train_loss["epoch"],
            train_loss["train_loss"],
            label="Train Loss",
            marker="o",
        )
        ax.plot(
            val_loss["epoch"],
            val_loss["val_loss"],
            label="Validation Loss",
            marker="s",
        )

        # Plot test loss as horizontal line
        if len(test_loss) > 0:
            test_loss_value = test_loss["test_loss"].iloc[0]
            ax.axhline(
                y=test_loss_value,
                color="green",
                linestyle="-",
                label=f"Test Loss ({test_loss_value:.4f})",
                linewidth=2,
            )

        ax.axhline(
            y=results["baseline_train_loss"],
            color="blue",
            linestyle="--",
            label=f"Baseline Train ({results['baseline_train_loss']:.4f})",
            alpha=0.5,
        )
        ax.axhline(
            y=results["baseline_val_loss"],
            color="orange",
            linestyle="--",
            label=f"Baseline Val ({results['baseline_val_loss']:.4f})",
            alpha=0.5,
        )
        ax.axhline(
            y=results["baseline_test_loss"],
            color="red",
            linestyle="--",
            label=f"Baseline Test ({results['baseline_test_loss']:.4f})",
            alpha=0.5,
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.set_title(f"Training Progress - {title}")
        ax.legend()
        ax.grid(True)

        # Limit y-axis for non-normalized plot
        if idx == 0:
            ax.set_ylim(0, 5)

    plt.tight_layout()
    plt.savefig("training_comparison.png", dpi=300, bbox_inches="tight")
    print("\nComparison plot saved to: training_comparison.png")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nWithout Normalization:")
    print(
        f"  Baseline Train Loss: {results_no_norm['baseline_train_loss']:.6f}"
    )
    print(f"  Baseline Val Loss: {results_no_norm['baseline_val_loss']:.6f}")
    print(f"  Baseline Test Loss: {results_no_norm['baseline_test_loss']:.6f}")

    print("\nWith Normalization:")
    print(f"  Baseline Train Loss: {results_norm['baseline_train_loss']:.6f}")
    print(f"  Baseline Val Loss: {results_norm['baseline_val_loss']:.6f}")
    print(f"  Baseline Test Loss: {results_norm['baseline_test_loss']:.6f}")


if __name__ == "__main__":
    main()
