"""
Training script for IceSlider encoder + inverse model.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from data import create_dataloader  # noqa: E402
from models import Encoder, InverseModel, ForwardModel  # noqa: E402
from losses import MarginLoss

# -- Hyperparameters --
LATENT_DIM = 16
NUM_ACTIONS = 5
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 10
EXPERIENCE_PATH = BASE_DIR / "iceslider_experience.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train(
    experience_path: str = EXPERIENCE_PATH,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
):
    """Main training loop."""
    print(f"Using device: {DEVICE}")

    # 1. Initialize models
    encoder = Encoder(latent_dim=LATENT_DIM).to(DEVICE)
    inverse_model = InverseModel(latent_dim=LATENT_DIM, num_actions=NUM_ACTIONS).to(DEVICE)
    forward_model = ForwardModel(latent_dim=LATENT_DIM, num_actions=NUM_ACTIONS).to(DEVICE)
    
    # 2. Create DataLoader
    dataloader = create_dataloader(experience_path=experience_path, batch_size=batch_size)

    # 3. Define Loss and Optimizer
    params_to_optimize = list(encoder.parameters()) + list(inverse_model.parameters()) + list(forward_model.parameters())
    optimizer = optim.Adam(params_to_optimize, lr=learning_rate)
    criterion_inverse= nn.CrossEntropyLoss()
    criterion_forward = nn.MSELoss()
    criterion_margin = MarginLoss()

    # 4. Training Loop
    print("Starting training...")
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, (s_t, s_t1, a_t) in enumerate(dataloader):
            s_t, s_t1, a_t = s_t.to(DEVICE), s_t1.to(DEVICE), a_t.to(DEVICE)

            optimizer.zero_grad()

            z_t = encoder(s_t)
            z_t1 = encoder(s_t1)

            predicted_action_logits = inverse_model(z_t, z_t1)
            predicted_latent_state = forward_model(z_t, a_t)

            loss_inverse = criterion_inverse(predicted_action_logits, a_t)
            loss_forward = criterion_forward(predicted_latent_state, z_t1)
            loss_margin = criterion_margin(z_t, z_t1)

            loss = loss_inverse + loss_forward + loss_margin

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}, Inverse Loss: {loss_inverse.item():.4f}, Forward Loss: {loss_forward.item():.4f}, Margin Loss: {loss_margin.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"--- End of Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f} ---")

    # 5. Save the trained models
    encoder_path = BASE_DIR / "encoder_model_grayscale.pth"
    inverse_path = BASE_DIR / "inverse_model_grayscale.pth"
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(inverse_model.state_dict(), inverse_path)
    print("Training complete. Models saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IceSlider encoder + inverse model")
    parser.add_argument("--experience", type=str, default=str(EXPERIENCE_PATH), help="Path to experience pickle")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    args = parser.parse_args()

    train(
        experience_path=args.experience,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

