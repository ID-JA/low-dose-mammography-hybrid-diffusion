import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.cbis_ddsm import CBISDDSM
from models.vqvae import VQVAE


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ds = CBISDDSM(
        case_csv_path="data/raw/csv/mass_case_description_train_set.csv",
        jpeg_root="data/raw/jpeg",
        dicom_info_csv="data/raw/csv/dicom_info.csv",
        mode="full",
        return_mask=False,
        resize_hw=(512, 512),
    )

    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

    model = VQVAE(in_channels=1, hidden=64, z_channels=64, num_embeddings=512).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(3): 
        model.train()
        total_loss = 0

        for x, meta in loader:
            x = x.to(device)

            x_hat, vq_loss = model(x)
            recon_loss = F.l1_loss(x_hat, x) 
            loss = recon_loss + 1.0 * vq_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"Epoch {epoch+1}: loss={avg:.4f}")

        torch.save(model.state_dict(), f"checkpoints/vqvae_epoch{epoch+1}.pt")

    print("✅ Training finished")


if __name__ == "__main__":
    main()
