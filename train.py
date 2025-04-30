import datetime as dt
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import ContrastiveVectorDataset
from model import ContrastiveEmbedder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def info_nce_loss(q, pos, neg, temperature=0.07):
    """
    q:   (B, D)
    pos: (B, D)
    neg: (B, N, D)
    """
    batch_size = q.size(0)

    pos = pos.unsqueeze(1)
    q = q.unsqueeze(1)
    candidates = torch.cat([pos, neg], dim=1)

    sim = torch.bmm(q, candidates.transpose(1, 2)).squeeze(1)

    sim = sim / temperature

    labels = torch.zeros(batch_size, dtype=torch.long, device=q.device)

    return nn.CrossEntropyLoss()(sim, labels)


def train(data_path, projection_dim=256, batch_size=32, negative_sample_size=10, epochs=10, lr=1e-4):
    dataset = ContrastiveVectorDataset(data_path, negative_sample_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    
    model = ContrastiveEmbedder(input_dim=768, projection_dim=projection_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for q_vec, pos_vec, neg_vecs in dataloader:
            q_vec = q_vec.to(device)
            pos_vec = pos_vec.to(device)
            neg_vecs = neg_vecs.to(device)
            q_proj = model(q_vec.squeeze(1))
            pos_proj = model(pos_vec)
            neg_proj = model(neg_vecs.view(-1, 768)).view(neg_vecs.size(0), neg_vecs.size(1), -1)
            loss = info_nce_loss(q_proj, pos_proj, neg_proj)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "contrastive_embedder.pt")

def train_from_checkpoint(data_path, checkpoint, output_path=f'checkpoint/{dt.datetime.today():%Y%m%d}.pt',epochs=5, batch_size=32, negative_sample_size=10, lr=1e-4):
    model = ContrastiveEmbedder()
    model.load_state_dict(torch.load(checkpoint))
    model.to(device)

    dataset = ContrastiveVectorDataset(data_path, negative_sample_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for batch in train_loader:
            q_vec, pos_vec, neg_vecs = [x.to(device) for x in batch]
            q_proj = model(q_vec.squeeze(1))
            pos_proj = model(pos_vec)
            neg_proj = model(neg_vecs.view(-1, 768)).view(neg_vecs.size(0), neg_vecs.size(1), -1)
            loss = info_nce_loss(q_proj, pos_proj, neg_proj)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Fine-Tune Epoch {epoch}] Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), output_path)