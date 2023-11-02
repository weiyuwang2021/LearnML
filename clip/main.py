import os 
print(os.getcwd())

from clip.CLIPDataset import build_loaders
from transformers import DistilBertTokenizer
import itertools
from datasets import load_dataset
from clip.CFG import CFG
from clip import CLIPModel
import torch
from clip.train import train_epoch, valid_epoch

def main():
    dataset = load_dataset('svjack/pokemon-blip-captions-en-ja', split='train')
    split_dataset = dataset.train_test_split(test_size=0.2, seed=18)

    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    train_loader = build_loaders(split_dataset, tokenizer, mode="train")
    valid_loader = build_loaders(split_dataset, tokenizer, mode="test")

    model = CLIPModel().to(CFG.device)
    params = [
        {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "best.pt")
            print("Saved Best Model!")
        
        lr_scheduler.step(valid_loss.avg)

if __name__ == "__main__":
    main()