import math
import torch


def compute_perplexity(model, loader, device):
    model.eval()
    loss_sum = 0.
    num_preds_total = 0
    with torch.no_grad():
        for batch in loader:
            X, M, I = [tensor.to(device) for tensor in batch]
            labels = X.masked_fill(~M, -100)
            output = model(X, attention_mask=M, labels=labels)
            num_preds = (labels > -100).sum().item() - X.size(0)
            loss_sum += output.loss.item() * num_preds
            num_preds_total += num_preds
    perp = math.exp(loss_sum / num_preds_total)
    return perp, loss_sum, num_preds_total
