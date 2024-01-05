import torch
from tqdm import tqdm
import numpy as np

def predict(model, dataloader):
    model.eval()
    with torch.no_grad():
        distances, labels = [], []
        print("Testing on IJB! ...")
        progress_bar = enumerate(tqdm(dataloader))

        for batch_index, (data_a, data_b, label) in progress_bar:
            data_a = data_a.to('cuda', non_blocking=False) #Turn non_blocking ON if you are running on a cluster
            data_b = data_b.to('cuda', non_blocking=False)
            output_a, output_b = model(data_a).detach().cpu().numpy(), model(data_b).detach().cpu().numpy()
            norms_a, norms_b = np.linalg.norm(output_a, axis=1, keepdims=True), np.linalg.norm(output_b, axis=1, keepdims=True)
            output_a, output_b = output_a / norms_a, output_b / norms_b
            diff = output_a - output_b
            distance = np.linalg.norm(diff,axis=1)  # Normalized cosine distance
            distances.append(distance)
            labels.append(label.cpu().detach().numpy())
    return distances ,labels