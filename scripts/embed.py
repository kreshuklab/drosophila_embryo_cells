import argparse
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from inferno.trainers.basic import Trainer
from embryo_loader import EmbryoDataloader


def predict_embed(model, loader):
    all_predictions = []
    model.eval()
    with torch.no_grad():
        for cells, _ in loader:
            preds = model.embed(cells.cuda()).cpu().numpy()
            all_predictions.extend(preds)
    return np.array(all_predictions)


def save_embeddings(embedding, dset, path):
    tf_emb_path = os.path.join(path, 'embed')
    if not os.path.exists(tf_emb_path):
        os.makedirs(tf_emb_path)
    rows = [dset.class_labels[idx] for idx in dset.indices]
    rows_and_ids = ['{}_{}'.format(dset.class_labels[idx], str(idx)) for idx in dset.indices]
    writer = SummaryWriter(tf_emb_path)
    writer.add_embedding(embedding, metadata=rows, tag='row_number')
    writer.add_embedding(embedding, metadata=rows_and_ids, tag='row_number_id')
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save embeddings')
    parser.add_argument('path', type=str, help='train path with model and configs')
    parser.add_argument('--device', type=str, default='0',
                        choices=[str(i) for i in range(8)], help='GPU to use')
    args = parser.parse_args()
    print("The model is", os.path.split(os.path.normpath(args.path))[-1])
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    config_file = os.path.join(args.path, 'data_config.yml')
    val_loader = EmbryoDataloader(config_file).get_predict_loader()
    model_path = os.path.join(args.path, 'Weights')
    best_model = Trainer().load(from_directory=model_path, best=True).model
    predictions = predict_embed(best_model, val_loader)
    save_embeddings(predictions, val_loader.dataset, args.path)
