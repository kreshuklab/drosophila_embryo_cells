import argparse
import os
import numpy as np
import torch
from inferno.trainers.basic import Trainer
from sklearn import metrics
from embryo_loader import EmbryoDataloader


def predict(model, loader):
    all_predictions = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for cells, labels in loader:
            preds = model(cells.cuda()).cpu().numpy()
            all_targets.extend(labels)
            all_predictions.extend(preds)
    return np.array(all_predictions), np.array(all_targets)


def print_results(pred_labels, true_labels):
    # FIX ME
    pred_labels = np.rint(pred_labels * 5 + 3)
    true_labels = true_labels * 5 + 3
    pred_distances = []
    for lbl in np.unique(true_labels):
        distance = np.sum(np.abs(pred_labels[np.where(true_labels == lbl)] - lbl)) \
                         / np.sum(true_labels == lbl)
        pred_distances.append(distance)
    print("Average distance is ", np.mean(pred_distances))
    print(metrics.confusion_matrix(true_labels, pred_labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict embryo cells phenotype')
    parser.add_argument('path', type=str, help='train path with model and configs')
    parser.add_argument('--device', type=str, default='0',
                        choices=[str(i) for i in range(8)], help='GPU to use')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    config_file = os.path.join(args.path, 'data_config.yml')
    val_loader = EmbryoDataloader(config_file).get_predict_loader()
    model_path = os.path.join(args.path, 'Weights')
    best_model = Trainer().load(from_directory=model_path, best=True).model
    print("The model is", os.path.split(os.path.normpath(args.path))[-1])
    predictions, targets = predict(best_model, val_loader)
    print_results(predictions, targets)
