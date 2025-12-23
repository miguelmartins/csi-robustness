import torch
import os
from scipy.stats import pearsonr as corr
from models.baselines import get_model


def LinReg(x, y):
    # beta = np.linalg.pinv(X.T @ X) @ X.T @ Y
    beta = torch.linalg.lstsq(x, y).solution
    return x @ beta


def inference_pass(dataloader, probe, net, device):
    xs, y_true, y_pred, zs = [], [], [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            z = net(x)
            # do the accumulation on cpu
            y = y.to(torch.float32).cpu()
            xs.append(x.cpu())
            y_true.append(y)
            y_pred_ = probe(z).argmax(1).cpu()
            y_pred.append(y_pred_)
            zs.append(z.detach().cpu())
    xs = torch.cat(xs, dim=0)  # for now we leave it in cpu so that it does not blow gpu
    y_true = torch.cat(y_true, dim=0).to(device)
    y_pred = torch.cat(y_pred, dim=0).to(device)
    zs = torch.cat(zs, dim=0).to(device)
    return xs, y_true, y_pred, zs


def log_validation(*, val_dataloader, net, readout, data, cat_ind, log_file, device):
    _, y_true, y_pred, zs = inference_pass(val_dataloader, readout, net, device)
    y_latent = LinReg(zs, y_true)  # TODO: maybe normalize y_true?
    # we need the cast to float since torch does not like to do broadcastable operations on bool
    val_acc = torch.mean((y_true[:, cat_ind] == y_pred).float()).cpu().numpy()
    print("val_acc", val_acc)
    zs = zs.detach().cpu()
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    y_latent = y_latent.cpu().numpy()
    with open(log_file, "a") as file:
        for i in range(y_true.shape[1]):
            print(
                "Coordinate",
                i,
                data.lat_names[i],
                corr(y_true[:, i], y_latent[:, i])[0],
                file=file,
            )

    return val_acc


# val_dataloader, net, readout, data, cat_ind, log_file, device)
def log_test_evaluation(args, dataset, device, log_file):
    with open(log_file, "a") as file:
        print("\n\nEvaluating:", file=file)
    (train_dataloader, val_dataloader, test_dataloader, data, out_size, nc, cat_ind) = (
        dataset
    )
    net = get_model(args.model, nc, out_size, device, args.seed)

    # prepare model
    if args.model == "image":
        pass
    else:
        net.load_state_dict(torch.load(os.path.join(args.log_dir, "model.pth")))
        net.eval()

    x, y_true, _, zs = inference_pass(
        train_dataloader, torch.nn.Identity(), net, device
    )
    x_test, y_true_test, _, zs_test = inference_pass(
        test_dataloader, torch.nn.Identity(), net, device
    )

    # decode all coordinates
    beta = LinReg(zs, y_true)  # TODO: maybe normalize y_true?
    for i in range(y_true.shape[1]):
        # y = y_train[:, i].copy() * 1.0
        # y -= np.mean(y)
        # y /= np.std(y)
        # beta = tmp @ y
        y_train_ = x @ beta
        y_test_ = x_test @ beta
        with open(log_file, "a") as file:
            print(
                "Coordinate",
                i,
                data.lat_names[i],
                "\ntrain",
                corr(y_true[:, i], y_train_),
                "\nval",
                corr(y_true_test[:, i], y_test_),
                file=file,
            )
