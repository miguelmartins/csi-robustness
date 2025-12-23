import torch
from scipy.stats import pearsonr as corr


def LinReg(x, y):
    # beta = np.linalg.pinv(X.T @ X) @ X.T @ Y
    beta = torch.linalg.lstsq(x, y).solution
    return x @ beta


def log_validation(*, val_dataloader, net, readout, data, cat_ind, log_file, device):
    y_true, y_pred, zs = [], [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(val_dataloader):
            x = x.to(device)
            z = net(x)
            # do the accumulation on cpu
            y = y.to(torch.float32).cpu()
            y_true.append(y)
            y_pred_ = readout(z).argmax(1).cpu()
            y_pred.append(y_pred_)
            zs.append(z.detach().cpu())

    y_true = torch.cat(y_true, dim=0).to(device)
    y_pred = torch.cat(y_pred, dim=0).to(device)
    zs = torch.cat(zs, dim=0).to(device)
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
