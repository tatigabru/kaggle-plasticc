import json
import warnings
from os import path

import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence


def ts_to_tensors(sample):
    times = torch.tensor(sample["mjd"].values, dtype=torch.float)
    bands = torch.tensor(sample["passband"].values, dtype=torch.long)
    masks = torch.zeros(times.shape[0], 6).type(torch.ByteTensor)
    masks.scatter_(1, bands.reshape(-1, 1), 1)
    return times, masks


class GaussianCurve(nn.Module):
    def __init__(self, batch_size=20, n_bands=6):
        super().__init__()
        self._batch_size = batch_size
        self._n_bands = n_bands

        self.params = nn.ParameterDict(
            {
                "f0": nn.Parameter(torch.zeros(batch_size, dtype=torch.float)),
                "lambda": nn.Parameter(torch.ones(batch_size, dtype=torch.float)),
                "t0": nn.Parameter(torch.ones(batch_size, dtype=torch.float)),
                "fm": nn.Parameter(torch.ones(batch_size, n_bands, dtype=torch.float)),
            }
        )

        self.params["lambda"].data *= 10
        self.params["t0"].data *= 0.5

    def set_refine(self):
        self.params["f0"].data *= 0
        self.params["lambda"].data *= 0
        self.params["lambda"].data += 50
        self.params["fm"].data *= 0
        self.params["fm"].data += 1

    def forward(self, times, masks=None):
        z = (times - self.params["t0"].unsqueeze(1)) ** 2 * torch.clamp(
            self.params["lambda"].unsqueeze(1), min=0
        )
        z = torch.exp(-z)
        y = self.params["fm"].unsqueeze(1) * z.unsqueeze(2) + self.params["f0"].reshape(
            -1, 1, 1
        )
        if masks is not None:
            y = y[masks]
        return y

    def predict(self, times, n):
        z = (times - self.params["t0"][n]) ** 2 * torch.clamp(
            self.params["lambda"][n], min=0
        )
        z = torch.exp(-z)
        y = (
            self.params["fm"][n, :].reshape(1, -1) * z.reshape(-1, 1)
            + self.params["f0"][n]
        )
        return y


class ExpRatioCurve(nn.Module):
    def __init__(self, batch_size=20, n_bands=6, refine=False):
        super().__init__()
        self.params = nn.ParameterDict(
            {
                "f0": nn.Parameter(torch.zeros(batch_size, dtype=torch.float)),
                "lambda_rise": nn.Parameter(torch.ones(batch_size, dtype=torch.float)),
                "lambda_fall": nn.Parameter(torch.ones(batch_size, dtype=torch.float)),
                "t0": nn.Parameter(torch.ones(batch_size, n_bands, dtype=torch.float)),
                "fm": nn.Parameter(torch.ones(batch_size, n_bands, dtype=torch.float)),
            }
        )

        if refine:
            self.params["lambda_rise"].data *= 50
            self.params["lambda_fall"].data *= 50
        else:
            self.params["lambda_rise"].data *= 1
            self.params["lambda_fall"].data *= 1
        self.params["t0"].data *= 0.5

    def set_init_params(self, init_t0, init_lambda=None):
        # assuming using t0 from gaussian:
        self.params["t0"].data *= 0
        self.params["t0"].data += torch.clamp(init_t0, 0, 1).reshape(-1, 1)
        if init_lambda is not None:
            self.params["lambda_rise"].data *= 0
            self.params["lambda_rise"].data += init_lambda
            self.params["lambda_fall"].data *= 0
            self.params["lambda_fall"].data += init_lambda

    def forward(self, times, masks=None):
        t_f = -(times.unsqueeze(2) - self.params["t0"].unsqueeze(1)) * torch.clamp(
            self.params["lambda_fall"].reshape(-1, 1, 1), min=0
        )
        t_r = -(times.unsqueeze(2) - self.params["t0"].unsqueeze(1)) * torch.clamp(
            self.params["lambda_rise"].reshape(-1, 1, 1), min=0
        )
        z = torch.exp(t_f - torch.log1p(torch.exp(t_r)))
        y = self.params["fm"].unsqueeze(1) * z + self.params["f0"].reshape(-1, 1, 1)
        if masks is not None:
            y = y[masks]
        return y

    def predict(self, times, n):
        t_f = -(times.unsqueeze(1) - self.params["t0"][n].unsqueeze(0)) * torch.clamp(
            self.params["lambda_fall"][n], min=0
        )
        t_r = -(times.unsqueeze(1) - self.params["t0"][n].unsqueeze(0)) * torch.clamp(
            self.params["lambda_rise"][n], min=0
        )
        z = torch.exp(t_f - torch.log1p(torch.exp(t_r)))
        y = self.params["fm"][n, :].reshape(1, -1) * z + self.params["f0"][n]
        return y


NORMALISATION_PARAMS = {"mjd_min": 59580.0338, "mjd_max": 60674.363}


def penalty(t0, k=1):
    return torch.mean(
        k * torch.sum((t0 - torch.mean(t0, dim=1, keepdim=True)) ** 2, dim=1)
    )


def batch_fit(
    chunk,
    n_iters_gaussian=5000,
    n_iters_exp=10000,
    sample_plot=False,
    augmentation_set=False,
):
    # normalise mjd
    # mjd_min = chunk['mjd'].min()
    # mjd_max = chunk['mjd'].max()
    group_col = "augmentation_id" if augmentation_set else "object_id"
    chunk["mjd"] = (chunk["mjd"] - NORMALISATION_PARAMS["mjd_min"]) / (
        NORMALISATION_PARAMS["mjd_max"] - NORMALISATION_PARAMS["mjd_min"]
    )

    # normalise flux and flux_err per object
    flux_mean = chunk.groupby(group_col)["flux"].transform("mean")
    flux_std = chunk.groupby(group_col)["flux"].transform("std")
    chunk["flux"] = (chunk["flux"] - flux_mean) / flux_std
    chunk["flux_err"] = chunk["flux_err"] / flux_std

    print()
    print("Tensor conversion...")
    selected = chunk
    groups = selected.groupby(group_col)
    times_list = []
    masks_list = []
    for object_id, group in groups:
        times, masks = ts_to_tensors(group)
        times_list.append(times)
        masks_list.append(masks)

    times = pad_sequence(times_list, batch_first=True)
    masks = pad_sequence(masks_list, batch_first=True)
    fluxes = torch.tensor(selected["flux"].values, dtype=torch.float)
    detected = torch.tensor(selected["detected"].values, dtype=torch.float)

    times = times.cuda()
    masks = masks.cuda()
    fluxes = fluxes.cuda()
    detected = detected.cuda()

    print("Gaussian curve fitting...")
    curve = GaussianCurve(batch_size=selected[group_col].unique().shape[0]).cuda()
    optimiser = optim.Adam(lr=3e-2, params=curve.parameters())
    # scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=1-1e-4)
    fac = 5
    sample_weights = detected * (fac - 1) + 1
    for i in range(n_iters_gaussian):
        f_pred = curve.forward(times, masks)
        loss = torch.mean(sample_weights * (f_pred - fluxes) ** 2)
        curve.zero_grad()
        loss.backward()
        for k, param in curve.params.items():
            if torch.sum(torch.isnan(param.grad)) > 0:
                param.grad.data[torch.isnan(param.grad)] = 0
                warnings.warn(f"NaN encountered in grad of {k}", RuntimeWarning)
        nn.utils.clip_grad_norm_(
            curve.parameters(), 1
        )  # clip gards to ensure stability
        optimiser.step()
        #     scheduler.step()
        if i % 1000 == 0:
            print(f"loss={loss.detach().cpu().numpy()}")

    print("Exp ratio curve fitting...")
    curve2 = ExpRatioCurve(
        batch_size=selected[group_col].unique().shape[0], refine=True
    ).cuda()
    curve2.set_init_params(init_t0=curve.params["t0"].data)
    del optimiser, curve
    optimiser = optim.Adam(lr=1e-2, params=curve2.parameters())
    scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=1 - 1e-4)
    # fac = 5
    # sample_weights = detected * (fac - 1) + 1
    for i in range(n_iters_exp):
        f_pred = curve2.forward(times, masks)
        p = penalty(curve2.params["t0"], k=10)
        loss = torch.mean(sample_weights * (f_pred - fluxes) ** 2) + p
        curve2.zero_grad()
        loss.backward()
        for k, param in curve2.params.items():
            if torch.sum(torch.isnan(param.grad)) > 0:
                param.grad.data[torch.isnan(param.grad)] = 0
                warnings.warn(f"NaN encountered in grad of {k}", RuntimeWarning)
        nn.utils.clip_grad_norm_(
            curve2.parameters(), 1
        )  # clip gards to ensure stability
        optimiser.step()
        scheduler.step()
        if i % 1000 == 0:
            print(
                f"loss={loss.detach().cpu().numpy():.5f}, penalty={p.detach().cpu().numpy():5f}"
            )

    if sample_plot:
        for n in range(10):
            obj = selected[selected[group_col] == selected[group_col].unique()[n]]
            f, ax = plt.subplots(figsize=(12, 6))
            colors = ["red", "orange", "yellow", "green", "blue", "purple"]
            ax.scatter(
                x=obj["mjd"],
                y=obj["flux"],
                c=[colors[b] for b in obj["passband"]],
                s=10,
            )
            ax.vlines(
                obj["mjd"],
                obj["flux"] - obj["flux_err"],
                obj["flux"] + obj["flux_err"],
                colors=[colors[b] for b in obj["passband"]],
                linewidth=1,
            )
            ax.autoscale(False)
            t_range = np.linspace(-0.2, 1.2, 1000)
            y = (
                curve2.predict(torch.tensor(t_range, dtype=torch.float).cuda(), n)
                .detach()
                .cpu()
                .numpy()
            )
            for band in range(6):
                ax.plot(t_range, y[:, band], c=colors[band], alpha=0.5)
            ax.set_title(
                f"object {obj[group_col].iloc[0]}, "
                f'tau_rise: {(1 / curve2.params["lambda_rise"][n]).detach().cpu().numpy()}, '
                f'tau_fall: {(1 / curve2.params["lambda_fall"][n]).detach().cpu().numpy()}\n'
                f't0: {curve2.params["t0"][n, :].detach().cpu().numpy()}\n'
                f'fm: {curve2.params["fm"][n, :].detach().cpu().numpy()}'
            )
            plt.savefig(f"experiments/run_data/plots/plot_{n}.png")

    print("Saving fitted parameters...")
    raw_loss = sample_weights * (f_pred - fluxes) ** 2
    per_obj_loss = (
        pd.Series(raw_loss.detach().cpu().numpy()).groupby(selected[group_col]).mean()
    )
    if augmentation_set:
        params_df = pd.DataFrame(
            {
                "augmentation_id": selected[
                    "augmentation_id"
                ].unique(),  # use only for augmentation
                "exp_ratio_fitting_loss": per_obj_loss,
                "tau_rise": 1
                / curve2.params["lambda_rise"].data.detach().cpu().numpy(),
                "tau_fall": 1
                / curve2.params["lambda_fall"].data.detach().cpu().numpy(),
                "f0": curve2.params["f0"].data.detach().cpu().numpy(),
            }
        )
    else:
        params_df = pd.DataFrame(
            {
                "object_id": selected[
                    "object_id"
                ].unique(),  # use only for non-augmentation
                "exp_ratio_fitting_loss": per_obj_loss,
                "tau_rise": 1
                / curve2.params["lambda_rise"].data.detach().cpu().numpy(),
                "tau_fall": 1
                / curve2.params["lambda_fall"].data.detach().cpu().numpy(),
                "f0": curve2.params["f0"].data.detach().cpu().numpy(),
            }
        )
    fm = curve2.params["fm"].data.detach().cpu().numpy()
    t0 = curve2.params["t0"].data.detach().cpu().numpy()
    for b in range(6):
        params_df[f"fm_{b}"] = fm[:, b]
        params_df[f"t0_{b}"] = t0[:, b]
    del times, masks, fluxes, detected, curve2, optimiser, scheduler
    return params_df


def main():
    torch.manual_seed(7777)

    OUTPUT_DIR = "data/gp_augmented"
    N_gaussian = 2000
    N_exp = 10000

    """
    Fitting training set
    """
    print("*" * 20 + "\nReading data...")
    train_series = pd.read_csv("data/training_set.csv")

    params_df = batch_fit(train_series)
    params_df.to_csv(path.join(OUTPUT_DIR, "train_exp_ratio_fitted.csv"), index=False)

    """
    Fitting aug 10x set
    """
    print("Reading data...")
    test_series = dd.read_csv("data/augmented/augmented_*.csv", blocksize=None)
    for i, part in tqdm(
        enumerate(test_series.partitions), total=test_series.npartitions
    ):
        chunk = part.compute().reset_index()
        params_df = batch_fit(
            chunk, n_iters_gaussian=N_gaussian, n_iters_exp=N_exp, sample_plot=True
        )
        break
        params_df.to_csv(
            path.join(OUTPUT_DIR, f"test_exp_ratio_fitted_{i}.csv"), index=False
        )

    """
    Fitting aug 30x set
    """
    print("*" * 20 + "\nReading data...")
    train_series = pd.read_csv(
        "data/gp_augmented/gp_augmented_ddf_to_nonddf_class_52.csv"
    )

    params_df = batch_fit(train_series, sample_plot=True, augmentation_set=True)
    params_df.to_csv(path.join(OUTPUT_DIR, "train_exp_ratio_fitted.csv"), index=False)


if __name__ == "__main__":
    main()
