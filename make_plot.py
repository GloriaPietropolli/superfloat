import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from hyperparameter import *

if mb_flag == 1:
    path = path_target + "mb/"
if mb_flag == 0:
    path = path_target + "no_mb/"


def validation_MAE_RMSE(len_validation, target_validation, result_validation):
    rmse = torch.square(1 / len_validation * (torch.sum((result_validation - target_validation) ** 2)))
    rmse = rmse.item()
    mae = (1 / len_validation) * torch.sum(torch.abs((result_validation - target_validation)))
    mae = mae.item()
    return mae, rmse


def comparison_plot(target, result_validation, mae, rmse):
    colors = cm.rainbow(np.linspace(0, 1, len(target.detach().numpy())))
    plt.text(0.5, -1.5, f"[MAE]: {mae},\n[RMSE]: {rmse}", style='italic',
             bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

    plt.scatter(target.detach().numpy(), result_validation.detach().numpy(), s=1, color=colors, label='data')
    x_min, x_max, y_min, y_max = -2, 3, -2, 3
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    variable = range(x_min, x_max + 1)
    plt.plot(variable, variable, label='Our Fitting Line')
    plt.xlabel('true target')
    plt.ylabel('approximated target')
    plt.title('Comparison between the true result and the approximated one')
    plt.legend()

    if mb_flag == 1:
        plt.savefig(path + "ep" + str(epochs) + "_samples" + "_lr" + str(lr) + "_bs" + str(batch_size) + "_comparison"
                                                                                                         ".png")
    if mb_flag == 0:
        plt.savefig(path + "ep" + str(epochs) + "_samples" + "_lr" + str(lr) + "_comparison.png")

    plt.show()


def error_plot(losses, mae, rmse):
    plt.plot(losses, '-b', label='loss')

    plt.xlabel("n iteration")
    plt.legend(loc='upper left')
    plt.title('Loss and accuracy plot')
    plt.text(epochs - epochs * (60 / 100), 0.45, f"[MAE]: {mae},\n[RMSE]: {rmse}", style='italic',
             bbox={'facecolor': 'red', 'alpha': 0.1, 'pad': 10})

    if mb_flag == 1:
        plt.savefig(path + "ep" + str(epochs) + "_samples" + "_lr" + str(lr) + "_bs" + str(
            batch_size) + "_loss.png")
    if mb_flag == 0:
        plt.savefig(path + "ep" + str(epochs) + "_samples" + "_lr" + str(lr) + "_bs" + "_loss.png")

    plt.show()


def LOG_error_plot(losses, mae, rmse):
    plt.plot(np.log(losses), '-b', label='loss')

    plt.xlabel("n iteration")
    plt.legend(loc='upper left')
    plt.title('LOGLoss and accuracy plot')
    plt.text(epochs - epochs * (60 / 100), 0.45, f"[MAE]: {mae},\n[RMSE]: {rmse}", style='italic',
             bbox={'facecolor': 'red', 'alpha': 0.1, 'pad': 10})

    if mb_flag == 1:
        plt.savefig(path + "ep" + str(epochs) + "_samples" + "_lr" + str(lr) + "_bs" + str(
            batch_size) + "_LOGloss.png")
    if mb_flag == 0:
        plt.savefig(path + "ep" + str(epochs) + "_samples" + "_lr" + str(lr) + "_bs" + "_LOGloss.png")

    plt.show()


def get_all_plot(target_validation, result_validation, losses):
    len_validation = len(target_validation[:, 0])  # number of element of the validation set
    mae, rmse = validation_MAE_RMSE(len_validation, target_validation, result_validation)
    comparison_plot(target_validation, result_validation, mae, rmse)
    error_plot(losses, mae, rmse)
    LOG_error_plot(losses, mae, rmse)
