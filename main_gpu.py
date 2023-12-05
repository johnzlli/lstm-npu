"""RUL Prediction with LSTM"""
from loading_data import *
from model import *
from visualize import *
import numpy as np
from apex import amp, optimizers

N_HIDDEN = 4096  # NUMBER OF HIDDEN STATES
N_LAYER = 8  # NUMBER OF LSTM LAYERS
N_EPOCH = 100  # NUM OF EPOCHS
MAX = 135  # UPPER BOUND OF RUL
LR = 0.000006  # LEARNING RATE


def testing_function(num, group_for_test, precision):
    rmse_test, result_test = 0, list()

    for ite in range(1, num + 1):
        X_test = group_for_test.get_group(ite).iloc[:, 2:]
        X_test_tensors = Variable(torch.Tensor(X_test.to_numpy()))
        X_test_tensors = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

        if precision == "fp16":
            X_test_tensors = X_test_tensors.half()
        test_predict = model.forward(X_test_tensors.cuda())
        data_predict = max(test_predict[-1].cpu().detach().numpy(), 0)
        result_test.append(data_predict)
        rmse_test = np.add(np.power((data_predict - y_test.to_numpy()[ite - 1]), 2), rmse_test)

    rmse_test = (np.sqrt(rmse_test / num)).item()
    return result_test, rmse_test


def train(model_for_train, ntrain, group_for_train, precision):
    """

    :param model_for_train: initialized model
    :param ntrain: number of samples in training set
    :param group_for_train: grouped data per sample
    :return: evaluation results
    """
    rmse_temp = 100

    total_time = 0
    epoch_num = 0
    for epoch in range(1, N_EPOCH + 1):

        model_for_train.train()
        epoch_loss = 0
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for i in range(1, ntrain + 1):
            X, y = group_for_train.get_group(i).iloc[:, 2:-1], group_for_train.get_group(i).iloc[:, -1:]
            X_train_tensors = Variable(torch.Tensor(X.to_numpy()))
            y_train_tensors = Variable(torch.Tensor(y.to_numpy()))
            X_train_tensors = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))

            if precision == "fp16":
                X_train_tensors = X_train_tensors.half()
            outputs = model_for_train(X_train_tensors.cuda())
            start.record()

            optimizer.zero_grad()  # calculate the gradient, manually setting to 0
            loss = criterion(outputs, y_train_tensors.cuda())  # obtain the loss function
            epoch_loss += loss.item()
            if precision == "fp16":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()  # improve from loss, i.e back propagation
            end.record()
            torch.cuda.synchronize()
            total_time += start.elapsed_time(end)

        epoch_num += ntrain

        if epoch % 1 == 0:  # evaluate the model on testing set with each epoch

            model_for_train.eval()  # evaluate model
            result, rmse = testing_function(num_test, group_test, precision)

            if rmse_temp < rmse and rmse_temp < 25:
                result, rmse = result_temp, rmse_temp
                break

            rmse_temp, result_temp = rmse, result  # store the last rmse
            print("Epoch: %d, loss: %1.5f, rmse: %1.5f" % (epoch, epoch_loss / ntrain, rmse))

    print("avg epoch training time: %1.3f ms" % (total_time / epoch_num))

    return result, rmse


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=["fp32", "fp16", None])
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.000006)
    args = parser.parse_args()
    LR=args.lr
    N_EPOCH=args.epoch

    # fetch basic information from data sets
    group, group_test, y_test = load_FD001(MAX)
    num_train, num_test = len(group.size()), len(group_test.size())
    input_size = group.get_group(1).shape[1] - 3  # number of features

    # LSTM model initialization
    model = LSTM1(input_size, N_HIDDEN, N_LAYER).cuda()  # our lstm class
    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-4)

    if args.precision == "fp16":
        model, optimizer = amp.initialize(model, optimizer, opt_level="O3", keep_batchnorm_fp32=True, loss_scale=1)
    # training and evaluation
    result, rmse = train(model, num_train, group, args.precision)
    visualize(result, y_test, num_test, rmse)
