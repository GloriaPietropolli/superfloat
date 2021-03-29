from preparation_data import *
from preparation_function import PrepareData
from torch.utils.data import Dataset, DataLoader

from MLP_Bayesian_NN import *
from train import *
from hyperparameter import batch_size, epochs
from make_plot import get_all_plot

ds = PrepareData(X=training_input, y=training_target)
ds = DataLoader(ds, batch_size=batch_size, shuffle=True)  # division of the dataset in one or more batches

training_target.resize_(training_set_size, 1)
validation_target.resize_(validation_set_size, 1)

losses = mylosses()

model_mlp = MLP_Bayesian()
optimizer = torch.optim.Adam(model_mlp.parameters(), lr=lr)  # momentum=0.5)
# train_mb(model_mlp, epochs, training_input, training_target, optimizer, ds)
train(model_mlp, epochs, training_input, training_target, optimizer)

result = model_mlp(training_input)  # result = model_selected(data)

result_validation = model_mlp(validation_input)

get_all_plot(training_target, result, losses)

# mae, rmse = validation_MAE_RMSE(len_validation, target_validation, result_validation)

# print('MAE : ', mae, '\n RMSE : ', rmse)
