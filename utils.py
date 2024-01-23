import torch

# print information on [model] onto the screen
def print_model_info(model, message=None):
    print(55*"-" if message is None else ' {} '.format(message).center(55, '-'))
    print(model)
    print(55*"-")
    _ = count_parameters(model)

# count number of parameters, print to screen
def count_parameters(model, verbose=True):
    total_params = learnable_params = fixed_params = 0
    for param in model.parameters():
        n_params = index_dims = 0
        for dim in param.size():
            n_params = dim if index_dims==0 else n_params*dim
            index_dims += 1
        total_params += n_params
        if param.requires_grad:
            learnable_params += n_params
        else:
            fixed_params += n_params
    if verbose:
        print( "--> this network has {} parameters (~{} million)"
              .format(total_params, round(total_params / 1000000, 1)))
        print("       of which: - learnable: {} (~{} million)".format(learnable_params,
                                                                     round(learnable_params / 1000000, 1)))
        print("                 - fixed: {} (~{} million)".format(fixed_params, round(fixed_params / 1000000, 1)))
    return total_params, learnable_params, fixed_params

# mini batch dataset 
class DataSet():
    def __init__(self, X, Y, transform=None, target_transform=None): 
        self.X = torch.tensor(X).float()
        self.Y = torch.tensor(Y)
        self.transform=transform
    
    def __len__(self):
        return len(self.X) 
    def __getitem__(self, index):
        return self.X[index], self.Y[index]