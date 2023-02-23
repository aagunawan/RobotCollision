import torch
import torch.nn as nn
# from Data_Loaders import *

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
# STUDENTS: __init__() must initiatize nn.Module and define your network's
# custom architecture
        pass
        super(Action_Conditioned_FF, self).__init__()
        input_size = 6
        hidden_layer_1_size = 4
        hidden_layer_2_size = 3
        output_size = 1
        self.input_hidden_1 = nn.Linear(input_size, hidden_layer_1_size)
        self.hidden_1_hidden_2 = nn.Linear(hidden_layer_1_size, hidden_layer_2_size)
        self.nonlinear_activation = nn.ReLU()
        self.hidden_2_to_output = nn.Linear(hidden_layer_2_size, output_size)

    def forward(self, input):
# STUDENTS: forward() must complete a single forward pass through your network
# and return the output which should be a tensor
        hidden_1 = self.input_hidden_1(input)
        hidden_1 = self.nonlinear_activation(hidden_1)
        hidden_2 = self.hidden_1_hidden_2(hidden_1)
        hidden_2 = self.nonlinear_activation(hidden_2)
        output = self.hidden_2_to_output(hidden_2)
        return output


    def evaluate(self, model, test_loader, loss_function):
# STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
# mind that we do not need to keep track of any gradients while evaluating the
# model. loss_function will be a PyTorch loss function which takes as argument the model's
# output and the desired output.
        input = []
        target = []
        for idx, sample in enumerate(test_loader):
            input.append(sample['input']) 
            target.append([sample['label']])
        
        output = model((torch.tensor(input)))
        loss = loss_function(torch.flatten(output), torch.flatten(torch.tensor(target))).item()
        return loss

def main():
    model = Action_Conditioned_FF()
    # batch_size = 16
    # data_loaders = Data_Loaders(batch_size)
    # model.evaluate(model, data_loaders.test_loader, nn.MSELoss())

if __name__ == '__main__':
    main()
