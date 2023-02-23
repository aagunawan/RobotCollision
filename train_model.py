from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def train_model(no_epochs):

    PATH = 'saved/saved_model.pkl'

    batch_size = 256
    learning_rate = 0.001
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()

    loss_function = nn.MSELoss()
    losses = []
    min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    losses.append(min_loss)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss = []
    for epoch_i in range(no_epochs):
        model.train()

        for idx, sample in enumerate(data_loaders.train_loader): # sample['input'] and sample['label']
            pass
            input = sample['input']
            target = [sample['label']]

            # forward
            output = model(torch.tensor(input))           
            loss = loss_function(torch.flatten(output), torch.flatten(torch.tensor(target)))

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss.append(loss.item())
        losses.append(model.evaluate(model, data_loaders.test_loader, loss_function))

    torch.save(model.state_dict(), PATH, _use_new_zipfile_serialization=False)
    plt.plot(train_loss)
    plt.plot(losses)
    plt.show()

if __name__ == '__main__':
    no_epochs = 150
    train_model(no_epochs)
