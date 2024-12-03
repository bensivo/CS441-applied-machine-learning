import torch

import torch.nn as nn
import torch.optim as optim

from operations.net import Net


def nn_train(dataloader, model_filepath):
    """
    Train the model on the given dataset, saving the model to the given path

    Params:
        dataloader: torch.utils.data.DataLoader - The data to train with
        output_path: str - The path to save the model to

    Returns:
        None
    """
    print('Training neural network')
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data

            # Initialize the gradients to zero
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Optimize
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}')
                running_loss = 0.0

    print('Finished Training')

    # Save the model to a file
    path = model_filepath
    torch.save(net.state_dict(), path)
