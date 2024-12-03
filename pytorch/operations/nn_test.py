import torch
from operations.net import Net

def nn_test(dataloader, model_filepath: str, classes: tuple):
    """
    Evaluate the given model on the given dataset

    Params:
        dataloader: torch.utils.data.DataLoader - The data to evaluate on
        model_filepath: str - The path to the model to evaluate

    Returns:
        None
    """
    print('Evaluating neural network')

    # Test the network on the test data
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    # New model instance, using the saved model weights
    net = Net()
    net.load_state_dict(torch.load(model_filepath, weights_only=True))

    # Get the outputs
    # outputs = net(images)
    # _, predicted = torch.max(outputs, 1)
    # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    # Test the network on the entire test data
    correct = 0
    correct_pred = {classname: 0 for classname in classes}
    total = 0
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad(): # Don't calculate any gradients, because we're not training
        for data in dataloader:
            images, labels = data
            outputs = net(images)

            # Get total accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Get class-wide accuracy
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total}%')
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * correct_count // total_pred[classname]
        print(f'Accuracy for class {classname:5s}: {accuracy:.1f}%')

