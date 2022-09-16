from Net import *

pretrained_model = './vgg.pth'
# Initialize the network
model = VGG16().to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()

def test(model, device, test_loader):

    each_num = [0] * 9
    each_correct = [0] * 9
    # Accuracy counter
    correct = 0

    # Loop over all examples in test set
    for batch in tqdm(test_loader):

        # Send the data and label to the device
        data, target = batch['x'], batch['y']
        # print(list(np.array(data[0][0])))
        data, target = data.to(device), target.to(device)
        # print(np.array(data[0][0]))
        # print(target)

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        if init_pred.item() == target.item():
            correct += 1
            each_correct[target.item()] += 1

        each_num[target.item()] += 1


    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Test Accuracy = {} / {} = {}".format(correct, len(test_loader), final_acc))
    print("Attack Accuracy of Center = {}".format(each_correct[0]/ float(each_num[0])))
    print("Attack Accuracy of Donut = {}".format(each_correct[1]/ float(each_num[1])))
    print("Attack Accuracy of Edge_Loc = {}".format(each_correct[2] / float(each_num[2])))
    print("Attack Accuracy of Edge_Ring = {}".format(each_correct[3] / float(each_num[3])))
    print("Attack Accuracy of Local = {}".format(each_correct[4] / float(each_num[4])))
    print("Attack Accuracy of Random = {}".format(each_correct[5] / float(each_num[5])))
    print("Attack Accuracy of Scratch = {}".format(each_correct[6] / float(each_num[6])))
    print("Attack Accuracy of Near_Full = {}".format(each_correct[7] / float(each_num[7])))
    print("Attack Accuracy of none = {}".format(each_correct[8] / float(each_num[8])))
    # Return the accuracy and an adversarial example
    return final_acc

if __name__ == '__main__':

    test(model, device, val_data)