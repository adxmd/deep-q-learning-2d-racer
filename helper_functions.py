import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import torch

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

def convertRGBtoGrayscale(image):

    #next_state = torch.tensor(reshaped, dtype=torch.float32, device=device)
    
    #image should be x,y,3 nparray
    grayscale = np.dot(image[...,:3], [0.299, 0.587, 0.114])
    return grayscale

#Process image from (3, 96, 96) RGB Image to (1, 96, 96) Grayscale Tensor
def processImage(image, device):
    grayscale = np.dot(image[...,:3], [0.299, 0.587, 0.114])
    reshaped = np.reshape(grayscale, (1,1,96,96))
    finalTensor = torch.tensor(reshaped, dtype=torch.float32, device=device)

    return finalTensor

#Plot the image (can view our states)
def plotImg(img):
    # img = x.reshape((84, 28))
    plt.imshow(img, cmap='gray')
    plt.show()

#Plot our episode durations in a line graph
def plot_durations(episode_durations, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 50:
        means = durations_t.unfold(0, 50, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(49), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


#Plot our episode rewards in a line graph
def plot_rewards(episode_rewards, show_result=False):
    plt.figure(2)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 50:
        means = rewards_t.unfold(0, 50, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(49), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())