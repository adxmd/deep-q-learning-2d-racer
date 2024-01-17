import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import random
import torch
import math
import time
import os
from collections import namedtuple, deque
from itertools import count
from helper_functions import processImage, plot_durations, plot_rewards

#Input as 96x96x3 = 9,216 x 3 = 27,648
#input size = 96x96 = 9216
#numActions = 5    (number of actions) (nothing, gas, brake, steer right, steer left)
#in_channels = 1 (because we are using grayscale)


# Define a CNN model using PyTorch
class CNN(nn.Module):
    def __init__(self, numActions=5):   
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4, stride=4, padding=0)
            # print(self.conv1)
            self.conv2 = nn.Conv2d(8, 32, kernel_size=4,stride=1,padding=0)
            # print(self.conv2)
            self.pool = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)
            # print(self.pool)
            self.fc1 = nn.Linear(32 * 11 * 11, 2000)
            # print(self.fc1)
            self.fc2 = nn.Linear(2000, 500)
            # print(self.fc2)
            self.fc3 = nn.Linear(500, numActions)
            # print(self.fc3)

    def forward(self, x):
        
        # print("before conv1: " + str(x[0].shape))
        x = F.relu(self.conv1(x))
        # print("after conv1: " + str(x[0].shape))
        x = F.relu(self.conv2(x))
        # print("after conv2: " + str(x[0].shape))
        x = self.pool(x)
        # print("after maxpool: " + str(x[0].shape))
        x = x.reshape(x.shape[0], -1)
        # print("after flatten: " + str(x[0].shape))
        x = F.relu(self.fc1(x))
        # print("after linear1: " + str(x[0].shape))
        x = F.relu(self.fc2(x))
        # print("after linear2: " + str(x[0].shape))
        x = self.fc3(x)
        # print("after linear3: " + str(x[0].shape))
        return x
 
class RaceCarDriver:
    def __init__(self, 
                 env,
                 learningRate=0.0001,
                 discountFactor=0.99,
                 epsilonStart=0.99,
                 epsilonEnd=0.05,
                 epsilonDecay=1000,
                 tUpdateRate=0.005,
                 memorySize=10000,
                 batchSize=128,
                 device=None
                 ):
        
        self.env = env

        self.device = device

        self.learningRate = learningRate
        self.discountFactor = discountFactor

        self.epsStart = epsilonStart
        self.epsEnd = epsilonEnd
        self.epsDecay = epsilonDecay

        self.targetUpdateRate = tUpdateRate
        
        self.memorySize = memorySize
        self.batchSize = batchSize

        numActions = self.env.action_space.n

        self.policyCNN = CNN(numActions).to(self.device)
        self.targetCNN = CNN(numActions).to(self.device)

        self.targetCNN.load_state_dict(self.policyCNN.state_dict())

        self.optimizer = optim.AdamW(self.policyCNN.parameters(), lr=self.learningRate, amsgrad=True)

        self.memory = Memory(self.memorySize)

        self.stepsCompleted = 0
        self.episodeDurations = []
        self.episodeRewards = []
    
    
    def chooseAction(self, state, exploit=False):
        #state is inputted as a (1,1,96,96) grayscale image representation
        
        #Random number from 0 to 1
        sample = random.random()

        #Get current epsilon value based on how many steps we've trained for
        EPSILON_THRESHOLD = self.epsEnd + (self.epsStart - self.epsEnd) * math.exp(-1 * self.stepsCompleted / self.epsDecay)

        #Increment timestep counter
        self.stepsCompleted += 1

        #if we are in exploitation mode or we don't choose random action
        if sample > EPSILON_THRESHOLD or exploit == True:
            with torch.no_grad():
                # print("agent choosing action")
                return self.policyCNN(state).max(1).indices.view(1,1)

        #If in exploration mode (default) and choosing random value
        else:
            # print("random action chosen")
            return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long, device=self.device)
        
    #4. https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    def optimize_model(self):
        if(len(self.memory) < self.batchSize):
            return
        
        memorySample = self.memory.getBatch(self.batchSize)

        batch = MemoryFrame(*zip(*memorySample))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, 
                                                batch.next_state)), device=self.device,dtype=torch.bool)
        
        non_final_next_states = torch.cat([s for s in batch.next_state
                                        if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policyCNN
        state_action_values = self.policyCNN(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batchSize, device=self.device)

        with torch.no_grad():
            next_state_values[non_final_mask] = self.targetCNN(non_final_next_states).max(1).values
        

        expected_state_action_values = (next_state_values * self.discountFactor) + reward_batch

        criterion = nn.SmoothL1Loss()
        
        #Calculate smooth l1 loss
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        #Reset gradient of optimizer
        self.optimizer.zero_grad()
        #Backpropogation
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policyCNN.parameters(), 100)
        #Take an optimizer step
        self.optimizer.step()

    def train(self, total_timesteps):
        plt.ion()

        numEpisodes = total_timesteps / 50

        start_time = time.time()

        for i_episode in range(int(numEpisodes)):
            #Keeps track of how many timesteps each episode was
            epSteps = 0
            #Keeps track of how many negative reward values in a row we get (to avoid super long training episodes of robot doing non-optimal tasks)
            numNegativeInRow = 0
            print("---------------------")
            print("Starting episode " + str(i_episode))

            #Training start time
            ep_start_time = time.time()

            #Keeps track of the reward of the current episode
            ep_reward = 0
            
            #Get the initial state
            state, info = self.env.reset()

            #Process image into grayscale tensor
            state = processImage(state, device=self.device)
            
            #Repeat until we break
            for t in count():

                #Get the agent to choose an action based on the given input state
                action = self.chooseAction(state)

                #Increment the episode steps counter
                epSteps += 1

                #Using the action chosen by the agent, make a move in the environment and get result of that move
                next_state, reward, terminated, truncated, _ = self.env.step(action.item())

                #Increase the total episode reward by this timestep's reward
                ep_reward += reward

                #Increment the negative counter by 1 if more than 50 steps have been completed 
                #this episode and we just got a reward that's less than 0.
                #If neither are true, set it to 0
                numNegativeInRow = numNegativeInRow + 1 if epSteps > 50 and reward < 0 else 0

                #Convert reward value to a tensor
                reward = torch.tensor([reward],device=self.device)
                
                #If episode ends for whatever reason
                done = terminated or truncated
                
                #If episode terminated
                if terminated:
                    next_state = None
                #If not
                else: 
                    #Process next_state image to grayscale tensor
                    next_state = processImage(next_state, device=self.device)
                
                #Store this information in memory as tensors
                self.memory.memorize(state, action, next_state, reward)

                state = next_state

                #Complete an optimize step 
                self.optimize_model()

                #Get the current state of the target and policy network
                target_cnn_state_dict = self.targetCNN.state_dict()
                policy_cnn_state_dict = self.policyCNN.state_dict()

                #Update target network 
                for key in policy_cnn_state_dict:
                    target_cnn_state_dict[key] = policy_cnn_state_dict[key] * self.targetUpdateRate + target_cnn_state_dict[key] * (1-self.targetUpdateRate)

                #Load new target network
                self.targetCNN.load_state_dict(target_cnn_state_dict)
                
                #If episode is done for any reason, or if we've gotten 25 negative rewards in a row
                if done or numNegativeInRow >= 25:
                    #Store this epsiodes duration and reward
                    self.episodeDurations.append(t + 1)
                    self.episodeRewards.append(ep_reward)

                    #Add this new information to the graph
                    plot_durations(self.episodeDurations)
                    plot_rewards(self.episodeRewards)

                    #Get end time
                    end_time = time.time()
                    #Elapsed time since episode started
                    epElapsed = end_time - ep_start_time
                    #Elapsed time since we started training
                    totalElapsed = end_time-start_time
                    print("Ending episode " + str(i_episode) + 
                          "\n | episode reward: " + str(ep_reward) + 
                          "\n | episode elapsed time: " + str(epElapsed) + 
                          "\n | episode timesteps: " + str(epSteps) +
                          "\n | total elapsed time: " + str(totalElapsed) + 
                          "\n | total timesteps: " + str(self.stepsCompleted))
                    print("---------------------")
                    
                    break
            #If completed all the timesteps, stop training
            if(self.stepsCompleted >= total_timesteps):
                print("Total Training Timesteps Reached")
                break
        
        print("Completed training")
        print("Saving model with " + str(self.stepsCompleted) + " timesteps of training completed")
        self.saveModels()

        #Plot the final results in the graph
        plot_durations(show_result=True, episode_durations=self.episodeDurations)
        plot_rewards(show_result=True,episode_rewards=self.episodeRewards)

        plt.ioff()
        plt.show()

    def saveModels(self):
        
        #Get the appropriate file path to save
        pName = 'Custom_P_Driving_Model_' + str(self.stepsCompleted) + 'steps'
        tName = 'Custom_T_Driving_Model_' + str(self.stepsCompleted) + 'steps'

        customPolicyCNN_path = os.path.join('Training', 'Saved Models', pName)
        customTargetCNN_path = os.path.join('Training', 'Saved Models', tName)

        # durationFile = open(("" + str(self.stepsCompleted) + "stepsDurationLog"), "r")
        
        #Save the state_dict of each of the CNNs

        print("Saving policy...")
        torch.save(self.policyCNN.state_dict(), customPolicyCNN_path)
        print("Saved Policy.")

        print("Saving target...")
        torch.save(self.targetCNN.state_dict(), customTargetCNN_path)
        print("Saved Target.")

    #Loads a previously trained model
    def loadModel(self, pPath, tPath, stepsCompleted):

        self.stepsCompleted = stepsCompleted

        print("Loading policy cnn")
        self.policyCNN = CNN(self.env.action_space.n).to(self.device)
        self.policyCNN.load_state_dict(torch.load(pPath))
        self.policyCNN.eval()
        print("Loaded policy cnn")

        print("Loading target cnn")
        self.targetCNN = CNN(self.env.action_space.n).to(self.device)
        self.targetCNN.load_state_dict(torch.load(tPath))
        self.targetCNN.eval()
        print("Loaded target cnn")

#Holds a tuple of four values representing the state, 
#                                              action when inputting state into cnn, 
#                                              next_state and reward when action is made
MemoryFrame = namedtuple('MemoryFrame', ('state', 'action','next_state','reward'))

#Holds our memory tuples and allows us to perform actions of them
class Memory():

    def __init__(self, memory_size):
        self.memory = deque([], maxlen=memory_size)

    def memorize(self, state, action, next_state, reward):
        memory_frame = MemoryFrame(state, action, next_state, reward)
        self.memory.append(memory_frame)
        
    def getBatch(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
   