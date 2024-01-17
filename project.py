import gymnasium

import os

from model import *  

env = gymnasium.make("CarRacing-v2",continuous=False, render_mode="human")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
GAMMA = 0.99        #discount factor

EPSILON_START = 0.99    #epsilon start value
EPSILON_END = 0.05      #epsilon end value
EPSILON_DECAY = 1000    #epsilon decay factor

TARGET_UPDATE_RATE = 0.005  #update rate of the target network
LEARNING_RATE = 0.0001      #learning rate for the Adam optimizer

racer = RaceCarDriver(env, LEARNING_RATE, GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY, TARGET_UPDATE_RATE, 10000, BATCH_SIZE, device)

# racer.train(200000)

# racer.train(100000)

# racer.train(10000)

# racer.saveModels()

# LOAD 10 episode TIME STEP MODEL
# custom_policy_cnn_path = os.path.join('Training', 'Saved Models', 'Custom_P_Driving_Model_10ep')
# custom_target_cnn_path = os.path.join('Training', 'Saved Models', 'Custom_T_Driving_Model_10ep')
# racer.loadModel(custom_policy_cnn_path, custom_target_cnn_path, 10000)

# #LOAD 40k TIME STEP MODEL
# custom_policy_cnn_path_40196 = os.path.join('Training', 'Saved Models', 'Custom_P_Driving_Model_40196steps')
# custom_target_cnn_path_40196 = os.path.join('Training', 'Saved Models', 'Custom_T_Driving_Model_40196steps')
# racer.loadModel(custom_policy_cnn_path_40196, custom_target_cnn_path_40196, 40196)

#LOAD 100k TIME STEP MODEL
# custom_policy_cnn_path_100490 = os.path.join('Training', 'Saved Models', 'Custom_P_Driving_Model_100490steps')
# custom_target_cnn_path_100490 = os.path.join('Training', 'Saved Models', 'Custom_T_Driving_Model_100490steps')
# racer.loadModel(custom_policy_cnn_path_100490, custom_target_cnn_path_100490, 100490)

#LOAD 200k TIME STEP MODEL
custom_policy_cnn_path_200888 = os.path.join('Training', 'Saved Models', 'Custom_P_Driving_Model_200888steps')
custom_target_cnn_path_200888 = os.path.join('Training', 'Saved Models', 'Custom_T_Driving_Model_200888steps')
racer.loadModel(custom_policy_cnn_path_200888, custom_target_cnn_path_200888, 200888)


#"""
print("Testing")

max_ep = 20

totalReward = 0

for ep_count in range(max_ep):

    episode_reward = 0
    done = False
    state, _ = env.reset()

    state = processImage(state, device)
    episode_steps = 0
    numNegativeInRow = 0

    while not done:
        env.render()
        actionTensor = racer.chooseAction(state, exploit=True)
        next_state, reward, truncated, done, info = env.step(actionTensor.item())
        episode_reward += reward

        episode_steps += 1

        numNegativeInRow = numNegativeInRow + 1 if episode_steps > 50 and reward < 0 else 0



        if truncated or numNegativeInRow >= 75:
            done = True
            break

        next_state = processImage(next_state, device)

        state = next_state

    print('Episode: {}, Episode Reward: {}'.format(ep_count, episode_reward))
    totalReward += episode_reward
env.close()

print("Average reward over " + str(max_ep) + " episodes: " + str(totalReward / max_ep))

#"""

"""
# env = gymnasium.make("CarRacing-v2", continuous=False, render_mode="human")

# print(env.action_space)
# print(env.observation_space)

# env = DummyVecEnv([lambda: env])

# log_path = os.path.join('Training','Logs')
# model = DQN('CnnPolicy', env, verbose=1, batch_size=128,tensorboard_log=log_path)

# print("start learn")
# model.learn(total_timesteps=200000)
# print("end learn")

# ppo_path = os.path.join('Training', 'Saved Models', 'DQN_Driving_Model100')
# model.save(ppo_path)
# del model

# model = DQN.load(ppo_path, env)

# evaluate_policy(model, env, n_eval_episodes=10, render=True)

max_ep = 20
totalReward = 0



for ep_count in range(max_ep):
    episode_steps = 0
    numNegativeInRow = 0        
    # step_count = 0
    episode_reward = 0
    done = False
    state, _ = env.reset()

    while not done:
        env.render()
        action, _ = model.predict(state)
        state, reward, truncated, done, info = env.step(action)
        episode_reward += reward
        episode_steps += 1

        numNegativeInRow = numNegativeInRow + 1 if episode_steps > 50 and reward < 0 else 0

        if numNegativeInRow >= 75:
            done = True
            break


    totalReward += episode_reward

        # if truncated:
        #     done = True

    print('Episode: {}, Episode Reward: {}'.format(ep_count, episode_reward))
print("Average reward over " + str(max_ep) + " episodes: " + str(totalReward / max_ep))
env.close()
"""



"""
state = env.reset()

#Env.Step takes an action as a parameter (for now using a sample action from the action_space of this current state)

#Returns 
# next_state | the next state after doing said action
#  reward | the reward gotten for doing the action
#  done  | whether or not this new state is the terminal state
#  info  | info which is usually empty

next_state, reward, done, info, var1, = env.step(env.action_space.sample())

print('State: {}; Reward: {}, Done: {}, Next State: {}'.format(state, reward, done, next_state))
print('Action Space: {}'.format(env.action_space.n))

print('Observation Space: {}'.format(env.observation_space))
"""
"""
max_ep = 1

for ep_count in range(max_ep):
    step_count = 0
    episode_reward = 0
    done = False
    state = env.reset()

    while not done:
        next_state, reward, done, truncated, info = env.step(env.action_space.sample())
        if truncated:
            done = True
        env.render()
        # print(step_count)
        step_count += 1
        episode_reward += reward
        state = next_state

    print('Episode: {}, Step Count: {}, Episode Reward: {}'.format(ep_count, step_count, episode_reward))
env.close()



# """


'''
=============================================================================================================
Refrences used:

Environment
1. https://gymnasium.farama.org/environments/box2d/car_racing/#car-racing
2. https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py

Reinforcement Learning Agent
3. https://www.nature.com/articles/s41598-022-06326-0
4. https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
5. https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
6. https://www.baeldung.com/cs/q-learning-vs-deep-q-learning-vs-deep-q-network
7. https://towardsdatascience.com/applying-a-deep-q-network-for-openais-car-racing-game-a642daf58fc9


=============================================================================================================
'''