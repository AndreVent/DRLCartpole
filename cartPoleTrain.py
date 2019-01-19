
import random
import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


class cartPoleAgent:

    def __init__(self, env, memorySize):
        self.memorySize = int(memorySize) #How many frames to keep track of
        self.observation_space = env.observation_space.shape[0] #Amount of info stored about the cart (Cart Position,Cart Velocity,Pole Angle,Pole Velocity At Tip)
        self.action_space = env.action_space.n #Amount of actions that can be taken
        self.memory = deque(maxlen=self.memorySize) #Used to keep track of frames for learning
        self.gamma = 0.8 #Discount rate, for future discounted rewards
        self.learningRate = 0.001 #Controls how fast we modify our estimates
        self.EPS = 0.3 #Epsilon rate, the chance the model makes a random decision
        self.EPSMin = 0.01 #Minimum epsilon rate, minimum chance the model will make a random decision
        self.EPSDecay = 0.995 #Through out the game reduce the rate it randomly chooses a action
        self.batchSize = 32 #How many samples from the previouse games does the net learn from
        self.model = self.createModel()

    def createModel(self):
        # Simple Neural Net
        # Sequential type for the layers
        model = Sequential()
        # Most basic form of a neural net layer, Dense
        # Input Layer of size 4 (Cart Position,Cart Velocity,Pole Angle,Pole Velocity At Tip)
        model.add(Dense(24, input_dim=self.observation_space, activation='relu'))
        # Hidden layer with 24 nodes
        model.add(Dense(24, activation='relu'))
        # Output Layer of size 2 the number of actions that can be taken (Left or Right)
        model.add(Dense(self.action_space, activation='linear'))
        # Create the model using the layers above
        model.compile(loss='mse',optimizer=Adam(lr=self.learningRate))
        return model

    def pushToMemory(self, state, action, reward, observation, done):
        #Save the previouses states action, reward, future state and if it failed or not to "memory"
        self.memory.append((state, action, reward, observation, done))

    def nextAction(self, state):
        if np.random.rand() <= self.EPS:
            #Have the agent take a random action
            return random.randrange(self.action_space)
        #Get the expected rewards for preforming actions in our current state
        qValues = self.model.predict(state)
        #pick the action that gives the greatest reward, qValue list looks like this [[0.8, 0.2]], the numbers are the rewards for pick either action 0 (left) or 1 (right)
        #argmax function picks the highest value and returns its index
        return np.argmax(qValues[0])

    def train(self, batchSize):
        if len(agent.memory) < batchSize:
            return
        #get a batch from the models memory for training
        currentBatch = random.sample(self.memory, batchSize)
        #Get the information from all the memory the model has stored
        for state, action, reward, observation, done in currentBatch:
            #if the done make loss = to the reward
            loss = reward
            if not done:
                #Predict future discounted reward
                #We want to decrease the gap between the prediction and the target (loss)
                #General Q learning formula but learning rate multiplactaion and subtraction of prediction done by the neural net
                loss = (reward + self.gamma * np.amax(self.model.predict(observation)[0]))
            #Map out the rewards for moving to different states from the current state e.g. rewards for action left or right
            qValues = self.model.predict(state)
            #set the actual reward for preforming the action
            qValues[0][action] = loss
            #retrain the model given the state and the qValues
            self.model.fit(state, qValues, epochs=1, verbose=0)
        #while the epsilon value is bigger than the min decrease so less random actions are taken
        if self.EPS > self.EPSMin:
            self.EPS *= self.EPSDecay
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


env = gym.make('CartPole-v1')
#print('Enter how many games you want to play:')
games = 500
#print('Enter the max score the AI can get:')
maxScore = 200
#print('Enter how many frames you want to store:')
memorySize = 1000000
#print('Enter how many frames you want to learn from:')
batchSize = 64
agent = cartPoleAgent(env, memorySize)
scoreList = deque(maxlen=1000)
#agent.load("./newModel5.h5")
done = False
# game = episode
for i in range(games):
    #Reset the state at the start of each new game
    state = env.reset()
    state = np.reshape(state, [1, (env.observation_space.shape[0])])
    # This is each frame of the game we want to get a score of 200 for every frame you get 1 point
    for frame in range(maxScore):
        currentScore = 0
        complete = False
        #Renders the cartpole environment
        env.render()
        #Get the next action from the current state
        action = agent.nextAction(state)
        #Preform the action selected and move to the next state, observation is the next state you are going too
        observation, reward, done, extraInfo = env.step(action)
        #postive reward if the pole was not dropped
        if not done:
            reward = reward
        #negative reward if pole was dropped
        else:
            reward = -10
        observation = np.reshape(observation, [1, (env.observation_space.shape[0])])
        #Save the previouses states action, reward, future state and reward to memory
        agent.pushToMemory(state, action, reward, observation, done)
        #For the next frame make the current state the next state
        state = observation
        #If the game is done
        if done:
            currentScore = frame
            print("Episode: ",i," Score: ", frame," Epsilon: ", agent.EPS)
            break
        complete = True
    if done:
        scoreList.append(currentScore)
    agent.train(batchSize)
    #if i % 10 == 0:
     #agent.save("./newModel5.h5")
    if(complete):
        print("Episode: ",i," Score: ", maxScore," Epsilon: ", agent.EPS)
        scoreList.append(maxScore)
plt.plot(scoreList)
plt.ylabel('Frames Alive')
#plt.savefig('results2.png')
plt.show()
