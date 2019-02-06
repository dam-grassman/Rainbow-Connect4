from collections import deque
import random
#import atari_py
import cv2
import torch
import numpy as np

from gym_connect4.envs.connect4_env import connect4Env


class Env():
  def __init__(self, args):
    self.device = args.device

    self.connect4 = connect4Env()
    actions = self.connect4.getActionSet()
    self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
    self.background = construct_background()


    self.state_buffer = deque([], maxlen=args.history_length)
    self.training = True  # Consistent with model training mode
    

  def _get_state(self):

    state = self.connect4._get_state()
    state = np.mean(state,axis=2)
    return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

  def _reset_buffer(self):

    self.state_buffer.append(torch.zeros(self.connect4.shape, device=self.device))

  def reset(self):

    self._reset_buffer()
    self.connect4.reset_game() 

    #for _ in range(random.randrange(30)):
    #   self.connect4.act(0) 
    #   if self.connect4.game.game_over:
    #     self.connect4.reset_game()

    observation = self._get_state()
    self.state_buffer.append(observation)
    return torch.stack(list(self.state_buffer), 0)

  def step(self, action):

    reward, done = 0, False
    available = self.connect4.game.get_choice()

    if available[action]==True : 
      reward = self.connect4._step(self.actions.get(action))
      done = self.connect4.game.game_over
    else :
      reward = -1
      done = True 

    #print('ACTION',self.actions.get(action), available[action], done, reward)
    observation = self._get_state()
    self.state_buffer.append(observation)
    # Return state, reward, done
    return torch.stack(list(self.state_buffer), 0), reward, done

  # Uses loss of life as terminal signal
  def train(self):
    self.training = True

  # Uses standard terminal signal
  def eval(self):
    self.training = False

  def action_space(self):
    return len(self.actions)

  def color(self, board):
    image = np.concatenate([board, board, board]).reshape((board.shape[0], board.shape[1],3))
    for i in range(board.shape[0]):
      for j in range(board.shape[1]):
        if image[i,j,0]==1 :
          image[i,j,:] = [0,0,256]
        elif image[i,j,0]==2 :
          image[i,j,:] = [0,256,0]
        else :
          image[i,j,:] = [255,255,255]
    return image

  def render(self):
    img = self.connect4._get_state(small=True)
    img = boad_to_image(img,10)
    for i in range(840):
      for j in range(840):
        if self.background[i,j,0] == 255 :
          img[i,j,:] = [255,0,0]

    cv2.imshow('screen', img[::-1, ::-1, :])#[:, :, ::-1]
    cv2.waitKey(10)

  def close(self):
    cv2.destroyAllWindows()

def boad_to_image(board, dim = 1):
  img = np.zeros((84*dim,84*dim,3))
  #print(board.shape)
  for i in range(board.shape[0]):
    for j in range(board.shape[1]):
      if board[i,j]==1 : 
        img[i*12*dim:(i+1)*12*dim,j*12*dim:(j+1)*12*dim,:] = np.ones((12*dim,12*dim,3))* [0,0,256]
      elif board[i,j]==2 : 
        img[i*12*dim:(i+1)*12*dim,j*12*dim:(j+1)*12*dim,:] = np.ones((12*dim,12*dim,3))* [0,256,0]
      else: 
        img[i*12*dim:(i+1)*12*dim,j*12*dim:(j+1)*12*dim,:] = np.ones((12*dim,12*dim,3))* [256,256,256]
  return img


def construct_background():
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage import draw
    arr = np.ones((840, 840,3))*[255,0,0]

    center = []
    lenght = 840
    for i in range(7):
        for j in range(7):
            x = int(lenght/14+i/7*lenght)
            y = int(lenght/14+j/7*lenght)
            center.append((x,y))

    for x,y in center :
        # Create stroke-many circles centered at radius. 
        #rr, cc, d = draw.circle_perimeter_aa(100, 100, radius=80+delta, shape=arr.shape)
        l = draw.circle(x, y, radius=lenght/18)
        for i in range(l[0].shape[0]):
            arr[l[0][i], l[1][i],:] = [0,0,0]
    return arr