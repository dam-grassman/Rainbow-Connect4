import gym
from gym import error, spaces, utils
from gym.utils import seeding
#import torch

import numpy as np


class Grid:

	def __init__(self,n):
		self.board = np.zeros((n,n))#, dtype = 'int32')
		self.available = np.zeros(n)#, dtype = 'int32')
		self.color = {1: (0,0,255), 2:(0,255,0)}
		self.board =self.create_bord(n)
		self.column_length = n
		self.row_length = n
		self.winner = 0
		self.game_over = False


	def get_choice(self):
		return (self.available < self.row_length)

	def update_grid(self, player, choice):
		# suppose that the choice belongs to the available choices
		x,y = int(self.available[choice]),int(choice)
		self.board[x,y] = player
		self.available[y] += 1
		self.winner = self.is_winner(player)
		#if self.winner !=0:
			#print('Winner is:', self.winner)
			#print(self.board)
		self.is_game_over()


	def create_bord(self,n):
		board = np.zeros((n,n))#, dtype = 'int8')
		return board

	def is_winner(self, player):
		# Check horizontal locations for win
		for c in range(self.column_length-3):
			for r in range(self.row_length):
				if self.board[r,c] == player and self.board[r,c+1] == player and self.board[r,c+2] == player and self.board[r,c+3] == player:
					return player

		# Check vertical locations for win
		for c in range(self.column_length):
			for r in range(self.row_length-3):
				if self.board[r,c] == player and self.board[r+1,c] == player and self.board[r+2,c] == player and self.board[r+3,c] == player:
					return player

		# Check positively sloped diaganols
		for c in range(self.column_length-3):
			for r in range(self.row_length-3):
				if self.board[r,c] == player and self.board[r+1,c+1] == player and self.board[r+2,c+2] == player and self.board[r+3,c+3] == player:
					return player

		# Check negatively sloped diaganols
		for c in range(self.column_length-3):
			for r in range(3, self.row_length):
				if self.board[r,c] == player and self.board[r-1,c+1] == player and self.board[r-2,c+2] == player and self.board[r-3,c+3] == player:
					return player

		return 0

	def is_game_over(self):
		if self.winner != 0 :
			self.game_over=True


class connect4Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,n=7):
        self.game = Grid(n)
        self.shape = (self.game.row_length,self.game.column_length)
        self.SecondPlayer = SecondPlayer()
        self.round = 0
        self.n = n

    def _get_reward(self):
        """ Reward is given for XY. """
        #print(self.game.winner, self.game.game_over)
        if self.game.winner == 1:
            #print('AGENT WON')
            return 1
        elif self.game.winner == 2:
            #print('AGENT LOST')
            return -1
        else:
            return 0

    def _step(self, action):

        #print('Action from the agent :', action)

        self._take_action(action, player=1)
        if self.game.game_over==False:
        	self.simulate_second_player()

        return self._get_reward()

        #self.status = self.env.step()
        #reward = self._get_reward()
        #ob = self.env.getState()
        #episode_over = self.status != hfo_py.IN_GAME

        #return ob, reward, episode_over, {}
		

    def _reset(self):
        pass

    def _render(self, mode='human', close=False):
        pass

    def _take_action(self, action, player=1):
        self.game.update_grid(player, action)

    def _get_state(self,small =False):
        if small :
        	return self.game.board
        
        return boad_to_image(self.game.board)


    def simulate_second_player(self):
    	action = self.SecondPlayer.strategy(board =self.game.board,available=self.game.get_choice())
    	self._take_action(action, player=2)

    def getScreenRGB(self):
    	pass

    def reset_game(self):
    	self.game = Grid(self.n)

    def getActionSet(self):
    	return [i for i in range(self.shape[0])]

    def act(self, i):
    	available = self.game.get_choice()
    	#print(available)
    	#print(available.nonzero())
    	#print(available.nonzero()[0])
    	#print( np.random.choice(available.nonzero()[0]))
    	return np.random.choice(available.nonzero()[0])


class SecondPlayer:

	def __init__(self):
		self.history =[]
		self.column_length=7
		self.row_length=7
		pass

	def counterattack(self, board, player, available):
		# Check horizontal locations for win
		for c in range(self.column_length-3):
			for r in range(self.row_length):
				if board[r,c] == player and board[r,c+1] == player and board[r,c+2] == player:
					if board[r,c+3] == 0 and available[c+3] == True :
						return c+3
					elif board[r,c-1] == 0 and available[c-1] == True :
						return c-1
					else :
						pass

		# Check vertical locations for win
		for c in range(self.column_length):
			for r in range(self.row_length-3):
				if board[r,c] == player and board[r+1,c] == player and board[r+2,c] == player:
					if board[r+3,c] == 0 and available[c] == True :
						return c

		# Check positively sloped diaganols
		for c in range(self.column_length-3):
			for r in range(self.row_length-3):
				if board[r,c] == player and board[r+1,c+1] == player and board[r+2,c+2] == player:
					if board[r+2,c+3] != 0 and available[c+3] :
						return c+3

		# Check negatively sloped diaganols
		for c in range(self.column_length-3):
			for r in range(3, self.row_length):
				if board[r,c] == player and board[r-1,c+1] == player and board[r-2,+2] == player:
					if board[r-2,c+3] != 0 and available[c+3] :
						return c+3
		return None

	def strategy(self, available, board):
		#print('SECDOND PLAYER')
		counter = self.counterattack(board, player=2,available=available)
		if counter != None : 
			return counter
		counter = self.counterattack(board, player=1,available=available)
		if counter != None : 
			return counter
		else :
			return np.random.choice(available.nonzero()[0])


def boad_to_image(board):
	img = np.zeros((84,84,3))
	#print(board.shape)
	for i in range(board.shape[0]):
		for j in range(board.shape[1]):
			if board[i,j]==1 : 
				img[i:i+12,j:j+12,:] = np.ones((12,12,3))* [0,0,256]
			elif board[i,j]==2 : 
				img[i:i+12,j:j+12,:] = np.ones((12,12,3))* [0,256,0]
			else: 
				img[i:i+12,j:j+12,:] = np.ones((12,12,3))* [256,256,256]
	return img


#python main.py --T-max=3000 --evaluation-interval=100 --learn-start=100 --batch-size=10  --target-update=200 --replay-frequency=2 --lr=0.006252 --hidden-size=10 --atoms=5 --evaluate --render