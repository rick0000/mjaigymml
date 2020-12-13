import numpy as np


class Action2Agent(object):
    """[summary]
    """
    
    def predict_proba(self, states:np.array):
        """
        docstring
        """
        raise NotImplementedError


    def update_supervised(self, states, actions):
        """
        docstring
        """
        raise NotImplementedError

    
    def update_reinforce(self, states, actions, rewards):
        """
        docstring
        """
        raise NotImplementedError