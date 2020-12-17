import torch

class RBM():

    def __init__(self, n_visible, n_hidden, device, lr = 0.01):

        self.W = torch.randn(n_visible, n_hidden).to(device) # Weight Matrix
        self.a = torch.randn(1, n_hidden).to(device) # hideen layer bias
        self.b = torch.randn(1, n_visible).to(device) # visible layer bias
        self.device = device
        self.lr = lr

    def calc_hidden(self, x):
        '''
        calculate hidden probabilities and activations from visible layer nodes
        '''
        wx = torch.mm(x, self.W)
        activation = wx + self.a.expand_as(wx)
        prob_h = torch.sigmoid(activation)
        return prob_h, torch.bernoulli(prob_h)

    def calc_visible(self, x):
        '''
        calculate visible probabilities and activations from hidden layer nodes
        '''
        wx = torch.mm(x, self.W.t())
        activation = wx + self.b.expand_as(wx)
        prob_v = torch.sigmoid(activation)
        return prob_v, torch.bernoulli(prob_v)

    def cont_div(self, x):
        '''
        Apply Contrastive Divergence on input
        x : input vector for RBM    
        '''
        batch_size = x.shape[0]
        x = x.to(self.device)
        init_hidden_prob, init_hidden_activatons = self.calc_hidden(x)
        init_assoc = torch.mm(x.t(), init_hidden_prob)

        hidden_activations = init_hidden_activatons

        for k in range(10):
            visible_prob, visible_activation = self.calc_visible(hidden_activations)
            #visible_activation[x < 0] = x[x < 0]
            hidden_prob, hidden_activations = self.calc_hidden(visible_prob)

        final_hidden_prob = hidden_prob
        visible_prob, visible_activation = self.calc_visible(hidden_activations)
        #visible_activation[x < 0] = x[x < 0]

        final_assoc = torch.mm(visible_prob.t(), final_hidden_prob)
        
        delta_W = init_assoc - final_assoc
        delta_a = torch.sum(init_hidden_prob - final_hidden_prob, dim = 0)
        delta_b = torch.sum(x - visible_activation, dim = 0)

        self.W += self.lr*delta_W
        self.a += self.lr*delta_a
        self.b += self.lr*delta_b

        return visible_activation

    def infer(self, x):
        x = x.to(self.device)
        init_hidden_prob, init_hidden_activatons = self.calc_hidden(x)
        hidden_activations = init_hidden_activatons
        _, visible_activation = self.calc_visible(hidden_activations)
        return visible_activation
