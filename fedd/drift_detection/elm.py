import numpy as np

class ELMRegressor:
    def __init__(self, n_hidden=10, random_state=42):
        """
        Inicializa a Extreme Learning Machine.
        O artigo utiliza h=10 neurônios na camada oculta por padrão.
        """
        self.n_hidden = n_hidden
        self.random_state = random_state
        
        self.weights_in = None
        self.bias_in = None
        self.weights_out = None

    def _sigmoid(self, x):
        """
        Função de ativação sigmoide, definida como padrão no artigo.
        Utiliza np.clip para evitar avisos de 'overflow' matemático.
        """
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))

    def fit(self, X, y):
        """
        Treina o modelo ELM analiticamente usando a matriz inversa.
        X: array 2D com os dados passados (lags)
        y: array 1D com os valores alvo (target)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape

        # 1. Random hidden weights and biases
        self.weights_in = np.random.normal(size=(n_features, self.n_hidden))
        self.bias_in = np.random.normal(size=(self.n_hidden,))

        # 2. Hidden layer output
        linear_output = np.dot(X, self.weights_in) + self.bias_in
        H = self._sigmoid(linear_output)

        # 3. Output weights via pseudo-inverse
        H_pinv = np.linalg.pinv(H)
        self.weights_out = np.dot(H_pinv, y)

    def predict(self, X):
        """
        Gera as previsões (y_pred) para as instâncias de entrada X.
        """
        if self.weights_out is None:
            raise ValueError("O modelo ELM precisa ser treinado com .fit() antes de fazer previsões.")
            
        # Hidden layer forward pass
        linear_output = np.dot(X, self.weights_in) + self.bias_in
        H = self._sigmoid(linear_output)
        
        # Output layer
        return np.dot(H, self.weights_out)