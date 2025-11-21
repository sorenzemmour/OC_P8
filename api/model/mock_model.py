class MockModel:
    def predict_proba(self, X):
        # Retourne une probabilité fixe pour les tests
        import numpy as np
        return np.array([[0.1, 0.9]])
