import requests

class ApiClient:
    def __init__(self, base_url: str, timeout: float = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self):
        return requests.get(f"{self.base_url}/health", timeout=self.timeout)

    def metadata(self):
        return requests.get(f"{self.base_url}/metadata", timeout=self.timeout)

    def get_client(self, sk_id: int):
        return requests.get(f"{self.base_url}/client/{sk_id}", timeout=self.timeout)

    def predict(self, features: dict):
        return requests.post(f"{self.base_url}/predict", json=features, timeout=self.timeout)

    def explain(self, features: dict, top_n: int = 10):
        return requests.post(f"{self.base_url}/explain", params={"top_n": top_n}, json=features, timeout=self.timeout)

    def population_sample(self, n: int = 2000):
        return requests.get(f"{self.base_url}/population/sample", params={"n": n}, timeout=self.timeout)

