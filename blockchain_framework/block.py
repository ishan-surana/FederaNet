import hashlib
import time
import json

class Block:
    def __init__(self, index, previous_hash, model_params, model_results, zk_proof, timestamp=None):
        self.index = index
        self.previous_hash = previous_hash
        self.model_params = model_params
        self.model_results = model_results
        self.model_update = hashlib.sha256((model_params + model_results).encode()).hexdigest()
        self.zk_proof = zk_proof
        self.timestamp = timestamp or time.time()
        self.hash = self.compute_hash()

    def compute_hash(self):
        block_string = json.dumps(self.__dict__, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def __str__(self):
        return json.dumps(self.__dict__, indent=4)
