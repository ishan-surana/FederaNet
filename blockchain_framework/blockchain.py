from block import Block
import json

class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = Block(0, "0", "0", "0", [0, 0, 0])
        self.chain.append(genesis_block)

    def add_block(self, new_block):
        if self.verify_block(new_block, self.chain[-1]):
            self.chain.append(new_block)
            return True
        return False

    def verify_block(self, block, previous_block):
        if block.previous_hash != previous_block.hash:
            return False
        return True

    def get_latest_block(self):
        return self.chain[-1]

    def get_chain(self):
        # send blockchain in json format
        return json.dumps(self.chain, default=lambda o: o.__dict__, sort_keys=True, indent=4)