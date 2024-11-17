from Crypto.Hash import SHA256
from Crypto.Random import random
from Crypto.Util.number import getPrime

class ZeroKnowledgeProof:
    def __init__(self, secret_value):
        self.secret_value = secret_value  # This is the 'x'
        self.generator = 2
        self.prime = getPrime(512)
        self.public_value = pow(self.generator, self.secret_value, self.prime)  # Public value (g^x mod p), simplified example

    def generate_proof(self):
        r = random.randint(1, self.prime - 1) # Step 1: Generate random value r
        commitment = pow(self.generator, r, self.prime)  # g^r mod p
        h = SHA256.new(str(commitment).encode()) # Step 2: Hash commitment to generate the challenge (Fiat-Shamir heuristic)
        challenge = int(h.hexdigest(), 16) % self.prime
        response = (r + challenge * self.secret_value) % self.prime
        return commitment, challenge, response

    def verify_proof(self, commitment, challenge, response):
        # Compute the expected commitment using challenge and response
        expected_commitment = (pow(self.generator, response, self.prime) * pow(self.public_value, self.prime - 1 - challenge, self.prime)) % self.prime
        return expected_commitment == commitment