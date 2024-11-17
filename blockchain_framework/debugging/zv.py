from Crypto.Random import random
from Crypto.Util.number import getPrime
from hashlib import sha256

# Parameters
g = 2  # Generator
p = getPrime(512)  # Large prime number
secret_value = 42  # Secret value

# Step 1: Generate random value r
r = random.randint(1, p - 1)

# Step 2: Compute the commitment
commitment = pow(g, r, p)  # g^r mod p
print(f"Commitment: {commitment}")

# Step 3: Create a challenge using the Fiat-Shamir heuristic (hash of commitment)
challenge = int(sha256(str(commitment).encode()).hexdigest(), 16) % p
print(f"Challenge: {challenge}")

# Step 4: Compute the response
response = (r + challenge * secret_value) % p
print(f"Response: {response}")

# Step 5: Compute the public value
public_value = pow(g, secret_value, p)  # g^secret_value mod p
print(f"Public Value: {public_value}")

# Step 6: Compute the modular inverse of public_value^challenge using Fermat's Little Theorem
# Modular inverse: public_value^-challenge mod p == public_value^(p-1-challenge) mod p
inverse = pow(public_value, p - 1 - challenge, p)

# Step 7: Compute the expected commitment using the corrected formula
expected_commitment = (pow(g, response, p) * inverse) % p
print(f"Expected commitment: {expected_commitment}")

# Step 8: Check if the received commitment matches the expected one
if commitment == expected_commitment:
    print("Proof is valid")
else:
    print("Proof is invalid")