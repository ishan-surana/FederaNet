import time
from Crypto.PublicKey import ECC
from Crypto.Hash import SHA256
from Crypto.Signature import DSS

class Consensus:
    def __init__(self):
        # Store authorized devices with their public keys and online status
        self.authorized_devices = {}

    def register_device(self, device_id, public_key):
        self.authorized_devices[device_id] = {
            "public_key": public_key,
            "online": False,  # Initially set to offline
            "last_seen": None  # Track the last time the device was seen
        }

    def update_device_status(self, device_id, status):
        if device_id in self.authorized_devices:
            self.authorized_devices[device_id]["online"] = status
            self.authorized_devices[device_id]["last_seen"] = time.time()

    def check_device_status(self, device_id):
        return self.authorized_devices[device_id]["online"] if device_id in self.authorized_devices else False

    def get_online_devices(self):
        return {device_id: device for device_id, device in self.authorized_devices.items() if device["online"]}

    def sign_block(self, private_key, model_update):
        key = ECC.import_key(bytes.fromhex(private_key)) # Load the private key
        h = SHA256.new(model_update.encode()) # Hash the model update
        signer = DSS.new(key, 'fips-186-3') # Sign the hash with the private key
        signature = signer.sign(h)
        return signature.hex()

    def validate_update(self, block, signature, device_id, public_key):
        if not self.check_device_status(device_id):
            return False
        key = ECC.import_key(bytes.fromhex(public_key)) # Load the public key
        h = SHA256.new(block.model_update.encode()) # Hash the block's model update
        verifier = DSS.new(key, 'fips-186-3') # Verify the signature
        try:
            verifier.verify(h, bytes.fromhex(signature))
            return True
        except ValueError:
            return False

    def reach_consensus(self, blockchain, block, device_signature, device_id, device_public_key):
        if self.validate_update(block, device_signature, device_id, device_public_key):
            return blockchain.add_block(block)
        return False
