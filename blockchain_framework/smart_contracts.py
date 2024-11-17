from block import Block

class SmartContracts:
    def __init__(self, blockchain):
        self.blockchain = blockchain

    def schedule_training(self, model_params, model_results, zk_proof):
        latest_block = self.blockchain.get_latest_block()
        new_block = Block(
            index=latest_block.index + 1,
            previous_hash=latest_block.hash,
            model_params=model_params,
            model_results=model_results,
            zk_proof=[zk_proof[0], zk_proof[1], zk_proof[2]]
        )
        return new_block

    def verify_protocol_compliance(self, zk_proof, zkp_instance):
        return zkp_instance.verify_proof(zk_proof[0], zk_proof[1], zk_proof[2])

    def update_protocol(self, consensus, new_block, device_signature, device_id):
        if consensus.reach_consensus(self.blockchain, new_block, device_signature, device_id, consensus.authorized_devices[device_id]["public_key"]):
            return True
        else:
            return False
