from flask import Flask, jsonify, request, render_template, redirect, url_for, render_template_string
from blockchain import Blockchain
from consensus import Consensus
from zkp import ZeroKnowledgeProof
from smart_contracts import SmartContracts
import pickle
import hashlib
from Crypto.PublicKey import ECC

app = Flask(__name__)

# Initialize blockchain, consensus mechanism, and smart contracts
# Check if a pickle of the blockchain exists
def load_blockchain():
    try:
        with open("blockchain.pkl", "rb") as f:
            blockchain = pickle.load(f)
    except FileNotFoundError:
        blockchain = Blockchain()
        with open("blockchain.pkl", "wb") as f:
            pickle.dump(blockchain, f)
    return blockchain

def save_blockchain(blockchain):
    with open("blockchain.pkl", "wb") as f:
        pickle.dump(blockchain, f)

blockchain = load_blockchain()
consensus = Consensus()
smart_contract = SmartContracts(blockchain)

# Register 3 static devices for PoA consensus
# consensus.register_device("Device1", "PublicKey1")
# consensus.register_device("Device2", "PublicKey2")
# consensus.register_device("Device3", "PublicKey3")

private_key1 = ECC.generate(curve='P-256')  # Private key stays on device
public_key1 = private_key1.public_key()      # Public key to be sent to the Flask app
# Export private key securely on the device for future use
device_private_key1 = private_key1.export_key(format='DER').hex()  # Store this securely
device_public_key1 = public_key1.export_key(format='DER').hex()
consensus.register_device("Device1", device_public_key1)

private_key2 = ECC.generate(curve='P-256')  # Private key stays on device
public_key2 = private_key2.public_key()
device_private_key2 = private_key2.export_key(format='DER').hex()  # Store this securely
device_public_key2 = public_key2.export_key(format='DER').hex()
consensus.register_device("Device2", device_public_key2)

private_key3 = ECC.generate(curve='P-256')  # Private key stays on device
public_key3 = private_key3.public_key()
device_private_key3 = private_key3.export_key(format='DER').hex()  # Store this securely
device_public_key3 = public_key3.export_key(format='DER').hex()
consensus.register_device("Device3", device_public_key3)

private_keys = {
    "Device1": device_private_key1,
    "Device2": device_private_key2,
    "Device3": device_private_key3
}

# Root endpoint
@app.route('/')
def index():
    return render_template('index.html')

# Device registration form and logic
@app.route('/register_device', methods=['GET', 'POST'])
def register_device():
    if request.method == 'POST':
        device_id = request.form.get('device_id')
        public_key = request.form.get('public_key')
        if device_id in consensus.authorized_devices:
            return render_template('register_device.html', error="Device already registered.")
        consensus.register_device(device_id, public_key)
        return redirect(url_for('index'))
    return render_template('register_device.html', error="")

# Device heartbeat to update online status
@app.route('/heartbeat', methods=['GET', 'POST'])
def heartbeat():
    if request.method == 'POST':
        device_id = request.form.get('device_id')
        status = True if request.form['status'] == 'online' else False
        consensus.update_device_status(device_id, status)
        return redirect(url_for('index'))
    devices = consensus.authorized_devices.keys()
    return render_template('heartbeat.html', devices=devices)

# Check device status
@app.route('/device_status/<device_id>', methods=['GET'])
def device_status(device_id):
    status = consensus.check_device_status(device_id)
    return jsonify({"device_id": device_id, "online": status}), 200

# Submit model update
@app.route('/submit_update', methods=['GET', 'POST'])
def submit_update():
    if request.method == 'POST':
        model_params = request.form.get('model_params')
        model_results = request.form.get('model_results')
        device_id = request.form.get('device_id')
        model_update = hashlib.sha256((model_params + model_results).encode()).hexdigest()
        
        if not consensus.check_device_status(device_id):
            return jsonify({"error": f"Device {device_id} is offline. Cannot accept update."}), 400

        device_signature = consensus.sign_block(private_keys[device_id], model_update)

        zkp = ZeroKnowledgeProof(secret_value=42)
        commitment, challenge, response = zkp.generate_proof()

        new_block = smart_contract.schedule_training(model_params, model_results, [commitment, challenge, response])

        if smart_contract.verify_protocol_compliance([commitment, challenge, response], zkp):
            if smart_contract.update_protocol(consensus, new_block, device_signature, device_id):
                # save the blockchain to a pickle file
                save_blockchain(blockchain)
                return jsonify({"message": "Model update added to blockchain", "block": new_block.__dict__}), 201
            else:
                return jsonify({"error": "Consensus failed. Block not added."}), 400
        else:
            return jsonify({"error": "ZKP verification failed."}), 400

    devices = consensus.authorized_devices.keys()  # Get device list for the dropdown
    return render_template('submit_update.html', devices=devices)

# Get blockchain devices status
@app.route('/blockchain_devices', methods=['GET'])
def blockchain_devices():
    chain = blockchain.get_chain()
    device_activity = [device["online"] for device in consensus.authorized_devices.values()]
    return render_template('blockchain_devices.html', chain=chain, device_activity=device_activity)
    # # devices = str(list(consensus.authorized_devices.keys())).replace("'", "")
    # # how do i return devices so that they dont cause an error in the template?
    # # for example, if i return devices directly, i get output like this
    # """
    #  let devices = "[
    # {
    #     &#34;last_seen&#34;: null,
    #     &#34;online&#34;: false,
    #     &#34;public_key&#34;: &#34;PublicKey1&#34;
    # },
    # {
    #     &#34;last_seen&#34;: null,
    #     &#34;online&#34;: false,
    # """
    # # this is because the template is interpreting the string as html entities.
    # # how do i use safe to prevent this?
    # # ans: use safe filter in the template, like this: {{ devices | safe }}
    # # import json
    # # # how do i return json of chain and devices? chain = json.dumps(chain, default=lambda o: o.__dict__, sort_keys=True, indent=4)
    # # return render_template('blockchain_devices.html', chain=json.dumps(chain, default=lambda o: o.__dict__, sort_keys=True, indent=4), 
    # #     devices=json.dumps(devices, default=lambda o: o.__dict__, sort_keys=True, indent=4))

# Get blockchain
@app.route('/blockchain', methods=['GET'])
def get_blockchain():
    chain = blockchain.get_chain()
    return render_template_string("<pre>"+chain+"</pre>")

# if __name__ == '__main__':
#     app.run(port=80, debug=False)
