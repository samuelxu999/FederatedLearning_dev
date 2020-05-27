import time
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities import TypesUtil, FileUtil
from wrapper_pyca import Crypto_Hash
from Index_Token import IndexToken 

logger = logging.getLogger(__name__)
# indextoken_logger = logging.getLogger("Index_Token")
# indextoken_logger.setLevel(logging.INFO)

# -------------------- Smart contract configuration -------------------
addr_list = "./addr_list.json"
http_provider = "http://localhost:8042"
contract_addr = IndexToken.getAddress('HashModelToken', addr_list)
contract_config = "./contracts/IndexToken.json"

# new IndexToken object
mytoken=IndexToken(http_provider, contract_addr, contract_config)

# tx commit timeout.
TX_TIMEOUT = 30

# Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class ModelUtils(object):

    @staticmethod
    def load_model(model_path, isShowInfo=False):
        device = torch.device("cpu")
        model = Net().to(device)
        if( os.path.isfile(model_path) ):
            model.load_state_dict(torch.load(model_path))
            model.eval()
        else:
            logger.info("{} is not existed. Use default parameters in Net.\n".format(model_path) )

        if( isShowInfo ):
    	    # Print model's state_dict
    	    logger.info("Model's state_dict:")
    	    for param_tensor in model.state_dict():
    	        logger.info("%s \t %s", param_tensor, model.state_dict()[param_tensor].size())
    	        # logger.info("%s \t %s", param_tensor, model.state_dict()[param_tensor])
    	    logger.info("")
        return model

    @staticmethod
    def evaluate_model(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
                pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        accuracy = 100.0 * correct / len(test_loader.dataset)
        logger.info(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss, correct, len(test_loader.dataset), accuracy
            )
        )

    @staticmethod
    def hash_model(model):
    	'''
    	Generate hash value of model data (tensor-->numpy-->string)

        Args:
            model: tensor model object

        Returns:
            Binary hash value
    	'''
    	str_model=[]
    	# For each model's state_dict to get str_model
    	logger.info("For each model's state_dict to get str_model...\n")
    	for param_tensor in model.state_dict():
    		# conver to numpy array
    		value_np = model.state_dict()[param_tensor].numpy()
    		# conver to string, which is used for hash function
    		str_model.append([param_tensor, str(value_np)])

    	# convert string to byte before hash operation
    	bytes_block = TypesUtil.string_to_bytes(str(str_model))

    	# generate hash value based on byte string
    	hash_value = Crypto_Hash.generate_hash(bytes_block)

    	return hash_value

class EtherUtils(object):
    @staticmethod
    def verify_hashmodel(model_name):
        # 1) Load model from file
        # model_name = "mnist_cnn.pt"
        ls_time_exec = []
        start_time=time.time()
        model=ModelUtils.load_model(model_name)
        ls_time_exec.append( format( time.time()-start_time, '.3f' ) ) 

        # 2) Calculate hash value of model
        start_time=time.time()
        hash_value=ModelUtils.hash_model(model)
        model=ModelUtils.load_model(model_name)
        ls_time_exec.append( format( time.time()-start_time, '.3f' ) ) 

        model_hash={}
        model_hash[model_name]=str(hash_value)

        # display contract information
        mytoken.Show_ContractInfo()

        # 3) Read token data using call
        start_time=time.time()
        token_data=mytoken.getIndexToken(model_name)
        IndexToken.print_tokendata(token_data)
        model=ModelUtils.load_model(model_name)
        ls_time_exec.append( format( time.time()-start_time, '.3f' ) ) 

        # Prepare log messgae
        str_time_exec=" ".join(ls_time_exec)
        FileUtil.save_testlog('test_results', 'exec_verify_hashmodel_ethereum.log', str_time_exec)

        # 4) return verify hash model result
        return model_hash[model_name]==token_data[1]

    @staticmethod
    def tx_evaluate(model_name):
        # calculate hash value for model
        # model_name = "mnist_cnn.pt"
        model=ModelUtils.load_model(model_name)
        hash_value=ModelUtils.hash_model(model)

        # evaluate tx committed time
        token_data=mytoken.getIndexToken(model_name)
        original_id = token_data[0] 

        logger.info("tx hashed model: {} to blockchain...\n".format(model_name)) 
        tx_time = 0.0
        start_time=time.time()
        mytoken.setIndexToken(model_name, str(hash_value))
        while(True):
            token_data=mytoken.getIndexToken(model_name)
            new_id = token_data[0]
            if(new_id > original_id ):
                IndexToken.print_tokendata(token_data)
                break
            time.sleep(0.1)
            tx_time +=0.1
            if(tx_time>=TX_TIMEOUT):
                logger.info("Timeout, tx commit fail.") 
                return False

        exec_time=time.time()-start_time
        logger.info("tx committed time: {:.3f}\n".format(exec_time, '.3f')) 
        FileUtil.save_testlog('test_results', 'exec_tx_commit_ethereum.log', format(exec_time, '.3f'))

        return True

