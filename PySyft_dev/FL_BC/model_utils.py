import time
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms

from utilities import TypesUtil, FileUtil
from wrapper_pyca import Crypto_Hash
from Index_Token import IndexToken 
from Tender_RPC import Tender_RPC
from Micro_RPC import Micro_RPC 

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
TX_TIMEOUT = 60

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
        '''
        Load model data given model file (*.pt)

        Args:
            model_path: model file path
            isShowInfo: display model information or not
        Returns:
            model: tensor model object
        '''
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
        '''
        Evaluate model and output loss and accuracy

        Args:
            model: tensor model object
            device: device selection for torch
            test_loader: test data loader object
        Returns:
            return accuracy and test_loss
        '''
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
        return accuracy, test_loss

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

class DatasetUtils(object):

    @staticmethod
    def load_dataset(dataset_path, training):
        '''
        Load MINST datset given dataset path

        Args:
            dataset_path: dataset folder path
            training: load training data or testing data
        Returns:
            model: tensor model object
        '''
        _dataset = datasets.MNIST(
            root=dataset_path,
            train=training,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        return _dataset

    @staticmethod
    def hash_dataset(dataset, keep_labels):
        '''
        Generate hash value of loaded dataset (tensor-->numpy-->string)

        Args:
            dataset: dataset object

        Returns:
            Binary hash value
        '''
        str_dataset=[]
        # For each model's state_dict to get str_dataset
        logger.info("For each dataset's touple to get str_dataset...\n")
        i=0
        for data_tensor, target_tensor in dataset:
            if(target_tensor in keep_labels):
                # conver to touple [data, target]
                value_np = [data_tensor, target_tensor]
                # conver to string, which is used for hash function
                str_dataset.append( str(value_np) )

        # convert string to byte before hash operation
        bytes_block = TypesUtil.string_to_bytes(str(str_dataset))

        # generate hash value based on byte string
        hash_value = Crypto_Hash.generate_hash(bytes_block)

        return hash_value


class EtherUtils(object):
    @staticmethod
    def verify_hashmodel(model_name):
        '''
        Verify model hash value by querying blockchain

        Args:
            model_name: model file
        Returns:
            Verified result: True or False
        '''
        # 1) Load model from file
        ls_time_exec = []
        start_time=time.time()
        model=ModelUtils.load_model(model_name)
        ls_time_exec.append( format( time.time()-start_time, '.3f' ) ) 

        # 2) Calculate hash value of model
        start_time=time.time()
        hash_value=ModelUtils.hash_model(model)
        ls_time_exec.append( format( time.time()-start_time, '.3f' ) ) 

        model_hash={}
        model_hash[model_name]=str(hash_value)

        # -------- display contract information -------------
        mytoken.Show_ContractInfo()

        # 3) Read token data using call
        start_time=time.time()
        token_data=mytoken.getIndexToken(model_name)
        IndexToken.print_tokendata(token_data)
        ls_time_exec.append( format( time.time()-start_time, '.3f' ) ) 

        # Prepare log messgae
        str_time_exec=" ".join(ls_time_exec)
        FileUtil.save_testlog('test_results', 'exec_verify_hashmodel_ethereum.log', str_time_exec)

        # 4) return verify hash model result
        return model_hash[model_name]==token_data[1]

    @staticmethod
    def tx_evaluate(model_name):
        '''
        Launch tx and evaluate tx committed time

        Args:
            model_name: model file
        Returns:
            tx committed reulst
        '''
        # 1) Load model from file
        model=ModelUtils.load_model(model_name)
        # 2) calculate hash value for model
        hash_value=ModelUtils.hash_model(model)

        # 3) evaluate tx committed time
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

class TenderUtils(object):
    @staticmethod
    def verify_hashmodel(model_name):
        '''
        Verify model hash value by querying blockchain

        Args:
            model_name: model file
        Returns:
            Verified result: True or False
        '''
        # 1) Load model from file
        ls_time_exec = []
        start_time=time.time()
        model=ModelUtils.load_model(model_name)
        ls_time_exec.append( format( time.time()-start_time, '.3f' ) ) 

        # 2) Calculate hash value of model
        start_time=time.time()
        hash_value=ModelUtils.hash_model(model)
        ls_time_exec.append( format( time.time()-start_time, '.3f' ) ) 

        model_hash={}
        model_hash[model_name]=str(hash_value)

        # 3) Read token data using call
        query_json = {}
        query_json['data']='"' + model_name +'"'
        start_time=time.time()
        query_ret=Tender_RPC.abci_query(query_json)
        ls_time_exec.append( format( time.time()-start_time, '.3f' ) ) 

        # -------- parse value from response and display it ------------
        key_str=query_ret['result']['response']['key']
        value_str=query_ret['result']['response']['value']
        logger.info("Fetched model hash value:")
        logger.info("model: {}".format(TypesUtil.base64_to_ascii(key_str)) )
        if( value_str!= None):
            query_hash_value = TypesUtil.hex_to_string(TypesUtil.base64_to_ascii(value_str))
        else:
            query_hash_value = ''
        logger.info("value: {}".format(query_hash_value))
        
        # Prepare log messgae
        str_time_exec=" ".join(ls_time_exec)
        FileUtil.save_testlog('test_results', 'exec_verify_hashmodel_tendermint.log', str_time_exec)

        # 4) return verify hash model result
        return model_hash[model_name]==str(query_hash_value)

    @staticmethod
    def tx_evaluate(model_name):
        '''
        Launch tx and evaluate tx committed time

        Args:
            model_name: model file
        Returns:
            tx committed reulst
        '''
        # 1) Load model from file
        model=ModelUtils.load_model(model_name)

        # 2) calculate hash value for model
        hash_value=ModelUtils.hash_model(model)

        # 3) evaluate tx committed time
        start_time=time.time()
        logger.info("tx hashed model: {} to blockchain...\n".format(model_name)) 
        # -------- prepare parameter for tx ------------
        tx_json = {}
        key_str = model_name
        value_str = TypesUtil.string_to_hex(hash_value)
        tx_data = key_str + "=" + value_str 
        # --------- build parameter string: tx=? --------
        tx_json['tx']='"' + tx_data +'"' 
        tx_ret=Tender_RPC.broadcast_tx_commit(tx_json)
        exec_time=time.time()-start_time
        logger.info("tx committed time: {:.3f}\n".format(exec_time, '.3f')) 
        FileUtil.save_testlog('test_results', 'exec_tx_commit_tendermint.log', format(exec_time, '.3f'))

        return tx_ret

class MicroUtils(object):
    @staticmethod
    def get_info(target_address="0.0.0.0:8080"):
        json_response = Micro_RPC.micro_info(target_address)
        for _item, _value in json_response.items():
            logger.info("{}    {}\n".format(_item, _value))

    @staticmethod
    def verify_hashmodel(model_name, target_address="0.0.0.0:8080"):
        '''
        Verify model hash value by querying blockchain

        Args:
            model_name: model file
        Returns:
            Verified result: True or False
        '''
        # 1) Load model from file
        ls_time_exec = []
        start_time=time.time()
        model=ModelUtils.load_model(model_name)
        ls_time_exec.append( format( time.time()-start_time, '.3f' ) ) 

        # 2) Calculate hash value of model
        start_time=time.time()
        hash_value=ModelUtils.hash_model(model)
        ls_time_exec.append( format( time.time()-start_time, '.3f' ) ) 

        model_hash={}
        model_hash[model_name]=str(hash_value)

        # 3) Read token data using call
        query_json = {}
        value_str = str(hash_value)
        query_json[model_name]=value_str
        # print(query_json)
        start_time=time.time()
        query_ret=Micro_RPC.tx_query(target_address, query_json)
        ls_time_exec.append( format( time.time()-start_time, '.3f' ) ) 

        # # -------- parse value from response and display it ------------
        verify_result = False
        # print(query_ret)
        logger.info("Fetched model hash value:")
        if(query_ret!={}):
            tx_json = TypesUtil.string_to_json(query_ret)
            for _name, _value in tx_json.items():
                logger.info("model: {}".format(_name) )
                logger.info("value: {}".format(_value) )
            verify_result = True

        
        # Prepare log messgae
        str_time_exec=" ".join(ls_time_exec)
        FileUtil.save_testlog('test_results', 'exec_verify_hashmodel_microchain.log', str_time_exec)

        # 4) return verify hash model result
        return verify_result

    @staticmethod
    def tx_evaluate(model_name, target_address="0.0.0.0:8080"):
        '''
        Launch tx and evaluate tx committed time

        Args:
            model_name: model file
        Returns:
            tx committed reulst
        '''
        # 1) Load model from file
        model=ModelUtils.load_model(model_name)

        # 2) calculate hash value for model
        hash_value=ModelUtils.hash_model(model)

        # 3) evaluate tx committed time
        tx_time = 0.0
        start_time=time.time()
        logger.info("tx hashed model: {} to blockchain...\n".format(model_name)) 
        # -------- prepare transaction data ------------
        tx_json = {}
        # value_str = TypesUtil.string_to_hex(hash_value)
        value_str = str(hash_value)
        tx_json[model_name]=value_str
        # print(tx_json)
        tx_ret=Micro_RPC.broadcast_tx_commit(target_address, tx_json)

        while(True):
            query_ret=Micro_RPC.tx_query(target_address, tx_json)

            if( query_ret!={} ):
                break
            time.sleep(0.5)
            tx_time +=0.5
            if(tx_time>=TX_TIMEOUT):
                logger.info("Timeout, tx commit fail.") 
                return False

        exec_time=time.time()-start_time
        logger.info("tx committed time: {:.3f}\n".format(exec_time, '.3f')) 
        FileUtil.save_testlog('test_results', 'exec_tx_commit_microchain.log', format(exec_time, '.3f'))

        # return tx_ret
        return tx_ret