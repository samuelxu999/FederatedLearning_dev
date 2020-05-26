import time
import logging
import argparse
import sys
import os
import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms

import syft as sy
from syft.workers import websocket_client
from syft.frameworks.torch.fl import utils

from utilities import DatetimeUtil, TypesUtil, FileUtil
from wrapper_pyca import Crypto_Hash
from Index_Token import IndexToken 

LOG_INTERVAL = 25

logger = logging.getLogger("test")

#global variable
addr_list = "./addr_list.json"
http_provider = "http://localhost:8042"
contract_addr = IndexToken.getAddress('HashModelToken', addr_list)
contract_config = "./contracts/IndexToken.json"

#new ABACToken object
mytoken=IndexToken(http_provider, contract_addr, contract_config)


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


def define_and_get_arguments(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Run federated learning using websocket client workers."
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch size of the training")
    parser.add_argument(
        "--test_batch_size", type=int, default=128, help="batch size used for the test data"
    )
    parser.add_argument(
        "--training_rounds", type=int, default=40, help="number of federated learning rounds"
    )
    parser.add_argument(
        "--federate_after_n_batches",
        type=int,
        default=10,
        help="number of training steps performed on each remote worker before averaging",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--seed", type=int, default=1, help="seed used for randomization")
    parser.add_argument("--save_model", action="store_true", help="if set, model will be saved")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="if set, websocket client workers will be started in verbose mode",
    )
    parser.add_argument(
        "--localworkers",
        action="store_true",
        help="if set, use (localhost) websocket server workers, otherwize, connect to remote worker server."
    )
    args = parser.parse_args(args=args)
    return args

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

def test_model(model, device, test_loader):
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

def test_main():
    args = define_and_get_arguments()

    use_cuda = args.cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    logger.info("module setup...\n")
    model=load_model("mnist_cnn.pt", True)

    logger.info("test_loader setup...\n")
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs,
    )

    logger.info("test module...\n")
    test_model(model, device, test_loader)

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

def test_hashmodel():
    # calculate hash value for model
    model_name = "mnist_cnn.pt"
    model=load_model(model_name)
    hash_value=hash_model(model)
    # logger.info("{} \n".format(hash_value))

    model_hash={}
    model_hash[model_name]=str(hash_value)

    # display contract information
    mytoken.Show_ContractInfo()

    # Read token data using call
    token_data=mytoken.getIndexToken(model_name)
    IndexToken.print_tokendata(token_data)

    # Send transact
    # mytoken.initIndexToken(model_name)
    # mytoken.setIndexToken(model_name, str(hash_value))

    # verify hash model
    logger.info("Verify model: {} --- {}".format(model_name, model_hash[model_name]==token_data[1]) )

def tx_evaluate():
    # calculate hash value for model
    model_name = "mnist_cnn.pt"
    model=load_model(model_name)
    hash_value=hash_model(model)

    # evaluate tx committed time
    token_data=mytoken.getIndexToken(model_name)
    original_id = token_data[0] 

    start_time=time.time()
    mytoken.setIndexToken(model_name, str(hash_value))
    while(True):
        token_data=mytoken.getIndexToken(model_name)
        new_id = token_data[0]
        if(new_id > original_id ):
            IndexToken.print_tokendata(token_data)
            break
        time.sleep(0.1)
    exec_time=time.time()-start_time
    logger.info("tx committed time: {:.3f}\n".format(exec_time, '.3f')) 
    FileUtil.save_testlog('test_results', 'exec_tx_commit_ethereum.log', format(exec_time, '.3f'))

if __name__ == "__main__":
    # Logging setup
    FORMAT = "%(asctime)s | %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(level=logging.DEBUG)

    test_main()
    test_hashmodel()

    # test_run = 1

    # for i in range(test_run):
    #     tx_evaluate()

    pass
