import time
import logging
import argparse
import sys
import os
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms

from utils.model_utils import ModelUtils, EtherUtils, TenderUtils, MicroUtils, DatasetUtils
from utils.utilities import FileUtil
from utils.Swarm_RPC import Swarm_RPC

LOG_INTERVAL = 25

logger = logging.getLogger(__name__)

#global variable

def define_and_get_arguments(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Run federated learning using websocket client workers."
    )
    parser.add_argument(
        "--id", type=str, help="name (id) of the websocket server worker, e.g. --id alice"
    )
    parser.add_argument(
        "--model_name", type=str, default='mnist_cnn.pt', help="model name that store state_dict()"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch size of the training")
    parser.add_argument(
        "--test_batch_size", type=int, default=128, help="batch size used for the test data"
    )
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--eval_model", action="store_true", help="Evaluate model")
    parser.add_argument("--seed", type=int, default=1, help="seed used for randomization")
    parser.add_argument("--tx_round", type=int, default=1, help="tx evaluation round")
    parser.add_argument("--wait_interval", type=int, default=1, 
                        help="break time between tx evaluate step.")
    parser.add_argument("--test_network", type=int, default=0, 
                        help="Blockchain test network: 0-None, 1-Etherem, 2-Tendermint, 3-Microchain")
    parser.add_argument("--test_func", type=int, default=0, 
                        help="Execute test function: 0-test_model, 1-test_hashmodel, \
                        	2-test_hashdataset, 3-test_maskedModel, 4-test_swarm")
    parser.add_argument("--op_status", type=int, default=0, 
                        help="operation status: based on app")
    parser.add_argument("--query_tx", type=int, default=0, 
                        help="Query tx or commit tx: 0-Query, 1-Commit")
    parser.add_argument("--mask_model", type=int, default=0, 
                        help="Mask model operation: 0-mask input, 1-Sum of model and Fedavg")
    args = parser.parse_args(args=args)
    return args

def test_model(args):
    use_cuda = args.cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    logger.info("module setup...\n")
    ## load model from file
    model_file = "./data/"+args.model_name
    model=ModelUtils.load_model(model_file, True)

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
    ModelUtils.evaluate_model(model, device, test_loader)


def test_hashmodel(args):
	## set model file path
	model_name = "./data/"+args.model_name
	for i in range(args.tx_round):
	    logger.info("Test run:{}".format(i+1))
	    if(args.test_network==1):
	        # logger.info("Round : {}".format(i+1) )
	        # -------------------------- Ethereum test ----------------------------------
	        if(args.query_tx==0):
	            # verify hash model
	            logger.info("Verify model: '{}' --- {}\n".format(model_name, 
	                                                        EtherUtils.verify_hashmodel(model_name)) )
	        else:                
	            # call tx_evaluate() and record tx_commit time
	            logger.info("Tx commit model '{}' --- {}\n".format(model_name, 
	                                                        EtherUtils.tx_evaluate(model_name)))
	    elif(args.test_network==2):
	        # -------------------------- Tendermint test ----------------------------------
	        if(args.query_tx==0):
	            # verify hash model
	            logger.info("Verify model: '{}' --- {}\n".format(model_name, 
	                                                        TenderUtils.verify_hashmodel(model_name)) )
	        else:
	            # call tx_evaluate() and record tx_commit time
	            logger.info("Tx commit model '{}' --- {}\n".format(model_name, 
	                                                        TenderUtils.tx_evaluate(model_name)))
	    elif(args.test_network==3):
	        # -------------------------- Microchain test ----------------------------------
	        # MicroUtils.get_info()
	        if(args.query_tx==0):
	            # verify hash model
	            logger.info("Verify model: '{}' --- {}\n".format(model_name, 
	                                                        MicroUtils.verify_hashmodel(model_name)) )
	        else:
	            # call tx_evaluate() and record tx_commit time
	            logger.info("Tx commit model '{}' --- {}\n".format(model_name, 
	                                                        MicroUtils.tx_evaluate(model_name)))
	    else:
	        pass
	    time.sleep(args.wait_interval)

def test_hashdataset(args, training=False):

    KEEP_LABELS_DICT = {
        "alice": [0, 1, 2, 3],
        "bob": [4, 5, 6],
        "charlie": [7, 8, 9],
        "testing": list(range(10)),
        None: list(range(10)),
    }

    ls_time_exec = []
    # 1) Load MINST datset
    start_time=time.time()
    mnist_dataset = DatasetUtils.load_dataset("./data", training)
    ls_time_exec.append( format( time.time()-start_time, '.3f' ) ) 

    # show dataset information
    logger.info("dataset shape: %s", mnist_dataset.data.shape)
    keep_labels=KEEP_LABELS_DICT[args.id]
    indices = np.isin(mnist_dataset.targets, keep_labels).astype("uint8")

    count = [0] * 10
    logger.info("number of true indices: %s", indices.sum())
    for i in range(10):
        if(i in keep_labels):
            count[i] = (mnist_dataset.targets == i).sum().item()
        logger.info("      %s: %s", i, count[i])

    # 2) hash dataset
    start_time=time.time()
    logger.info(DatasetUtils.hash_dataset(mnist_dataset, [0]))
    ls_time_exec.append( format( time.time()-start_time, '.3f' ) ) 

    logger.info("load dataset: {} s    hash dataset: {} s".format(ls_time_exec[0], ls_time_exec[1]))

def test_maskedModel(args):
	use_cuda = args.cuda and torch.cuda.is_available()

	torch.manual_seed(args.seed)

	device = torch.device("cuda" if use_cuda else "cpu")
	kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

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
	## set model file path
	model_name = "./data/"+args.model_name
	model=ModelUtils.load_model(model_name, False)

	for i in range(args.tx_round):
		if(args.mask_model==0):
			# test mask input
			logger.info("Test run:{}".format(i+1))
			start_time=time.time()
			masked_model = ModelUtils.mask_model(model, 0.1)
			str_time_exec= format( (time.time()-start_time)*1000, '.3f' ) 
			FileUtil.save_testlog('test_results', 'mask_model.log', str_time_exec)

			if(args.eval_model):
				logger.info("test masked module...\n")
				ModelUtils.evaluate_model(masked_model, device, test_loader)

				unmask_model = ModelUtils.mask_model(masked_model, -0.1)

				logger.info("test unmasked module...\n")
				ModelUtils.evaluate_model(unmask_model, device, test_loader)
		else:
			# test sum masked value
			logger.info("Test run:{}".format(i+1))
			model_list = []
			for i in range(1, 21):
				model_list.append(model)

			start_time=time.time()
			fedavg_model = ModelUtils.FedAvg_model(model_list)
			str_time_exec= format( (time.time()-start_time)*1000, '.3f' )
			FileUtil.save_testlog('test_results', 'sum_model.log', str_time_exec)

			if(args.eval_model):
				logger.info("test fedavg_model module...\n")
				ModelUtils.evaluate_model(fedavg_model, device, test_loader)

def test_swarm(args):
	for i in range(args.tx_round):
		logger.info("Test run:{}".format(i+1))
		## get swarm server node address
		target_address = Swarm_RPC.get_service_address()

		##==================== test upload file =========================
		tx_json = {}
		if(args.op_status==0):
			file_name = "swarm_file.txt"
			file_download = "./data/download_file.txt"
		else:
			file_name = "mnist_cnn.pt"
			file_download = "./data/download_model.pt"

		tx_json['upload_file']="./data/"+file_name

		ls_time_exec = []
		start_time=time.time()
		## send file to swarm server
		post_ret = Swarm_RPC.upload_file(target_address, tx_json)
		ls_time_exec.append(format( (time.time()-start_time)*1000, '.3f' )) 
		logger.info(post_ret)

		##==================== test download file =========================
		## get swarm hash from post_ret by upload_file
		swarm_hash = post_ret['data']

		start_time=time.time()
		## call smarm server to retrive file
		query_ret = Swarm_RPC.download_file(target_address, swarm_hash, file_name, file_download)
		ls_time_exec.append(format( (time.time()-start_time)*1000, '.3f' )) 
		logger.info(query_ret)

		str_time_exec=" ".join(ls_time_exec)
		FileUtil.save_testlog('test_results', 'test_swarm.log', str_time_exec)



if __name__ == "__main__":
	# Logging setup
	FORMAT = "%(asctime)s | %(message)s"
	logging.basicConfig(format=FORMAT)
	logger.setLevel(level=logging.DEBUG)

	modelUtils_logger = logging.getLogger("utils.model_utils")
	modelUtils_logger.setLevel(logging.INFO)
	indextoken_logger = logging.getLogger("utils.Index_Token")
	indextoken_logger.setLevel(logging.INFO)

	args = define_and_get_arguments()

	if(args.test_func==0):
		test_model(args)
	elif(args.test_func==1):
		test_hashmodel(args)
	elif(args.test_func==2):
		test_hashdataset(args)
	elif(args.test_func==3):
		test_maskedModel(args)
	elif(args.test_func==4):
		test_swarm(args)
	else:
		pass
