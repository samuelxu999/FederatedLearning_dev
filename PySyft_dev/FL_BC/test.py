import time
import logging
import argparse
import sys
import os

import torch
from torchvision import datasets
from torchvision import transforms

from model_utils import ModelUtils, EtherUtils, TenderUtils

LOG_INTERVAL = 25

logger = logging.getLogger(__name__)

#global variable

def define_and_get_arguments(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Run federated learning using websocket client workers."
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch size of the training")
    parser.add_argument(
        "--test_batch_size", type=int, default=128, help="batch size used for the test data"
    )
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--seed", type=int, default=1, help="seed used for randomization")
    parser.add_argument("--tx_round", type=int, default=1, help="tx evaluation round")
    parser.add_argument("--test_network", type=int, default=0, 
                        help="Blockchain test network: 0-None, 1-Etherem, 2-Tendermint")
    args = parser.parse_args(args=args)
    return args

def test_model():
    args = define_and_get_arguments()

    use_cuda = args.cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    logger.info("module setup...\n")
    # model=load_model("mnist_cnn.pt", True)
    model=ModelUtils.load_model("mnist_cnn.pt", True)

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


def test_hashmodel(model_name):
    args = define_and_get_arguments()

    for i in range(args.tx_round):
        if(args.test_network==1):
            # logger.info("Round : {}".format(i+1) )
            # -------------------------- Ethereum test ----------------------------------
            # verify hash model
            logger.info("Verify model: '{}' --- {}\n".format(model_name, 
                                                            EtherUtils.verify_hashmodel(model_name)) )
            
            # # call tx_evaluate() and record tx_commit time
            # logger.info("Tx commit model '{}' --- {}\n".format(model_name, 
            #                                                 EtherUtils.tx_evaluate(model_name)))
        elif(args.test_network==2):
            # -------------------------- Tender test ----------------------------------
            # verify hash model
            logger.info("Verify model: '{}' --- {}\n".format(model_name, 
                                                            TenderUtils.verify_hashmodel(model_name)) )

            # # call tx_evaluate() and record tx_commit time
            # logger.info("Tx commit model '{}' --- {}\n".format(model_name, 
            #                                                 TenderUtils.tx_evaluate(model_name)))
        else:
            pass

if __name__ == "__main__":
    # Logging setup
    FORMAT = "%(asctime)s | %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(level=logging.DEBUG)

    modelUtils_logger = logging.getLogger("model_utils")
    modelUtils_logger.setLevel(logging.INFO)
    indextoken_logger = logging.getLogger("Index_Token")
    indextoken_logger.setLevel(logging.INFO)

    test_model()
    test_hashmodel("mnist_cnn.pt")

    pass
