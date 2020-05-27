import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torchvision import datasets, transforms
import logging
import argparse
import sys
import os
import threading
import time
import queue

import syft as sy
from syft.workers.websocket_client import WebsocketClientWorker
from syft.workers.virtual import VirtualWorker
from syft.frameworks.torch.fl import utils

logger = logging.getLogger(__name__)
LOG_INTERVAL = 25

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.max_pool2d(x, 2, 2)
        x = f.relu(self.conv2(x))
        x = f.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return f.log_softmax(x, dim=1)

class TrainThread (threading.Thread):
    '''
    Threading class to handle training task by multiple threads pool
    [ret_queue, [worker, curr_batches, model, device, lr]]
    '''
    def __init__(self, argv):
        threading.Thread.__init__(self)
        self.argv = argv

    #The run() method is the entry point for a thread.
    def run(self):
        # set parameters
        worker =       self.argv[1][0]
        curr_batches = self.argv[1][1]
        model =        self.argv[1][2]
        device =       self.argv[1][3]
        lr =           self.argv[1][4]
        # call train_on_batches() for worker
        _model, _loss = train_on_batches(
            worker, curr_batches, model, device, lr
        )
        #save results into queue
        self.argv[0].put( [worker, _model, _loss] )
        
def train_on_batches(worker, batches, model_in, device, lr):
    """Train the model on the worker on the provided batches

    Args:
        worker(syft.workers.BaseWorker): worker on which the
        training will be executed
        batches: batches of data of this worker
        model_in: machine learning model, training will be done on a copy
        device (torch.device): where to run the training
        lr: learning rate of the training steps

    Returns:
        model, loss: obtained model and loss after training

    """
    model = model_in.copy()
    optimizer = optim.SGD(model.parameters(), lr=lr)  # TODO momentum is not supported at the moment

    model.train()
    model.send(worker)
    loss_local = False

    for batch_idx, (data, target) in enumerate(batches):
        loss_local = False
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = f.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            loss = loss.get()  # <-- NEW: get the loss back
            loss_local = True
            logger.debug(
                "Train Worker {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    worker.id,
                    batch_idx,
                    len(batches),
                    100.0 * batch_idx / len(batches),
                    loss.item(),
                )
            )

    if not loss_local:
        loss = loss.get()  # <-- NEW: get the loss back
    model.get()  # <-- NEW: get the model back
    return model, loss


def get_next_batches(fdataloader: sy.FederatedDataLoader, nr_batches: int):
    """retrieve next nr_batches of the federated data loader and group
    the batches by worker

    Args:
        fdataloader (sy.FederatedDataLoader): federated data loader
        over which the function will iterate
        nr_batches (int): number of batches (per worker) to retrieve

    Returns:
        Dict[syft.workers.BaseWorker, List[batches]]

    """
    batches = {}
    for worker_id in fdataloader.workers:
        worker = fdataloader.federated_dataset.datasets[worker_id].location
        batches[worker] = []
    try:
        for i in range(nr_batches):
            next_batches = next(fdataloader)
            for worker in next_batches:
                batches[worker].append(next_batches[worker])
    except StopIteration:
        pass
    return batches


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += f.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logger.debug("\n")
    accuracy = 100.0 * correct / len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), accuracy
        )
    )


def define_and_get_arguments(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Run federated learning using websocket client workers."
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size of the training")
    parser.add_argument(
        "--test_batch_size", type=int, default=1000, help="batch size used for the test data"
    )
    parser.add_argument("--epochs", type=int, default=2, help="number of epochs to train")
    parser.add_argument(
        "--federate_after_n_batches",
        type=int,
        default=50,
        help="number of training steps performed on each remote worker " "before averaging",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--seed", type=int, default=1, help="seed used for randomization")
    parser.add_argument("--save_model", action="store_true", help="if set, model will be saved")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="if set, websocket client workers will " "be started in verbose mode",
    )
    parser.add_argument(
        "--use_virtual", action="store_true", help="if set, virtual workers will be used"
    )
    parser.add_argument(
        "--localworkers", action="store_true", 
        help="if set, use (localhost) websocket server workers, otherwize, use (remote) worker servers."
    )
    args = parser.parse_args(args=args)
    return args

def wroker_config(args):
    hook = sy.TorchHook(torch)

    logger.info("Worker setup.\n")
    if args.use_virtual:
        alice = VirtualWorker(id="alice", hook=hook, verbose=args.verbose)
        bob = VirtualWorker(id="bob", hook=hook, verbose=args.verbose)
        charlie = VirtualWorker(id="charlie", hook=hook, verbose=args.verbose)
        workers = [alice, bob, charlie]
    else:
        if args.localworkers:
            kwargs_websocket = {"host": "localhost", "hook": hook, "verbose": args.verbose}
            alice = WebsocketClientWorker(id="alice", port=8777, **kwargs_websocket)
            bob = WebsocketClientWorker(id="bob", port=8778, **kwargs_websocket)
            charlie = WebsocketClientWorker(id="charlie", port=8779, **kwargs_websocket)
            workers = [alice, bob, charlie]
        else:
            kwargs_websocket_Pi4_R1_1 = {"host": "128.226.77.157", "hook": hook}
            Pi4_R1_1 = WebsocketClientWorker(id="Pi4_R1_1", port=8777, **kwargs_websocket_Pi4_R1_1)

            kwargs_websocket_Pi4_R1_2 = {"host": "128.226.78.128", "hook": hook}
            Pi4_R1_2 = WebsocketClientWorker(id="Pi4_R1_2", port=8777, **kwargs_websocket_Pi4_R1_2)

            kwargs_websocket_Pi4_R1_3 = {"host": "128.226.88.155", "hook": hook}
            Pi4_R1_3 = WebsocketClientWorker(id="Pi4_R1_3", port=8777, **kwargs_websocket_Pi4_R1_3)

            kwargs_websocket_Pi4_R1_4 = {"host": "128.226.79.31", "hook": hook}
            Pi4_R1_4 = WebsocketClientWorker(id="Pi4_R1_4", port=8777, **kwargs_websocket_Pi4_R1_4)

            workers = [Pi4_R1_1, Pi4_R1_2, Pi4_R1_3, Pi4_R1_4]

    return workers

def async_train(
    model, device, federated_train_loader, lr, federate_after_n_batches, abort_after_one=False
):
    model.train()

    nr_batches = federate_after_n_batches

    models = {}
    loss_values = {}
    # Create queue to save results
    ret_queue = queue.Queue()

    iter(federated_train_loader)  # initialize iterators
    batches = get_next_batches(federated_train_loader, nr_batches)
    counter = 0

    while True:
        logger.debug(f"Starting training round, batches [{counter}, {counter + nr_batches}]")
        data_for_all_workers = True

        # Create thread pool
        threads_pool = []
        # clear queue
        ret_queue.queue.clear()

        # for each work to start train thread
        for worker in batches:
            curr_batches = batches[worker]
            if curr_batches:
                # Create new threads
                p_thread = TrainThread( [ret_queue, 
                    [worker, curr_batches, model, device, lr] ] )

                # append to threads pool
                threads_pool.append(p_thread)

                # The start() method starts a thread by calling the run method.
                p_thread.start() 
            else:
                data_for_all_workers = False

        # The join() waits for all threads to terminate.
        for p_thread in threads_pool:
            p_thread.join()

        # get all results from queue
        while not ret_queue.empty():
            q_data = ret_queue.get()
            models[q_data[0]] = q_data[1]
            loss_values[q_data[0]] = q_data[2]
        
        counter += nr_batches
        if not data_for_all_workers:
            logger.debug("At least one worker ran out of data, stopping.")
            break

        logger.info("Execute federated avg.")
        model = utils.federated_avg(models)

        logger.info("Get next batches.")
        batches = get_next_batches(federated_train_loader, nr_batches)
        if abort_after_one:
            break
    return model

def main():
    logger.info("Configure arguments.\n")
    args = define_and_get_arguments()

    workers=wroker_config(args)

    use_cuda = args.cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    logger.info("federated_train_loader setup.\n")
    federated_train_loader = sy.FederatedDataLoader(
        datasets.MNIST(
            "./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ).federate(tuple(workers)),
        batch_size=args.batch_size,
        shuffle=True,
        iter_per_worker=True,
        **kwargs,
    )

    logger.info("test_loader setup.\n")
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs,
    )

    logger.info("Net() setup and start traning.\n")
    model = Net().to(device)
    if( os.path.isfile('mnist_cnn.pt') ):
        model.load_state_dict(torch.load("mnist_cnn.pt"))
        model.eval()

    for epoch in range(1, args.epochs + 1):
        logger.info("Starting epoch %s/%s", epoch, args.epochs)
        model = async_train(model, device, federated_train_loader, args.lr, args.federate_after_n_batches)        
        test(model, device, test_loader)
        
        # save indermediate model
        model_dir = "models"
        if(not os.path.exists(model_dir)):
            os.makedirs(model_dir)
        model_name = "{}/mnist_cnn_{}.pt".format(model_dir, epoch)
        torch.save(model.state_dict(), model_name)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == "__main__":
    FORMAT = "%(asctime)s %(levelname)s %(filename)s(l:%(lineno)d) - %(message)s"
    LOG_LEVEL = logging.DEBUG
    logging.basicConfig(format=FORMAT, level=LOG_LEVEL)

    # Websockets setup
    websockets_logger = logging.getLogger("websockets")
    websockets_logger.setLevel(logging.INFO)
    websockets_logger.addHandler(logging.StreamHandler())

    main()

