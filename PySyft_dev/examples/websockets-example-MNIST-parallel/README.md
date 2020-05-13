# Federated learning using websockets - MNIST Asynchronous & parallel example


Reference: https://github.com/OpenMined/PySyft/tree/master/examples/tutorials/advanced/websockets_mnist_parallel

The scripts in this folder let you execute a a synchronous federated training via four websocket connections includes 3 training workers and a testing worker.

The script start_websocket_servers.py will start the (localhost) Websocket server workers for Alice, Bob and Charlie.
```
$ python3 start_websocket_servers.py --localworkers
```

If you want to use remote workers that are deployed on networking machine, like Rpi. You need start (remote) Websocket server workers by execute script on that machine.

Training worker: 

```
$ python3 start_websocket_servers.py --host @IP --port 8777 --id @Worker_ID
```

Testing worker: 

```
$ python3 start_websocket_servers.py --host @IP --port 8777 --id @Worker_ID --testing
```

The federated training could be started by running the script run_websocket_client.py:
```
$ python run_websocket_client.py
```
Using --localworkers can select mode: local work or remote worker.

This script
 * configure task and model,
 * distributes it onto the workers
 * starts a federated training
 * train model on training workers
 * evaluate model on testing worker.

 The federated training loop contains the following steps
 * the current model is sent to the workers
 * the workers train on a fixed number of batches
 * the received models from workers are then averaged (federated averaging)

 This training loop is then executed for a given number of epochs.
 The performance on the test set of MNIST is shown after each epoch.
