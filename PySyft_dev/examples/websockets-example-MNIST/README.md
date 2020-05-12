# Federated learning using websockets - MNIST example


Reference: https://github.com/OpenMined/PySyft/tree/master/examples/tutorials/advanced/websockets_mnist

The scripts in this folder let you execute a federated training via three websocket connections.

The script start_websocket_servers.py will start the (localhost) Websocket server workers for Alice, Bob and Charlie.
```
$ python3 start_websocket_servers.py
```

If you want to use remote workers that are deployed on networking machine, like Rpi. You need start (remote) Websocket server workers by execute script on that machine.
```
$ python3 run_websocket_server.py --host @IP --port 8777 --id @Worker_ID
```

The training is then started by running the script run_websocket_client.py:
```
$ python run_websocket_client.py
```
This script
 * loads the MNIST dataset,
 * distributes it onto the workers
 * starts a federated training.

 The federated training loop contains the following steps
 * the current model is sent to the workers
 * the workers train on a fixed number of batches
 * the received models from workers are then averaged (federated averaging)

 This training loop is then executed for a given number of epochs.
 The performance on the test set of MNIST is shown after each epoch.
