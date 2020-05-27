#!/usr/bin/env python3.5

'''
========================
Index token module
========================
Created on July.02, 2018
@author: Xu Ronghua
@Email:  rxu22@binghamton.edu
@TaskDescription: This module provide encapsulation of web3.py API to interact with IndexToken.sol smart contract.
'''

from web3 import Web3, HTTPProvider, IPCProvider
from utilities import DatetimeUtil, TypesUtil
import json, datetime, time
import logging

logger = logging.getLogger(__name__)

class IndexToken(object):
	def __init__(self, http_provider, contract_addr, contract_config):
		# configuration initialization
		self.web3 = Web3(HTTPProvider(http_provider))
		self.contract_address=Web3.toChecksumAddress(contract_addr)
		self.contract_config = json.load(open(contract_config))

		# new contract object
		self.contract=self.web3.eth.contract()
		self.contract.address=self.contract_address
		self.contract.abi=self.contract_config['abi']

	def Show_ContractInfo(self):  
		logger.info("Show contract information:")
		logger.info("blockNumber: {}".format(self.web3.eth.blockNumber) )
		logger.info("Contract address: {}".format(self.contract.address) )
		accounts = self.web3.eth.accounts
		for account in accounts:
			logger.info("Host account: {}    Balance: {}".format(account,
					self.web3.fromWei(self.web3.eth.getBalance(self.web3.eth.accounts[0]), 'ether'))
			)


	# return accounts address
	def getAccounts(self):
		return self.web3.eth.accounts

	# return accounts balance
	def getBalance(self, account_addr=''):
		if(account_addr==''):
			# get accounts[0] balance
			checksumAddr=self.web3.eth.coinbase
		else:
			#Change account address to EIP checksum format
			checksumAddr=Web3.toChecksumAddress(account_addr)	
		return self.web3.fromWei(self.web3.eth.getBalance(checksumAddr), 'ether')

	#get token data by call getIndexToken()
	def getIndexToken(self, str_index):

		token_data = []
		'''
		Call a contract function, executing the transaction locally using the eth_call API. 
		This will not create a new public transaction.
		'''

		# get token status
		token_data=self.contract.call({'from': self.web3.eth.coinbase}).getIndexToken(str_index)

		return token_data

	#get token data by call getIndexToken()
	def getAuthorizedNodes(self):

		node_data = []
		'''
		Call a contract function, executing the transaction locally using the eth_call API. 
		This will not create a new public transaction.
		'''

		# get token status
		node_data=self.contract.call({'from': self.web3.eth.coinbase}).getAuthorizedNodes()

		return node_data

	#initialized token by sending transact to call initIndexToken()
	def initIndexToken(self, str_index):

		# Execute the specified function by sending a new public transaction.
		ret=self.contract.transact({'from': self.web3.eth.coinbase}).initIndexToken(self.web3.eth.coinbase, str_index)

	# set isValid flag in token
	def setIndexToken(self, str_index, hash_index):

		# Execute the specified function by sending a new public transaction.	
		ret=self.contract.transact({'from': self.web3.eth.coinbase}).setIndexToken(self.web3.eth.coinbase, str_index, hash_index)

	# set authorizaed nodes
	def addAuthorizedNodes(self, node_addr):
		checksumAddr=Web3.toChecksumAddress(node_addr)
		# Execute the specified function by sending a new public transaction.	
		ret=self.contract.transact({'from': self.web3.eth.coinbase}).addAuthorizedNodes(checksumAddr)

	# remove authorizaed nodes
	def removeAuthorizedNodes(self, node_addr):
		checksumAddr=Web3.toChecksumAddress(node_addr)
		# Execute the specified function by sending a new public transaction.	
		ret=self.contract.transact({'from': self.web3.eth.coinbase}).removeAuthorizedNodes(checksumAddr)

	# Print token date
	@staticmethod
	def print_tokendata(token_data):
		#print token status
		logger.info("Fetched model hash value:")
		for i in range(0,len(token_data)):
			logger.info(token_data[i])


	# get address from json file
	@staticmethod
	def getAddress(node_name, datafile):
		address_json = json.load(open(datafile))
		return address_json[node_name]

def test_main():
	addr_list = "./addr_list.json"
	http_provider = "http://localhost:8042"
	contract_addr = IndexToken.getAddress('HashModelToken', addr_list)
	contract_config = "./contracts/IndexToken.json"

	#new ABACToken object
	mytoken=IndexToken(http_provider, contract_addr, contract_config)
	mytoken.Show_ContractInfo()


	#------------------------- test contract API ---------------------------------
	#Read token data using call
	token_data=mytoken.getIndexToken('1')
	IndexToken.print_tokendata(token_data)

	#read node data using call
	node_list=mytoken.getAuthorizedNodes()
	logger.info("Authorized node:")
	for node in node_list:
		logger.info("    {}".format(node))

	#Send transact
	#mytoken.initIndexToken('1');
	#mytoken.setIndexToken('1', 'dave')

	node_address = IndexToken.getAddress('rack1_PI_Plus_1', './addr_list.json')
	# mytoken.addAuthorizedNodes(node_address)
	#mytoken.removeAuthorizedNodes(node_address)

def tx_commit():
	addr_list = "./addr_list.json"
	http_provider = "http://localhost:8042"
	contract_addr = IndexToken.getAddress('HashModelToken', addr_list)
	contract_config = "./contracts/IndexToken.json"	

	#new ABACToken object
	mytoken=IndexToken(http_provider, contract_addr, contract_config)

	token_data=mytoken.getIndexToken('1')
	original_id = token_data[0]

	start_time=time.time()
	mytoken.setIndexToken('1', 'dave')

	while(True):
		token_data=mytoken.getIndexToken('1')
		new_id = token_data[0]
		if(new_id > original_id ):
			print(token_data)
			break
		time.sleep(0.1)
	exec_time=time.time()-start_time
	print(format(exec_time, '.3f'))

if __name__ == "__main__":
	# Logging setup
	FORMAT = "%(asctime)s | %(message)s"
	logging.basicConfig(format=FORMAT)
	logger.setLevel(level=logging.DEBUG)

	test_main()
	# tx_commit()