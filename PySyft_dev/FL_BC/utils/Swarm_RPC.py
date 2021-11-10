'''
========================
Swarm_RPC module
========================
Created on Nov.2, 2021
@author: Xu Ronghua
@Email:  rxu22@binghamton.edu
@TaskDescription: This module provide encapsulation of basic API that access to RPC server side of Swarm node.
'''
import time
import requests
import json
import random

from utils.utilities import FileUtil, TypesUtil

SWARM_SERVER = "./config_data/swarm_server.json"

class Swarm_RPC(object):
	'''
	Swarm RPC class to provide client-based RESTfull APIs
	'''
	@staticmethod
	def download_data(target_address, swarm_hash):
		'''
		fetch data from swarm node
		'''
		headers = {'Content-Type' : 'application/json'}
		api_url = 'http://'+target_address+'/swarm/data/download'
		data_args = {}
		data_args['hash']=swarm_hash

		response = requests.get(api_url, data=json.dumps(data_args), headers=headers)

		json_results = {}
		json_results['status']=response.status_code
		if(json_results['status']==200):
			json_results['data']=response.json()['data']
		else:
			json_results['data']=''

		return json_results

	@staticmethod
	def upload_data(target_address, tx_json):
		'''
		save data on swarm node
		'''
		headers = {'Content-Type' : 'application/json'}
		api_url = 'http://'+target_address+'/swarm/data/upload'
		data_args = {}
		data_args['data']=tx_json

		response = requests.post(api_url, data=json.dumps(data_args), headers=headers)

		json_results = {}
		json_results['status']=response.status_code
		if(json_results['status']==200):
			json_results['data']=response.json()['data']
		else:
			json_results['data']=''

		return json_results

	@staticmethod
	def download_file(target_address, swarm_hash, file_name, download_file='download_data'):
		headers = {'Content-Type' : 'application/json'}
		api_url = 'http://'+target_address+'/swarm/file/download'
		data_args = {}
		data_args['hash']=swarm_hash
		data_args['file_name']=file_name
		data_args['download_file']=download_file

		response = requests.get(api_url, data=json.dumps(data_args), headers=headers)

		json_results ={}
		json_results['status'] = response.status_code
		if(json_results['status']==200):
			file_handle = open(download_file, 'wb')
			file_handle.write(response.content)
			file_handle.close()

			json_results['data']="Download: {} from address: {} to local as: {}".format(file_name, swarm_hash, download_file)
		else:
			json_results['data']=response.json()
		# logger.info(json_results)

		return json_results

	@staticmethod
	def upload_file(target_address, tx_json):
		api_url = 'http://'+target_address+'/swarm/file/upload'

		files = {'uploaded_file': open(tx_json['upload_file'],'rb')}

		response = requests.post(api_url, files=files)

		json_results = {}
		json_results['status']=response.status_code
		if(json_results['status']==200):
			json_results['data']=response.json()['data']
		else:
			json_results['data']=''
		# logger.info(json_results)

		return json_results

	@staticmethod
	def get_service_address():
		'''
		random choose a swarm server from node list 
		'''
		services_host = FileUtil.JSON_load(SWARM_SERVER)
		server_id = random.randint(0,len(services_host['all_nodes'])-1)

		## get address of swarm server
		target_address = services_host['all_nodes'][server_id]

		return target_address


