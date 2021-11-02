#!/usr/bin/env python3

'''
========================
Micro_RPC module
========================
Created on May.28, 2020
@author: Xu Ronghua
@Email:  rxu22@binghamton.edu
@TaskDescription: This module provide encapsulation of basic API that access to Microchain RPC.
'''

import requests
import json
import time
import os
from utils.utilities import TypesUtil,FileUtil

class Micro_RPC(object):
    '''
    Get microchain information
    '''
    @staticmethod
    def micro_info(target_address):
        headers = {'Content-Type' : 'application/json'}
        api_url='http://'+target_address+'/test/validator/getinfo'        
        response = requests.get(api_url, headers=headers)

        #get response json
        json_response = response.json()      

        return json_response

    '''
    Execute query to fetch tx
    '''
    # @staticmethod
    def tx_query(target_address, tx_json={}):            
        headers = {'Content-Type' : 'application/json'}
        api_url='http://'+target_address+'/test/transaction/query' 
        response = requests.get(api_url, data=json.dumps(tx_json), headers=headers)
        
        #get response json
        json_response = response.json()      

        return json_response

    '''
    Send transaction to network for commit
    '''
    @staticmethod
    def broadcast_tx_commit(target_address, tx_json={}):          
        headers = {'Content-Type' : 'application/json'}
        api_url='http://'+target_address+'/test/transaction/commit'
        response = requests.post(api_url, data=json.dumps(tx_json), headers=headers)
        
        #get response json
        json_response = response.json()      

        return json_response

