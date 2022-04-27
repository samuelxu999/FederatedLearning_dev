'''
========================
Crypto_SYM module
========================
Created on April.7, 2022
@author: Xu Ronghua
@Email:  rxu22@binghamton.edu
@TaskDescription: This module provide symmetric cryptography function based on Fernet API.
@Reference:https://cryptography.io/en/latest/
'''

import base64
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

## ============================= private functions ==============================
## string to bytes
def string_to_bytes(str_data):
	bytes_data=str_data.encode(encoding='UTF-8')
	return bytes_data
	
## bytes to string
def bytes_to_string(byte_data):
	str_data=byte_data.decode(encoding='UTF-8')
	return str_data

## string to random
def string_to_random(str_data):
	random_data=base64.urlsafe_b64decode(str_data.encode('utf-8'))
	return random_data
	
## random to string
def random_to_string(random_data):
	str_data=base64.urlsafe_b64encode(random_data).decode('utf-8')
	return str_data

def save_salt(salt_str, salt_file):
	'''
	Save salt (string) data in salt_file
		@in: salt_bytes 
		@in: salt_file
	'''	
	fname=open(salt_file, 'w') 
	fname.write("%s" %(salt_str))
	fname.close()

def load_salt(salt_file):
	'''
	Load salt (string) data from salt_file 
		@in: salt_file
		@out: salt_str
	'''	
	fname=open(salt_file, 'r') 
	salt_str=fname.read()
	fname.close()
	return salt_str

## ============================= public class ==============================
class Crypto_SYM(object): 
	@staticmethod
	def generate_salt(salt_file):
		'''
		Generate a random number as salt, which is used to derive the same key from the password
			@in: local file to save salt
		'''
		## generate salt by using pseudo random number generator
		salt = os.urandom(16)

		## conver salt from random to string
		str_salt = random_to_string(salt)

		## the salt has to be stored in a retrievable location in salt_file
		save_salt(str_salt, salt_file)

	@staticmethod
	def reload_salt(salt_file):
		'''
		Reload a salt from local file
			@in: local file to save salt
			@out: salt random number
		'''

		## retrive the salt string from location in salt_file
		str_salt = load_salt(salt_file)

		## convert salt from string to random
		random_salt = string_to_random(str_salt)

		return random_salt		

	@staticmethod
	def encrypt(salt, str_passwd, message):
		'''
		Encrypt message given str_passwd and loaded salt
			@in: salt string
			@in: password string
			@in: message string
			@out: encryped token data
		'''

		## convert to binary password
		b_passwd = string_to_bytes(str_passwd)

		## construct PBKDF2HMAC
		kdf = PBKDF2HMAC(	
						algorithm=hashes.SHA256(),
						length=32,
						salt=salt,
						iterations=390000,
						)
		## generate key object
		_key = base64.urlsafe_b64encode(kdf.derive(b_passwd))

		## use key to construct Fernet
		fernet = Fernet(_key)

		## convert to byte message
		b_msg = string_to_bytes(message)

		## encrypt data
		token = fernet.encrypt(b_msg)

		return token

	@staticmethod
	def decrypt(salt, str_passwd, token):
		'''
		Dncrypt message from token given str_passwd and loaded salt
			@in: salt string
			@in: password string
			@in: token string
			@out: decryped message
		'''

		## convert to binary password
		b_passwd = string_to_bytes(str_passwd)

		## construct PBKDF2HMAC
		kdf = PBKDF2HMAC(	
						algorithm=hashes.SHA256(),
						length=32,
						salt=salt,
						iterations=390000,
						)
		## generate key object
		_key = base64.urlsafe_b64encode(kdf.derive(b_passwd))

		## use key to construct Fernet
		fernet = Fernet(_key)

		## decrypt cypher data
		b_msg = fernet.decrypt(token)

		## convert to string message
		decryped_msg = bytes_to_string(b_msg)

		return decryped_msg
