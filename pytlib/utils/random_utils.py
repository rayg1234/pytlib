import random
import string

def random_str(len=10):
	return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(len))