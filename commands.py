import os


for i in range(1, 5):
    print(i)
    os.system('python main.py --image_name input{}'.format(i))