# Import necessary modules
from train import train
from test import test
from save_video import save_video

if __name__ == "__main__":
    # Call the necessary functions based on requirements
    # For example:
    
    # train the model 
    train()
    
    #test the model 
    max = 0
    for i in range(100):
        temp = test()
        if temp > max:
            max = temp
    print("Return in the test is :", max)

    # OPTIONAL!  save the game video  
    # save_video()
    
    pass
