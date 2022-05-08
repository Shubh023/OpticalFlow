import numpy as np

def angular_err(wr, we):
    angular_error = np.mean(np.arccos((1 + np.dot(we.flatten(), wr.flatten())) / 
                                (np.sqrt(1 + np.linalg.norm(wr)**2) * np.sqrt(1 + np.linalg.norm(we)**2))))
    return angular_error
    
def error(wr, we):
    angular_error = np.mean(np.arccos((1 + np.dot(we.flatten(), wr.flatten())) / 
                                (np.sqrt(1 + np.linalg.norm(wr)**2) * np.sqrt(1 + np.linalg.norm(we)**2))))
    
    mean_squared_error = np.mean((wr - we)**2)
    
    euclidean_error = np.mean(np.sqrt((we[:,:,0] - wr[:,:,0])**2 + (we[:,:,1] - wr[:,:,1])**2))
       
    return angular_error, euclidean_error, mean_squared_error
