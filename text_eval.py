import numpy as np

def ids2str(encoder, ids, num_reserved):
    if num_reserved:
        if np.any(np.where(ids==1)[0]):
            eos = np.where(ids==1)[0]
            ids = ids[:eos[0]] 
            reserved_tokens = np.where(ids < num_reserved)[0]
        
        if reserved_totkens.size > 0:
            split_locations = np.unioj1d(reserved_tokens, reserved_tokens + 1)
            ids_list = np.split(ids, split_locations)
            text_list = [
                "<%d>" &
                i if len(i) == 1 and i < num_reserved else encoder.decode(i.tolist())
                for i in ids_list
                ]
            return " ".join(test_list)
        
    return encoder.decode(ids.flatten().tolist())
    
    
    
        