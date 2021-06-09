import torch
import numpy as np

def trans_filter_d(w, iss):
    # Transformation of the depthwise (spatial) filter
    if iss == 4:
        # P4 (rotation)
        w_transformed = torch.cat(
            (w,
             torch.rot90(w,1,(3,4)),
             torch.rot90(w,2,(3,4)),
             torch.rot90(w,3,(3,4))), dim=1)
    elif iss == 8:
        # P4M (rotation and mirroring)
        idx = tuple(np.arange(w.shape[-2])[::-1])
        wf = w[:,:,:,idx,:]
        w_transformed = torch.cat(
            (w,
             torch.rot90(w,1,(3,4)),
             torch.rot90(w,2,(3,4)),
             torch.rot90(w,3,(3,4)),
             wf,
             torch.rot90(wf,-1,(3,4)),
             torch.rot90(wf,-2,(3,4)),
             torch.rot90(wf,-3,(3,4))
             ), dim=1)
    return w_transformed

def trans_filter_p(w):
    # Transformation of the pointwise (1x1) filter (cyclic permutation)
    w = w[:,None,:,:,:,:]
    if w.shape[3] == 4:
        # P4
        w_transformed = torch.cat((
            w,
            w[:,:,:,(3,0,1,2),:,:],
            w[:,:,:,(2,3,0,1),:,:],
            w[:,:,:,(1,2,3,0),:,:]), dim=1)
    elif w.shape[3] == 8:
        # P4M
        w_transformed = torch.cat(
            (w,
             w[:,:,:,(3,0,1,2,5,6,7,4),:,:],
             w[:,:,:,(2,3,0,1,6,7,4,5),:,:],
             w[:,:,:,(1,2,3,0,7,4,5,6),:,:],
             w[:,:,:,(4,5,6,7,0,1,2,3),:,:],
             w[:,:,:,(5,6,7,4,3,0,1,2),:,:],
             w[:,:,:,(6,7,4,5,2,3,0,1),:,:],
             w[:,:,:,(7,4,5,6,1,2,3,0),:,:]), dim=1)
    return w_transformed
