import numpy as np
import numpy.linalg as la

def kalman(mu,P,F,Q,B,u,z,H,R):
    # mu    : state matrix (position and velocity)
    # P     : covariance matrix (uncertainty)
    # F,Q   : dynamic system (laws of physics) and its noise
    # B     : model control
    # u     : input (which variables are going to be dynamic (with accelerations))
    # z     : observation [x,y,z]
    # H,R   : observation model and its noise
    
    # New predicted state matrix and covariance matrix
    # @ means matrix multiplication
    mup = F @ mu + B @ u;
    pp  = F @ P @ F.T + Q;

    # Predicted observation [x, y, z]
    zp = H @ mup

    # if there is no observation we only make predictions 
    if z is None:
        return mup, pp, zp
    # how much the prediction differs from observation
    epsilon = z - zp

    #Calculate Kalman Gain
    k = pp @ H.T @ la.inv(H @ pp @ H.T +R)
    #print(k[0][0])
    # based on kalman gain we calculate more precise state and covariance
    new_mu = mup + k @ epsilon;
    new_P  = (np.eye(len(P))-k @ H) @ pp;
    return new_mu, new_P, zp
