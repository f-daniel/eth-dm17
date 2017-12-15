import numpy as np


A = {}
b = {}
d = 6

#LinUCB parameter initialization
DELTA = 0.05
ALPHA = 1 + np.sqrt(0.5 *(np.log(2) - np.log(DELTA)))
R0 = -1
R1 = 20
print "ALPHA = {}, R0 = {}, R1 = {}".format(ALPHA, R0, R1)

# save interim calculations
A_inverse = {}
THETA = {}
arm_selected = 0
u_features = np.zeros((d, 1))


def set_articles(articles):
    return


def update(reward):
    if reward == -1:
        return
    else:
        # reward scaling
        if reward == 0:
            reward = R0
        else:
            reward = R1
        # update
        A[arm_selected] += u_features.dot(u_features.T)
        b[arm_selected] += reward*u_features
        A_inverse[arm_selected] = np.linalg.inv(A[arm_selected])
        THETA[arm_selected] = (A_inverse[arm_selected].dot(b[arm_selected])).T


def recommend(time, user_features, choices):
    global arm_selected, u_features
    u_features = np.array(user_features).reshape(d, 1)

    first = 1
    for arm in choices:
        # create arrays for new article 
        if arm not in A:
            A[arm] = np.identity(d)
            b[arm] = np.zeros((d, 1))
            A_inverse[arm] = np.linalg.inv(A[arm])
            THETA[arm] = (A_inverse[arm].dot(b[arm])).T

        # UCB calculation
        p = THETA[arm].dot(u_features) + ALPHA*np.sqrt(u_features.T.dot(A_inverse[arm]).dot(u_features))

        # select content with highest UCB
        if first == 1 or p > max_p:
            first = 0
            max_p = p
            arm_selected = arm

    return arm_selected