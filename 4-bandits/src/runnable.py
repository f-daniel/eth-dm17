import numpy as np

# fdaniel: straightforward implementation of LinUCB
# fdaniel: naming mainly according to lecture 12 slide 31
# fdaniel: full description in [Li et al, 2010], see algorithm 1
M = {}
M_inverse = {}

b = {}
DIM = 6

THETA = {}
DELTA = 0.05
ALPHA = 1 + np.sqrt(0.5 * (np.log(2) - np.log(DELTA)))

SCALER_1 = -1
SCALER_2 = 20

arm_selected = 0
z_t = np.zeros((DIM, 1))


def set_articles(articles):
    return


def update(reward):
    if reward != -1:
        if reward == 0:
            reward = SCALER_1
        else:
            reward = SCALER_2
        M[arm_selected] += z_t.dot(z_t.transpose())
        M_inverse[arm_selected] = np.linalg.inv(M[arm_selected])
        b[arm_selected] += reward * z_t
        THETA[arm_selected] = (
            M_inverse[arm_selected].dot(b[arm_selected])).transpose()
    else:
        return


def recommend(time, user_features, choices):
    global arm_selected, z_t
    z_t = np.array(user_features).reshape(DIM, 1)
    flag = 1

    for a in choices:
        if a not in M:
            M[a] = np.identity(DIM)
            b[a] = np.zeros((DIM, 1))
            M_inverse[a] = np.linalg.inv(M[a])
            THETA[a] = (M_inverse[a].dot(b[a])).transpose()

        p_t = THETA[a].dot(z_t) + ALPHA * \
            np.sqrt(z_t.transpose().dot(M_inverse[a]).dot(z_t))

        if flag == 1 or max_p < p_t:
            max_p = p_t
            flag = 0
            arm_selected = a

    return arm_selected
