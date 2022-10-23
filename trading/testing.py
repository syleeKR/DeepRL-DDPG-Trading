import matplotlib.pyplot as plt
from learners import *
from environment import *
from utils import save_snapshot, recover_snapshot, load_model



def test(agent, env, path = None):
    if path is not None:

        load_model(agent, path, device="cpu")

    sum_scores = 0.


    store_pv = []
    store_reward = [0]
    price = []
    actions = []
    check = []

    obs = env.reset()
    done = False
    score = 0.
    with torch.no_grad():
        agent.actor.eval()
        while not done:
            action = agent.act(obs,2)
            obs, rew, done, _ = env.step(action)
            score += rew

            store_pv.append(env.portfolio_value)
            store_reward.append(store_reward[-1] + rew)
            price.append(obs[0])
            actions.append(action)
            check.append(rew - (env.portfolio_value-env.past_portfolio_value)/env.past_portfolio_value)
    del price[-1]

    meanprice =0
    '''
    std = 0
    for i in range(len(price)):
        meanprice += price[i]
    meanprice = meanprice / len(price)
    for i in range(len(price)):
        std += (price[i] - meanprice)**2
    std = std/len(price)

    for i in range(len(price)):
        price[i] = (price[i]-meanprice)/(std**0.5)
    '''
    plt.plot(store_pv)
    plt.show()
    plt.plot(store_reward)
    plt.show()
    plt.plot(price)
    plt.show()
    #plt.show()
    plt.plot(actions)
    plt.show()
    #plt.plot(check)
    #plt.show()



    sum_scores += score

    print(store_pv[-1])
    return store_pv, actions