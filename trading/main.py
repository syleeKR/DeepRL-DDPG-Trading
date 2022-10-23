
import data_manager
from learners import *
from environment import *
from testing import *


if __name__ == '__main__':

    #print(torch.cuda.is_available())

    chart_data, training_data = data_manager.load_data('C:/Users/sylee/PycharmProjects/pythonProject2/rltrader-master/quantylab/rltrader/data/BTCUSDT/train.csv')

    initial_balance = 100
    env = Environment(initial_balance, chart_data, training_data)



    chart_data_test1, training_data_test1 = data_manager.load_data('C:/Users/sylee/PycharmProjects/pythonProject2/rltrader-master/quantylab/rltrader/data/BTCUSDT/test1.csv')
    testenv1 = Environment(initial_balance, chart_data_test1, training_data_test1)
    chart_data_test2, training_data_test2 = data_manager.load_data(
        'C:/Users/sylee/PycharmProjects/pythonProject2/rltrader-master/quantylab/rltrader/data/BTCUSDT/test2.csv')
    testenv2 = Environment(initial_balance, chart_data_test2, training_data_test2)
    chart_data_test3, training_data_test3 = data_manager.load_data(
        'C:/Users/sylee/PycharmProjects/pythonProject2/rltrader-master/quantylab/rltrader/data/BTCUSDT/test3.csv')
    testenv3 = Environment(initial_balance, chart_data_test3, training_data_test3)
    chart_data_test4, training_data_test4 = data_manager.load_data(
        'C:/Users/sylee/PycharmProjects/pythonProject2/rltrader-master/quantylab/rltrader/data/BTCUSDT/test4.csv')
    testenv4 = Environment(initial_balance, chart_data_test4, training_data_test4)


    rlagent = DDPGAgent(obs_dim = env.observation_space_dim, act_dim = env.action_space_dim, ctrl_range=env.action_space_high)


    gamma = 0.99
    actor_lr = 2e-5
    critic_lr = 2e-5
    tau = 1e-3
    noise_std = 2
    ep_len = 100000000000
    num_updates = 10000 * 30
    batch_size = 256

    '''
    train(rlagent, env =env, gamma =gamma, actor_lr=actor_lr, critic_lr=critic_lr, tau=tau, noise_std=noise_std, ep_len=ep_len,
          num_updates=num_updates, batch_size=batch_size,init_buffer=10000, buffer_size=50000,
              start_train=10000, train_interval=200,
              eval_interval=1000, snapshot_interval=25000, warmup=20000, path=None)

    '''
    l1, a1=test(agent= rlagent, env = testenv1, path ='C:/Users/sylee/PycharmProjects/pythonProject2/rltrader-master/quantylab/rltrader/snapshots/iter200000_model.pth.tar')
    l2, a2=test(agent= rlagent, env = testenv2, path ='C:/Users/sylee/PycharmProjects/pythonProject2/rltrader-master/quantylab/rltrader/snapshots/iter200000_model.pth.tar')
    l3, a3=test(agent= rlagent, env = testenv3, path ='C:/Users/sylee/PycharmProjects/pythonProject2/rltrader-master/quantylab/rltrader/snapshots/iter200000_model.pth.tar')
    l4, a4=test(agent= rlagent, env = testenv4, path ='C:/Users/sylee/PycharmProjects/pythonProject2/rltrader-master/quantylab/rltrader/snapshots/iter200000_model.pth.tar')


    plt.subplot(221)
    plt.plot(l1)
    #plt.title('Test1')
    plt.xlabel('n')
    plt.ylabel('P')


    plt.subplot(222)
    plt.plot(l2)
    #plt.title('Test2')
    plt.xlabel('n')
    plt.ylabel('P')

    plt.subplot(223)
    plt.plot(l3)
    #plt.title('Test3')
    plt.xlabel('n')
    plt.ylabel('P')

    plt.subplot(224)
    plt.plot(l4)
    #plt.title('Test4')
    plt.xlabel('n')
    plt.ylabel('P')
    plt.show()

    plt.subplot(221)
    plt.plot(a1)
    # plt.title('Test1')
    plt.xlabel('n')
    plt.ylabel('leverage')

    plt.subplot(222)
    plt.plot(a2)
    # plt.title('Test2')
    plt.xlabel('n')
    plt.ylabel('leverage')

    plt.subplot(223)
    plt.plot(a3)
    # plt.title('Test3')
    plt.xlabel('n')
    plt.ylabel('leverage')

    plt.subplot(224)
    plt.plot(a4)
    # plt.title('Test4')
    plt.xlabel('n')
    plt.ylabel('leverage')
    plt.show()

    '''
    test(agent= rlagent, env = testenv1, path =None  )
    test(agent= rlagent, env = testenv2, path = None)
    test(agent= rlagent, env = testenv3, path = None)
    test(agent= rlagent, env = testenv4, path = None)

    '''






