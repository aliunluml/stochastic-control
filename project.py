import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp



def main():
    random_generator=np.random.default_rng(seed=1)
    etas=np.array([0.9,0.7,0.01])
    eta=etas[0]
    beta=0.5

    U=np.array([1,0])
    X=np.array(['G','B'])

    P_G=np.array([0.1,0.95,0.75,0.5])
    P_B=1-P_G
    P=np.vstack((P_G,P_B)).reshape((2,2,2))

    print(P)
    print(P[0,1,0])

    def pmf(x_star,x_t,u_t):
        return P[np.where(X==x_star),np.where(X==x_t),np.where(U==u_t)][0,0]

    def cost(x_t,u_t):
        result=eta*u_t
        if (x_t=='G') & (u_t==1):
            result-=1
        return result

    vcost=np.vectorize(cost)

    def expected_val(func,dist,space):
        for x_t,u_t in zip(X,U):
            pmf(x_t,u_t)
        return result

    def value_iteration():
        # Discounted value at infinity is 0 for both G and B
        vs=[np.zeros(2)]
        # Assume the first optimal action is 0 for both G and B
        us=[np.zeros(2)]
        t=0
        while (t==0) or (vs[t]!=vs[t-1]).any():
            print(t)

            v=np.vstack((np.transpose(beta*P[:,:,1])@vs[t]+vcost(X,u_t=0),np.transpose(beta*P[:,:,0])@vs[t]+vcost(X,u_t=1)))
            vs.append(np.min(v,axis=0))
            # print(np.min(v,axis=0))
            us.append(np.argmin(v,axis=0))
            t+=1

        return (vs,us)



    def policy_iteration():
        # Discounted value at infinity is 0 for both G and B
        vs=[np.zeros(2)]
        # Assume the first optimal action is 0 for both G and B
        us=[np.zeros(2)]
        t=0
        while (t==0) or (us[t]!=us[t-1]).any():
            print(t)
            # Policy evaluation
            c=np.array([cost(x,u) for x,u in zip(X,us[t])])
            v=np.linalg.inv(np.eye(len(X))-beta*np.vstack([P[:,i,list(U).index(us[t][i])] for i in range(len(X))]))@c
            vs.append(v)
            # print(v)
            # print(P[:,:,0])
            # Policy improvement
            q=np.vstack([np.transpose(beta*P[:,:,1])@v+vcost(X,u_t=0),(np.transpose(beta*P[:,:,0])@v+vcost(X,u_t=1))])
            u=np.argmin(q,axis=0)
            us.append(u)
            t+=1

        return (vs,us)


    def bayes_rule(joint_probs):
        # Optimal policy at state x_t
        posterior=np.zeros((2,2))
        for x,u in zip(X,U):
            u_index=list(U).index(u)
            x_index=list(X).index(x)
            posterior[u_index,x_index]=joint_probs[u_index,x_index]/np.sum(joint_probs[:,x_index])

        return posterior


    def convex_analysis():
        # Construct the problem.
        cost=np.vstack((vcost(X,u_t=1),vcost(X,u_t=0)))
        ipm = cp.Variable((2,2))
        objective = cp.Minimize(cp.sum(cp.multiply(ipm,cost)))
        constraints = [0 <= ipm,
                       ipm <= 1,
                       cp.sum(ipm)==1,
                       cp.sum(ipm[:,0])==cp.sum(cp.multiply(P[0,:,:].T,ipm)),
                       cp.sum(ipm[:,1])==cp.sum(cp.multiply(P[1,:,:].T,ipm))]
        prob = cp.Problem(objective, constraints)

        # The optimal objective value is returned by `prob.solve()`.
        min_avg_cost = prob.solve()
        # The optimal value for x is stored in `x.value`.
        print(min_avg_cost)

        u_index=np.argmax(bayes_rule(np.array(ipm.value)),axis=0)
        u=np.array([U[u_index[0]],U[u_index[1]]])
        return u


    def q_learning():
        q_values=[random_generator.normal(size=(2,2))]
        cost=np.vstack((vcost(X,u_t=1),vcost(X,u_t=0)))
        # Assume the first optimal state is G for both actions 0 and 1
        xs=np.array([np.zeros(2)])
        # Assume the first optimal action is 0 for both G and B
        us=np.array([np.zeros(2)])
        t=0
        while (t==0) or (q_values[t]!=q_values[t-1]).any():
            print(t)
            alpha=np.zeros((2,2))
            for x,u in zip(X,U):
                u_index=list(U).index(u)
                x_index=list(X).index(x)
                alpha[u_index,x_index]=1/(1+sum([(optimal_x==x) and (optimal_u==u) for optimal_x,optimal_u in zip(xs,us)]))

            q_value_new=q_values[t]+alpha*(cost+beta*np.tile(np.min(q_values[t],axis=0),(2,1))-q_values[t])
            q_values.append(q_value_new)

            x_index=np.argmin(q_value_new,axis=1)
            x=np.array([X[x_index[0]],X[x_index[1]]])
            np.append(xs,[x],axis=0)

            u_index=np.argmin(q_value_new,axis=0)
            u=np.array([U[u_index[0]],U[u_index[1]]])
            np.append(us,[u],axis=0)
            t+=1

        return (xs,us)

    def kalman_filter():
        ipm_means=[np.zeros(4)]
        ipm_covariances=[np.eye(4)]
        A=np.array([[1.2,1,0,0],
                    [0,1.2,1,0],
                    [0,0,1.2,1],
                    [0,0,0,1.5]])
        W=np.eye(4)
        c=np.array([[2,0,0,0]])
        xs=[random_generator.normal(size=4)]
        for t in range(500):

            w_t=random_generator.normal(size=4)
            v_t=random_generator.normal(size=1)
            x_new=A@xs[t]+w_t
            xs.append(x_new)
            # y_t=c@xs[t]+v_t This is erroneous.
            y_t=c@x_new+v_t

            m_x=ipm_means[t]
            K_xx=ipm_covariances[t]

            K_xy=(A@K_xx@c.T)
            # v_t is from np.randn so its variance is 1
            K_yy_inv=np.linalg.inv(c@K_xx@c.T+1)
            K_xx_new=A@K_xx@A.T+W-K_xy@K_yy_inv@K_xy.T
            ipm_covariances.append(K_xx_new)

            m_x_new=A@m_x+K_xx@c.T@K_yy_inv@(y_t-c@A@m_x)
            ipm_means.append(m_x_new)

        return (xs,ipm_means,ipm_covariances)


    values,actions=value_iteration()

    print(values[-1])
    print(actions[-1])

    values,actions=policy_iteration()
    print(values[-1])
    print(actions[-1])

    actions=convex_analysis()
    print(actions)

    _,actions=q_learning()
    print(actions[-1])

    # Data for plotting
    xs,ms,covs=kalman_filter()
    xs=np.array(xs)
    ms=np.array(ms)
    diffs=xs-ms
    print("Values in the first and last two iterations:")
    print("xs")
    print(xs[np.r_[0:2, -2:0]])
    print("ms")
    print(ms[np.r_[0:2, -2:0]])
    print("diffs")
    print(diffs[np.r_[0:2, -2:0]])
    print("covs")
    print(covs[-1])


    iters=np.arange(0,501,1)

    fig, axs = plt.subplots(4,2,sharex=True)
    for i in range(4):
        axs[i,0].plot(iters, xs[:,i],label='x_t',color='c')
        axs[i,0].plot(iters, ms[:,i],label='m_t',color='b',linestyle='dashed')
        axs[i,1].plot(iters, diffs[:,i],label='x_t-m_t',color='r')

        axs[i,0].set(ylabel='value',title='dim='+str(i))
        axs[i,0].set_yscale('log')
        for j in range(2):
            axs[i,j].legend()
            axs[i,j].grid()
            if i==3:
                axs[i,j].set(xlabel='iteration')


    fig.suptitle("Kalman Filtering")
    fig.savefig("test.png")
    plt.show()

if __name__=="__main__":
    main()
