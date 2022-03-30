import numpy as np
import time
import matplotlib.pyplot as plt
import openpyxl

wb = openpyxl.Workbook()
sheet = wb.active
ws1 = wb.create_sheet(index=0, title='数据集')


def showcontourf(lx1, lx2, ly1, My, mat, D, cmap=plt.cm.get_cmap('jet'), fsize=(12, 12), vmin=0, vmax=100):
    plt.clf()
    levels = np.arange(vmin, vmax, 1)
    x = np.linspace(D[0], D[1], mat.shape[1])
    y = np.linspace(D[2], D[3], mat.shape[0])
    X, Y = np.meshgrid(x, y)
    z_max = np.max(mat)
    i_max, j_max = np.where(mat == z_max)[0][0], np.where(mat == z_max)[1][0]
    show_max = "U_max: {:.1f}".format(z_max)
    plt.plot(x[j_max], y[i_max], 'ro')
    Z = mat.copy()
    Z[ly1 + 1:My + 1, 0:lx1] = Z[ly1 + 1:My + 1, lx2 + 1:-1] = Z[ly1 + 1:My + 1, -1] = Z[My - 2:My, :] = np.nan  # 限定范围
    plt.contourf(X, Y, Z, 100, cmap=cmap, origin='lower', levels=levels)
    plt.annotate(show_max, xy=(x[j_max], y[i_max]), xytext=(x[j_max], y[i_max]), fontsize=14)
    plt.colorbar()
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.axis('equal')
    plt.draw()
    plt.pause(0.001)
    plt.clf()


def Data_maker(lx1, lx2, ly1, u_gas, u_env, a):
    D = np.array([0, 12, 0, 6])

    Mx = 120
    My = 60

    T, Tn = 10000, 5
    Nt = 1000 * 10000

    dx = (D[1] - D[0]) / Mx
    dy = (D[3] - D[2]) / My
    dt = 200
    Frame = 100
    fq = 10
    u0 = u_env

    c1 = -10304000333333333333
    c2 = -6544384752

    ### 初始化U、A、B矩阵，各点初始温度u0
    U = u0 * np.ones((My + 1, Mx + 1))
    # U[ly1 + 1:My, 0:lx1] = U[ly1 + 1:My, lx2 + 1:-1] = np.nan
    # x方向二阶导系数矩阵A
    A = (-2) * np.eye(Mx + 1, k=0) + (1) * np.eye(Mx + 1, k=-1) + (1) * np.eye(Mx + 1, k=1)
    A[0, -1] = A[-1, 0] = 1  # 周期边界
    # y方向二阶导系数矩阵B
    B = (-2) * np.eye(My + 1, k=0) + (1) * np.eye(My + 1, k=-1) + (1) * np.eye(My + 1, k=1)

    rx, ry, ft = a * dt / dx ** 2, a * dt / dy ** 2, fq * dt
    heat = 0
    start = time.time()
    ### 按时间增量逐次计算
    for k in range(Nt + 1):
        tt = k * dt
        # solve inside nodes
        U = U + rx * np.dot(U, A) + ry * np.dot(B, U) + heat * ft
        # solve boundary nodes
        ch1 = 20/2000  # 热导率/对流换热系数h
        ch2 = 20/1500
        U[ly1:My, lx1 - 1] = (u_env + ch2 * U[ly1:My, lx1] / dx) / (1 + ch2 / dx)  # 冷流体侧 竖向
        U[ly1:My, lx2 - 1] = (u_env + ch2 * U[ly1:My, lx2] / dx) / (1 + ch2 / dx)

        U[0, :] = (u_gas + ch1 * U[1, :] / dy) / (1 + ch1 / dy)  # 热流体侧
        U[ly1 - 1, 0:lx1] = (u_env + ch2 * U[ly1 - 2, 0:lx1] / dy) / (1 + ch2 / dy)  # 冷流体侧 横向 0-x1
        # U[-1, lx1 - 1:lx2] = (u_env + 9999999999 * U[-2, lx1 - 1:lx2] / dy) / (1 + 9999999999 / dy)  # 对称侧 横向 x1-x2 绝热
        U[-1, lx1 - 1:lx2] = U[-2, lx1 - 1:lx2]  # 对称侧 横向 x1-x2 绝热
        U[ly1 - 1, lx2 - 1:-1] = (u_env + ch2 * U[ly1 - 2, lx2 - 1:-1] / dy) / (1 + ch2 / dy)  # 冷流体侧 横向 x2-x3

        mean_total_0 = np.mean(U)
        std_total_0 = np.std(U)
        min_total_0 = np.min(U)

        # if k % Frame == 0:
        #     end = time.time()
        #     print(
        #         'T = {:.3f} s  max_U= {:.1f}  min_U = {:.1f}  mean_U = {:.1f}  std_U = {:.1f}  heat = {:.1f}  process time = {:.1f}'.format(
        #             tt,
        #             np.max(
        #                 U),
        #             np.min(
        #                 U), np.mean(U), np.std(U), heat,
        #
        #             end - start))

        if abs(mean_total_0 - c1) < 0.001 and abs(std_total_0 - c2) < 0.001:
            # print('NHT结束')
            showcontourf(lx1, lx2, ly1, My, U, D, vmax=u_gas * 1.05)
            break
        else:
            c1 = mean_total_0
            c2 = std_total_0
            # print(c1, c2)
            # showcontourf(lx1, lx2, ly1, My, U, D, vmax=u_gas * 1.05)

    # print('记录运算结果')

    U1 = U[0:ly1, 0:lx1]
    U2 = U[:, lx1:lx2]
    U3 = U[0:ly1, lx2:Mx]
    # print(np.shape(U1), np.shape(U2), np.shape(U3))

    mean1 = np.mean(U1, axis=0)
    mid1_t = np.median(U1, axis=0)
    max1_t = np.amax(U1, axis=0)
    min1_t = np.amin(U1, axis=0)

    mean2 = np.mean(U2, axis=0)
    mid2_t = np.median(U2, axis=0)
    max2_t = np.amax(U2, axis=0)
    min2_t = np.amin(U2, axis=0)

    mean3 = np.mean(U3, axis=0)
    mid3_t = np.median(U3, axis=0)
    max3_t = np.amax(U3, axis=0)
    min3_t = np.amin(U3, axis=0)

    # mean_model = np.concatenate([mean1, mean2, mean3])
    mean_model = np.concatenate([mean1 * ly1 / (ly1 + ly1 + My) / lx1, mean2 * My / (ly1 + ly1 + My) / (lx2 - lx1),
                                 mean3 * ly1 / (ly1 + ly1 + My) / (Mx - lx2)])  # 加权平均温度
    mid_model = np.concatenate([mid1_t, mid2_t, mid3_t])
    max_model = np.concatenate([max1_t, max2_t, max3_t])
    min_model = np.concatenate([min1_t, min2_t, min3_t])

    # print(np.shape(mean_model))

    # result = np.mean(mean_model)
    result = np.sum(mean_model)
    # print(result)

    list1 = result.tolist()

    full_list = [lx1, lx2, ly1, u_gas, u_env, a, list1]

    ws1.append(full_list)


i = 0
start0 = time.time()
lx1_scope = range(5, 55, 20)
lx2_scope = range(5, 55, 20)
ly1_scope = range(5, 55, 20)
u_gas_scope = range(1800, 2400, 500)
u_env_scope = range(700, 900, 100)
total_len_scope = len(lx2_scope) * len(lx2_scope) * len(ly1_scope) * len(u_gas_scope) * len(u_env_scope)

for lx1 in lx1_scope:
    for lx2 in lx2_scope:
        for ly1 in ly1_scope:
            for u_gas in u_gas_scope:
                for u_env in u_env_scope:
                    start = time.time()
                    Data_maker(lx1, lx2 + lx1, ly1, u_gas, u_env, a=0.000006)
                    i += 1
                    end = time.time()
                    prid_time = (end - start0) / (i / total_len_scope)
                    print('进度：{3:.2%}  预计{5:.0f}秒后完成  总循环数：{6}  已完成循环数：{0}  单次循环用时：{1:.3f}秒  预计总用时：{4:.0f}秒'
                          .format(i, end - start, end - start0, i / total_len_scope, prid_time,
                                  prid_time - (end - start0), total_len_scope))

wb.save(filename='mean.xlsx')
