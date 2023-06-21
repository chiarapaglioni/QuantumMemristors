import numpy as np
from q_memristor.numerical.num_memristor import memristor
from time_plot import Tplot
from iv_plot import IVplot

"""
    Comparison of expected estimation of voltage and current with the ones derived by the numerical calculations. 
    Compute margin or error. 

    Author: Chiara Paglioni
"""


def get_margin_error(correct, computed):
    """
        Compute margin of error based on the absolute difference of correct and estimated values.
    """
    abs_diff = np.abs(np.array(correct) - np.array(computed))

    average_err = np.mean(abs_diff)
    max_err = np.max(abs_diff)

    return average_err, max_err


if __name__ == '__main__':
    # Estimated approximations of current and voltage
    V = [-0.2, -0.3, -0.36, -0.25, 0, 0.05, 0.06, 0.03, 0.01, 0, -0.1]
    I = [-0.01, -0.015, -0.05, -0.1, 0, 0.06, 0.1, 0.05, 0.03, 0, 0]
    exp = [0.6, 0.85, 1.03, 0.714, 0, -0.143, -0.171, -0.08, -0.028, 0, 0.285]
    t = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    I_mem = []
    V_mem = []
    exp_mem = []
    exp_pauliY = []

    a = np.pi / 4
    b = np.pi / 5
    m = 1
    h = 1
    w = 1
    y0 = 0.4

    mem = memristor(y0, w, h, m, a, b)

    t_plot = Tplot()
    iv_plot = IVplot(V[0], I[0])

    for i in range(len(t)-1):
        t_plot.update(t[i], V[i], I[i])
        iv_plot.update(V[i], I[i])

        i_memVal = mem.gamma(t[i])*V[i]
        I_mem.append(i_memVal)
        print('Time: ', t[i])
        print('Current --> Manual: ', I[i], ' Memristor: ', I_mem[i])

        k_val = mem.k(t[i], t[i + 1])
        print('K: ', k_val)

        pI_mat = mem.get_density_mat(k_val)
        pI_mat_Krauss = mem.get_density_mat2(t[i])
        print('p_I Density Matrix: ')
        print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in pI_mat]))
        print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in pI_mat_Krauss]))
        print()

        p2_mat = mem.get_Schrodinger(t[i], pI_mat)
        p2_mat_Krauss = mem.get_Schrodinger(t[i], pI_mat_Krauss)
        print('p_2 Schrödinger Matrix: ')
        print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in p2_mat]))
        print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in p2_mat_Krauss]))
        print()

        pauli_y = np.array([[0, -1j], [1j, 0]])

        exp_value1 = np.trace(p2_mat)
        exp_value2 = np.trace(np.dot(p2_mat, pauli_y))
        exp_value3 = np.trace(np.dot(p2_mat_Krauss, pauli_y))

        exp_mem.append(exp_value1)
        exp_pauliY.append(exp_value2)
        print('Trace Exp Value --> Estimate: ', exp[i], ' Memristor: ', exp_mem[i])
        print('Pauli Y Exp Value --> Estimate: ', exp[i], ' Memristor: ', exp_pauliY[i])
        print('Pauli Y + Krauss Exp Value --> Estimate: ', exp[i], ' Memristor: ', exp_value3)
        print()

        v_memVal1 = -(1 / 2) * np.sqrt((m * h * w) / 2) * exp_value1
        v_memVal2 = -(1 / 2) * np.sqrt((m * h * w) / 2) * exp_value2
        V_mem.append(v_memVal1)
        print('Voltage --> Estimate: ', V[i], ' Memristor: ', v_memVal1, ' or ', v_memVal2)
        print()

    # Compute margin of error
    V_temp = [-0.2, -0.3, -0.36, -0.25, 0, 0.05, 0.06, 0.03, 0.01, 0]
    I_temp = [-0.01, -0.015, -0.05, -0.1, 0, 0.06, 0.1, 0.05, 0.03, 0]
    exp_temp = [0.6, 0.85, 1.03, 0.714, 0, -0.143, -0.171, -0.08, -0.028, 0]

    v_avg_err, v_max_err = get_margin_error(V_temp, V_mem)
    i_avg_err, i_max_err = get_margin_error(I_temp, I_mem)
    exp_avg_err, exp_max_err = get_margin_error(exp_temp, exp_mem)
    print('Average Error Voltage: ', v_avg_err, ' Max Error Voltage: ', v_max_err)
    print('Average Error Current: ', i_avg_err, ' Max Error Current: ', i_max_err)
    print('Average Error Expectation Value: ', exp_avg_err, ' Max Error Expectation Value: ', exp_max_err)

    t_plot.save_plot()
    iv_plot.save_plot()
