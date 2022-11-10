import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import sys


def changes(m, t, r):
    """ Функция оценки принятия решения на базе нечеткой логики.
        Что лучше потратить 3,5 млн и 2 мес с риском или 0.5 млн и 4 мес без рисков?
        usage: ./changes.py money time risk(0/1)
    """
    base_filename = str(m) + '_' + str(t) + '_' + str(r)
    m = float(m)
    t = float(t)

    # Generate universe variables
    #   * Money has a range of [0, 13] in units of millions points
    #   * Time has a range of [0, 7] in units of months points
    #   * Decision has a range of [0, 100] in units of percentage points
    x_money = np.arange(0, 14, 0.1)
    x_time = np.arange(0, 8, 0.1)
    x_decision = np.arange(0, 101, 1)

    # Generate fuzzy membership functions
    money_stf = fuzz.zmf(x_money, 1, 6)                               # Z-function
    money_moder = fuzz.trimf(x_money, [1, 6, 11])                     # triangle-function
    money_ntapl = fuzz.smf(x_money, 6, 11)                            # S-function
    time_stf = fuzz.zmf(x_time, 1, 3)
    time_moder = fuzz.trimf(x_time, [1, 3, 5])
    time_ntapl = fuzz.smf(x_time, 3, 5)
    decision_cancelsure = fuzz.zmf(x_decision, 3, 5)
    decision_cancelrevision = fuzz.trapmf(x_decision, [0, 10, 14, 30])        # trapezoid-function
    decision_apprestric = fuzz.trapmf(x_decision, [20, 50, 55, 80])
    decision_appcaution= fuzz.trapmf(x_decision, [70, 86, 90, 100])
    decision_appsure = fuzz.smf(x_decision, 95, 97)

    # Visualize these universes and membership functions
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

    ax0.plot(x_money, money_stf, 'b', linewidth=1.5, label='satisfied')
    ax0.plot(x_money, money_moder, 'g', linewidth=1.5, label='moderate')
    ax0.plot(x_money, money_ntapl, 'r', linewidth=1.5, label='not applicable')
    ax0.set_title('Money')
    ax0.legend()

    ax1.plot(x_time, time_stf, 'b', linewidth=1.5, label='satisfied')
    ax1.plot(x_time, time_moder, 'g', linewidth=1.5, label='moderate')
    ax1.plot(x_time, time_ntapl, 'r', linewidth=1.5, label='not applicable')
    ax1.set_title('Time')
    ax1.legend()

    ax2.plot(x_decision, decision_cancelsure, 'b', linewidth=1.5, label='отказаться с уверенностью')
    ax2.plot(x_decision, decision_cancelrevision, 'g', linewidth=1.5, label='отклонить для пересмотра')
    ax2.plot(x_decision, decision_apprestric, 'r', linewidth=1.5, label='принять с ограничениями')
    ax2.plot(x_decision, decision_appcaution, 'm', linewidth=1.5, label='принять с осторожносью')
    ax2.plot(x_decision, decision_appsure, 'orange', linewidth=1.5, label='точно принять')
    ax2.set_title('Decision')
    ax2.legend()

    # Turn off top/right axes
    for ax in (ax0, ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    # Save result in a picture
    plt.savefig(base_filename + '_' + '1.png')

    # We need the activation of our fuzzy membership functions at these values.
    # The exact values "m" and "t" do not exist on our universes...
    # This is what fuzz.interp_membership exists for!

    S1 = fuzz.interp_membership(x_money, money_stf, m)
    S2 = fuzz.interp_membership(x_money, money_moder, m)
    S3 = fuzz.interp_membership(x_money, money_ntapl, m)

    T1 = fuzz.interp_membership(x_time, time_stf, t)
    T2 = fuzz.interp_membership(x_time, time_moder, t)
    T3 = fuzz.interp_membership(x_time, time_ntapl, t)


    # Now we take our rules and apply them.
    # Rule 1 if money (m) is satisfied AND time (t) is satisfied then decision is applicable.
    # The AND operator means we take the minimum of "m" and "t".
    # Now we apply this by clipping the top off the corresponding output membership function with "np.fmin"
    if int(r) == 0:
        active_rule1 = np.fmin(np.fmin(S1, T1), decision_appsure)
        active_rule2 = np.fmin(np.fmin(S1, T2), decision_appcaution)
        active_rule3 = np.fmin(np.fmin(S1, T3), decision_cancelrevision)
        active_rule4 = np.fmin(np.fmin(S2, T1), decision_appcaution)
        active_rule5 = np.fmin(np.fmin(S2, T2), decision_apprestric)
        active_rule6 = np.fmin(np.fmin(S2, T3), decision_cancelrevision)
        active_rule7 = np.fmin(np.fmin(S3, T1), decision_cancelrevision)
        active_rule8 = np.fmin(np.fmin(S3, T2), decision_cancelrevision)
        active_rule9 = np.fmin(np.fmin(S3, T3), decision_cancelsure)
        tip0 = np.zeros_like(x_decision)  # For visualization to fill area

        # Visualize this
        fig, ax0 = plt.subplots(figsize=(8, 3))

        ax0.fill_between(x_decision, tip0, active_rule1, facecolor='b', alpha=0.7)
        ax0.plot(x_decision, decision_appsure, 'b', linewidth=0.5, linestyle='--', )
        ax0.fill_between(x_decision, tip0, active_rule2, facecolor='g', alpha=0.7)
        ax0.plot(x_decision, decision_appcaution, 'g', linewidth=0.5, linestyle='--')
        ax0.fill_between(x_decision, tip0, active_rule3, facecolor='r', alpha=0.7)
        ax0.plot(x_decision, decision_cancelrevision, 'r', linewidth=0.5, linestyle='--')
        ax0.fill_between(x_decision, tip0, active_rule4, facecolor='olive', alpha=0.7)
        ax0.plot(x_decision, decision_appcaution, 'olive', linewidth=0.5, linestyle='--')
        ax0.fill_between(x_decision, tip0, active_rule5, facecolor='navy', alpha=0.7)
        ax0.plot(x_decision, decision_apprestric, 'navy', linewidth=0.5, linestyle='--')
        ax0.fill_between(x_decision, tip0, active_rule6, facecolor='tan', alpha=0.7)
        ax0.plot(x_decision, decision_cancelrevision, 'tan', linewidth=0.5, linestyle='--')
        ax0.fill_between(x_decision, tip0, active_rule7, facecolor='m', alpha=0.7)
        ax0.plot(x_decision, decision_cancelrevision, 'm', linewidth=0.5, linestyle='--')
        ax0.fill_between(x_decision, tip0, active_rule8, facecolor='pink', alpha=0.7)
        ax0.plot(x_decision, decision_cancelrevision, 'pink', linewidth=0.5, linestyle='--')
        ax0.fill_between(x_decision, tip0, active_rule9, facecolor='orange', alpha=0.7)
        ax0.plot(x_decision, decision_cancelsure, 'orange', linewidth=0.5, linestyle='--')

        ax0.set_title('Output Decision Activity')

        # Turn off top/right axes
        for ax in (ax0,):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

        # Save result in a picture
        plt.savefig(base_filename + '_' + '2.png')


    else:
        active_rule1 = np.fmin(np.fmin(S1, T1), decision_appcaution)
        active_rule2 = np.fmin(np.fmin(S1, T2), decision_apprestric)
        active_rule3 = np.fmin(np.fmin(S1, T3), decision_cancelsure)
        active_rule4 = np.fmin(np.fmin(S2, T1), decision_apprestric)
        active_rule5 = np.fmin(np.fmin(S2, T2), decision_cancelrevision)
        active_rule6 = np.fmin(np.fmin(S2, T3), decision_cancelsure)
        active_rule7 = np.fmin(np.fmin(S3, T1), decision_cancelsure)
        active_rule8 = np.fmin(np.fmin(S3, T2), decision_cancelsure)
        active_rule9 = np.fmin(np.fmin(S3, T3), decision_cancelsure)

        tip0 = np.zeros_like(x_decision)  # For visualization to fill area

        # Visualize this
        fig, ax0 = plt.subplots(figsize=(8, 3))

        ax0.fill_between(x_decision, tip0, active_rule1, facecolor='b', alpha=0.7)
        ax0.plot(x_decision, decision_appcaution, 'b', linewidth=0.5, linestyle='--', )
        ax0.fill_between(x_decision, tip0, active_rule2, facecolor='g', alpha=0.7)
        ax0.plot(x_decision, decision_apprestric, 'g', linewidth=0.5, linestyle='--')
        ax0.fill_between(x_decision, tip0, active_rule3, facecolor='r', alpha=0.7)
        ax0.plot(x_decision, decision_cancelsure, 'r', linewidth=0.5, linestyle='--')
        ax0.fill_between(x_decision, tip0, active_rule4, facecolor='olive', alpha=0.7)
        ax0.plot(x_decision, decision_apprestric, 'olive', linewidth=0.5, linestyle='--')
        ax0.fill_between(x_decision, tip0, active_rule5, facecolor='navy', alpha=0.7)
        ax0.plot(x_decision, decision_cancelrevision, 'navy', linewidth=0.5, linestyle='--')
        ax0.fill_between(x_decision, tip0, active_rule6, facecolor='tan', alpha=0.7)
        ax0.plot(x_decision, decision_cancelsure, 'tan', linewidth=0.5, linestyle='--')
        ax0.fill_between(x_decision, tip0, active_rule7, facecolor='m', alpha=0.7)
        ax0.plot(x_decision, decision_cancelsure, 'm', linewidth=0.5, linestyle='--')
        ax0.fill_between(x_decision, tip0, active_rule8, facecolor='pink', alpha=0.7)
        ax0.plot(x_decision, decision_cancelsure, 'pink', linewidth=0.5, linestyle='--')
        ax0.fill_between(x_decision, tip0, active_rule9, facecolor='orange', alpha=0.7)
        ax0.plot(x_decision, decision_cancelsure, 'orange', linewidth=0.5, linestyle='--')

        ax0.set_title('Output Decision Activity')

        # Turn off top/right axes
        for ax in (ax0,):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

        # Save result in a picture
        plt.savefig(base_filename + '_' + '2.png')



    # Aggregate all three output membership functions together
    # aggregated = np.fmax(active_rule9,
    #                     np.fmax(active_rule8, active_rule7))

    active_rule_list = [active_rule1, active_rule2, active_rule3, active_rule4, active_rule5, active_rule6, active_rule7, active_rule8, active_rule9]
    aggregated = active_rule_list.pop(0)
    for a in active_rule_list:
        aggregated = np.fmax(aggregated, a)

    # Calculate defuzzified result
    decision = fuzz.defuzz(x_decision, aggregated, 'centroid')
    decision_activation = fuzz.interp_membership(x_decision, aggregated, decision)  # for plot


    # Write result in file
    output_file = "decision_result.txt"
    output_file = open(output_file, 'a')
    output_file.write(str(m) + " " + str(t) + " " + str(r) + " " + str(decision) + "\n")
    output_file.close()

    # Visualize this
    fig, ax0 = plt.subplots(figsize=(8, 3))

    ax0.plot(x_decision, decision_cancelsure, 'b', linewidth=0.5, linestyle='--', )
    ax0.plot(x_decision, decision_cancelrevision, 'g', linewidth=0.5, linestyle='--')
    ax0.plot(x_decision, decision_apprestric, 'r', linewidth=0.5, linestyle='--')
    ax0.plot(x_decision, decision_appcaution, 'm', linewidth=0.5, linestyle='--')
    ax0.plot(x_decision, decision_appsure, 'orange', linewidth=0.5, linestyle='--')
    ax0.fill_between(x_decision, tip0, aggregated, facecolor='Orange', alpha=0.7)
    ax0.plot([decision, decision], [0, decision_activation], 'k', linewidth=1.5, alpha=0.9)
    ax0.text(decision + 1, 0.2, decision)
    ax0.set_title('Aggregated decision and result (line)')

    # Turn off top/right axes
    for ax in (ax0,):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    # Save result in a picture
    plt.savefig(base_filename + '_' + '3.png')

    # Найдем в какую облать решения попал наш ответ
    r_dic = {
            fuzz.interp_membership(x_decision, decision_cancelsure, decision): "отказаться с уверенностью",
            fuzz.interp_membership(x_decision, decision_cancelrevision, decision): "отклонить для пересмотра",
            fuzz.interp_membership(x_decision, decision_apprestric, decision): "принять с ограничениями",
            fuzz.interp_membership(x_decision, decision_appcaution, decision): "принять с осторожносью",
            fuzz.interp_membership(x_decision, decision_appsure, decision): "точно принять"
    }

    r0 = max(r_dic.keys())
    what_to_do = r_dic.get(r0)
    return [decision, what_to_do]


def main():
    if len(sys.argv) != 4:
        print('usage: ./changes.py money time risk(0/1)')
        sys.exit(1)

    m = sys.argv[1]
    t = sys.argv[2]
    r = sys.argv[3]



    print("Money is: " + m, "Time is: " + t, "Risk is: " + r)

    if m.replace('.', '', 1).isdigit() and t.replace('.', '', 1).isdigit():
        decision, what_to_do = changes(m, t, r)
        print("Decision is:  " + str(decision) + " %")
        print(what_to_do)
    else:
        print('parameters are not numbers:')
        sys.exit(1)


if __name__ == '__main__':
    main()
