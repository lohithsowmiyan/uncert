import numpy as np
import matplotlib.pyplot as plt

def plot_model_metrics(accuracies_unc, recalls_unc, f1s_unc, fprs_unc,
                       accuracies_rnd, recalls_rnd, f1s_rnd, fprs_rnd, 
                       accuracies_eu, recalls_eu, f1s_eu, fprs_eu,
                       accuracies_au, recalls_au, f1s_au, fprs_au,num_repeats=10):
    """
    Plots the performance metrics (Accuracy, Recall, F1, False Positive Rate) 
    for both uncertainty and random sampling over multiple repeats.
    
    Parameters:
    - accuracies_unc: List of accuracy scores for uncertainty sampling across repeats.
    - recalls_unc: List of recall scores for uncertainty sampling across repeats.
    - f1s_unc: List of F1 scores for uncertainty sampling across repeats.
    - fprs_unc: List of false positive rates for uncertainty sampling across repeats.
    - accuracies_rnd: List of accuracy scores for random sampling across repeats.
    - recalls_rnd: List of recall scores for random sampling across repeats.
    - f1s_rnd: List of F1 scores for random sampling across repeats.
    - fprs_rnd: List of false positive rates for random sampling across repeats.
    - num_repeats: Number of times the experiment was repeated (default is 10).
    """

    # Convert lists to numpy arrays for easier mean and variance calculation
    accuracies_unc = np.array(accuracies_unc)
    recalls_unc = np.array(recalls_unc)
    f1s_unc = np.array(f1s_unc)
    fprs_unc = np.array(fprs_unc)

    accuracies_rnd = np.array(accuracies_rnd)
    recalls_rnd = np.array(recalls_rnd)
    f1s_rnd = np.array(f1s_rnd)
    fprs_rnd = np.array(fprs_rnd)

    accuracies_eu = np.array(accuracies_eu)
    recalls_eu = np.array(recalls_eu)
    f1s_eu = np.array(f1s_eu)
    fprs_eu = np.array(fprs_eu)

    accuracies_au = np.array(accuracies_au)
    recalls_au = np.array(recalls_au)
    f1s_au = np.array(f1s_au)
    fprs_au = np.array(fprs_au)

    # Calculate means and variances
    mean_acc_unc = accuracies_unc.mean(axis=0)
    mean_acc_rnd = accuracies_rnd.mean(axis=0)
    mean_acc_eu = accuracies_eu.mean(axis=0)
    mean_acc_au = accuracies_au.mean(axis=0)
    var_acc_unc = accuracies_unc.var(axis=0)
    var_acc_rnd = accuracies_rnd.var(axis=0)
    var_acc_eu = accuracies_eu.var(axis=0)
    var_acc_au = accuracies_au.var(axis=0)

    mean_rec_unc = recalls_unc.mean(axis=0)
    mean_rec_rnd = recalls_rnd.mean(axis=0)
    mean_acc_eu = recalls_eu.mean(axis=0)
    mean_acc_au = recalls_au.mean(axis=0)
    var_rec_unc = recalls_unc.var(axis=0)
    var_rec_rnd = recalls_rnd.var(axis=0)
    var_acc_eu = recalls_eu.var(axis=0)
    var_acc_au = recalls_au.var(axis=0)

    mean_f1_unc = f1s_unc.mean(axis=0)
    mean_f1_rnd = f1s_rnd.mean(axis=0)
    mean_f1_eu = f1s_eu.mean(axis=0)
    mean_f1_au = f1s_au.mean(axis=0)
    var_f1_unc = f1s_unc.var(axis=0)
    var_f1_rnd = f1s_rnd.var(axis=0)
    var_f1_eu = f1s_eu.var(axis=0)
    var_f1_au = f1s_au.var(axis=0)

    mean_fpr_unc = fprs_unc.mean(axis=0)
    mean_fpr_rnd = fprs_rnd.mean(axis=0)
    mean_fpr_eu = fprs_eu.mean(axis=0)
    mean_fpr_au = fprs_au.mean(axis=0)
    var_fpr_unc = fprs_unc.var(axis=0)
    var_fpr_rnd = fprs_rnd.var(axis=0)
    var_fpr_eu = fprs_eu.var(axis=0)
    var_fpr_au = fprs_au.var(axis=0)

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Metrics over Time: Uncertainty vs Random Sampling')

    # Accuracy
    axs[0, 0].plot(mean_acc_unc, label='Entropy')
    axs[0, 0].plot(mean_acc_rnd, label='Random', linestyle='--')
    axs[0, 0].plot(mean_acc_unc, label='EU')
    axs[0, 0].plot(mean_acc_unc, label='AU')
    axs[0, 0].fill_between(range(len(mean_acc_unc)), mean_acc_unc - np.sqrt(var_acc_unc), mean_acc_unc + np.sqrt(var_acc_unc), alpha=0.2)
    axs[0, 0].fill_between(range(len(mean_acc_rnd)), mean_acc_rnd - np.sqrt(var_acc_rnd), mean_acc_rnd + np.sqrt(var_acc_rnd), alpha=0.2)
    axs[0, 0].fill_between(range(len(mean_acc_eu)), mean_acc_eu - np.sqrt(var_acc_eu), mean_acc_eu + np.sqrt(var_acc_eu), alpha=0.2)
    axs[0, 0].fill_between(range(len(mean_acc_au)), mean_acc_au - np.sqrt(var_acc_au), mean_acc_au + np.sqrt(var_acc_au), alpha=0.2)
    axs[0, 0].set_title('Accuracy')
    axs[0, 0].set_ylabel('Score')
    axs[0, 0].legend()

    # Recall
    axs[0, 1].plot(mean_rec_unc, label='Entropy')
    axs[0, 1].plot(mean_rec_rnd, label='Random', linestyle='--')
    axs[0, 1].fill_between(range(len(mean_rec_unc)), mean_rec_unc - np.sqrt(var_rec_unc), mean_rec_unc + np.sqrt(var_rec_unc), alpha=0.2)
    axs[0, 1].fill_between(range(len(mean_rec_rnd)), mean_rec_rnd - np.sqrt(var_rec_rnd), mean_rec_rnd + np.sqrt(var_rec_rnd), alpha=0.2)
    axs[0, 1].set_title('Recall')
    axs[0, 1].legend()

    # F1
    axs[1, 0].plot(mean_f1_unc, label='Entropy')
    axs[1, 0].plot(mean_f1_rnd, label='Random', linestyle='--')
    axs[1, 0].fill_between(range(len(mean_f1_unc)), mean_f1_unc - np.sqrt(var_f1_unc), mean_f1_unc + np.sqrt(var_f1_unc), alpha=0.2)
    axs[1, 0].fill_between(range(len(mean_f1_rnd)), mean_f1_rnd - np.sqrt(var_f1_rnd), mean_f1_rnd + np.sqrt(var_f1_rnd), alpha=0.2)
    axs[1, 0].set_title('F1 Score')
    axs[1, 0].set_xlabel('Batch Number')
    axs[1, 0].set_ylabel('Score')
    axs[1, 0].legend()

    # False Positive Rate
    axs[1, 1].plot(mean_fpr_unc, label='Entropy')
    axs[1, 1].plot(mean_fpr_rnd, label='Random', linestyle='--')
    axs[1, 1].fill_between(range(len(mean_fpr_unc)), mean_fpr_unc - np.sqrt(var_fpr_unc), mean_fpr_unc + np.sqrt(var_fpr_unc), alpha=0.2)
    axs[1, 1].fill_between(range(len(mean_fpr_rnd)), mean_fpr_rnd - np.sqrt(var_fpr_rnd), mean_fpr_rnd + np.sqrt(var_fpr_rnd), alpha=0.2)
    axs[1, 1].set_title('False Positive Rate')
    axs[1, 1].set_xlabel('Batch Number')
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust space for suptitle
    plt.show()
