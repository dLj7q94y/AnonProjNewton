import matplotlib.pyplot as plt
import utils.utils as utils
import math
import os


# Style keys for each method
FILE_KEY = {"NewtonCGAlt.json": "CG", "NewtonMRTMP.json": "MR", "ProjGrad.json" : "PG", "FISTA.json" : "FISTA", "ProxGradMomentum.json" : "PGM"}

ORDER_KEY = {"NewtonMRTMP.json": 1, "NewtonCGAlt.json": 2, "ProjGrad.json" : 3, "FISTA.json" : 4, "ProxGradMomentum.json" : 5}

STYLE_KEY = {"NewtonCGAlt.json":  {"ls" : "--", "c" : '#ff7f0e'},
                "NewtonMRTMP.json": {"ls" : "-", "c" : '#1f77b4'}, 
                "ProjGrad.json" : {"ls" : "-.", "c" : '#2ca02c'}, 
                "FISTA.json" : {"ls" : ":", "c" : '#d62728'}, 
                "ProxGradMomentum.json" : {"ls": ":", "c" : '#d62728'}}


def get_results(folder):
    results_folder = os.path.join(utils.Results.FOLDER, folder)
    result_files = os.listdir(results_folder)
    result_files.sort(key=lambda x: ORDER_KEY[x] )
    label = [FILE_KEY[x] for x in result_files]

    results = {}
    for file in result_files:
        results[file] = utils.Results.load(results_folder, file).results
    
    return results, label

def plot_full_results(results_folder, eg_hline=None, ops_weighting=None):
    """
    Plots the objective value, step size, termination conditions and size of the active set. 
    """
    ALPHA = 0.8
    results, label = get_results(results_folder)
    
    # Process data for oracle call.
    max_oracle_calls=0
    for r in results.values():
        r["oracle_calls"] = []
        flags = r["Dtype"]
        r["NPC_Flags"] =  [flagk =="NPC" or flagk=="SO-NPC" for flagk in flags]
        r["GRAD_Flags"] =  [flagk =="GRAD" for flagk in flags]
        for t in r["ops"]:
            # Add one oracle call so don't start at 0. 
            r["oracle_calls"].append(1+sum([i*j for i, j in zip(ops_weighting,t) ]))

        max_oracle_calls = max(r["oracle_calls"][-1], max_oracle_calls)        

    fig, axs  = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=[10, 8])
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)

    xlabel = "Oracle Calls"
    
    for r, r_val in results.items():
        colour = STYLE_KEY[r]["c"]
        linestyle = STYLE_KEY[r]["ls"]
    
        plt.subplot(3, 2, 1)
        plt.loglog(r_val["oracle_calls"], r_val["f"], c=colour, ls=linestyle, alpha=ALPHA)
        plt.ylabel("f")

        plt.subplot(3, 2, 2)
        plt.loglog(r_val["oracle_calls"], r_val["alpha"], marker="x", markevery=r_val["NPC_Flags"], alpha=ALPHA, c=colour, ls=linestyle)
        plt.ylabel("alphas")

        plt.subplot(3, 2, 3)
        if "inactive_grad_norm" in r_val:
            plt.loglog(r_val["oracle_calls"], utils.clip_zeros(r_val["inactive_grad_norm"]), c=colour, ls=linestyle, alpha=ALPHA)

        if eg_hline is not None:
            plt.hlines(eg_hline, 1, max_oracle_calls, linestyles="dashed", colors="k")
        plt.ylabel("||g^I||")

        plt.subplot(3, 2, 4)
        if "active_gx_norm" in r_val:
            plt.loglog(r_val["oracle_calls"], utils.clip_zeros(r_val["active_gx_norm"]), c=colour, ls=linestyle, alpha=ALPHA)

        if eg_hline is not None:
            plt.hlines(eg_hline, 1, max_oracle_calls, linestyles="dashed", colors="k")
        plt.ylabel("||X^Ag^A||")

        plt.subplot(3, 2, 5)
        if "active_min_grad" in r_val:
            plt.loglog(r_val["oracle_calls"], utils.clip_zeros(r_val["active_min_grad"]), c=colour, ls=linestyle, alpha=ALPHA)

        if eg_hline is not None:
            plt.hlines(math.sqrt(eg_hline), 1, max_oracle_calls, linestyles="dashed", colors="k")    
        plt.ylabel("-min(g, 0)")
        plt.xlabel(xlabel)

        plt.subplot(3, 2, 6)
        if "num_active_constraints" in r_val:
            plt.semilogx(r_val["oracle_calls"], r_val["num_active_constraints"],c=colour, ls=linestyle, alpha=ALPHA)
            
        plt.xlabel(xlabel)
        plt.ylabel("#e_act-active constraints")

 
    fig.legend(label, loc="lower right")
    fig.tight_layout()
    plt.show()


def plot_example(results_folder, xlabel="Oracle Calls", ylabel="Objective Value", ops_weighting=None, hline=None):
    """
    Plot the objective value at each iteration. Used to generate the main plots.
    """
    if xlabel is None:
        xlabel="Oracle Calls"
    ALPHA = 0.8
    # unpack to raw dictionary.

    results, label = get_results(results_folder)
    # Process data for oracle call.
    max_oracle_calls = 0
    for r in results.values():
        r["oracle_calls"] = []
        for t in r["ops"]:
            # Add one oracle call so don't start at 0. 
            r["oracle_calls"].append(1+sum([i*j for i, j in zip(ops_weighting,t) ]))
            max_oracle_calls = max(r["oracle_calls"][-1], max_oracle_calls)  

    fig, axs  = plt.subplots(nrows=1, ncols=1, figsize=[5, 4])
    for r, r_val in results.items():
        ls = STYLE_KEY[r]["ls"]
        colour = STYLE_KEY[r]["c"]
        plt.loglog(r_val["oracle_calls"], r_val["f"], ls=ls, alpha=ALPHA, c=colour, lw=2.5)
        plt.ylabel(ylabel, fontweight="bold")
        plt.xlabel(xlabel, fontweight="bold")

    if hline is not None:
        plt.hlines(hline, 1, max_oracle_calls, linestyles="dashed", colors="k")   

    plt.tight_layout()
    plt.legend(label)

    return fig

def plot_example_termination_conditions(results_folder, xlabel="Oracle Calls", ops_weighting=None, hline=None):
    """
    Plot the termination conditions. Specifically used to generate the local convergence plots.
    """  
    results_folder = os.path.join(utils.Results.FOLDER, results_folder)

    if xlabel is None:
        xlabel="Oracle Calls"
    ALPHA = 0.5
    # unpack to raw dictionary.

    result_files = os.listdir(results_folder)
    if "FISTA.json" in result_files:
        result_files.remove("FISTA.json")
    if "ProxGradMomentum.json" in result_files:
        result_files.remove("ProxGradMomentum.json")

    result_files.sort(key=lambda x: ORDER_KEY[x] )
    label = [FILE_KEY[x] for x in result_files]
    max_oracle_calls=0

    results = {}
    for file in result_files:
        results[file] = utils.Results.load(results_folder, file).results

    max_oracle_calls = 0
    for r in results.values():
        r["oracle_calls"] = []
        for t in r["ops"]:
            # Add one oracle call so don't start at 0. 
            r["oracle_calls"].append(1+sum([i*j for i, j in zip(ops_weighting,t) ]))
            max_oracle_calls = max(r["oracle_calls"][-1], max_oracle_calls)  

    fig, axs  = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=[10, 4])

    y_keys = ["inactive_grad_norm", "active_min_grad", "active_gx_norm"]
    y_labs = ["(a)", "(b)", "(c)"]
    for i in range(3):
        for r, r_val in results.items():
            if y_keys[i] in r_val.keys():
                ls = STYLE_KEY[r]["ls"]
                colour = STYLE_KEY[r]["c"]
                axs[i].loglog(r_val["oracle_calls"], utils.clip_zeros(r_val[y_keys[i]]), ls=ls, alpha=ALPHA, c=colour, lw=2)
                axs[i].set_ylabel(y_labs[i], fontweight="bold")
                axs[i].set_xlabel(xlabel, fontweight="bold")

        if hline is not None:
            if i == 1:
                axs[i].hlines(math.sqrt(hline), 1, max_oracle_calls, linestyles="dashed", colors="k")  
            else:
                axs[i].hlines(hline, 1, max_oracle_calls, linestyles="dashed", colors="k")   

    plt.tight_layout()
    plt.legend(label)

    return fig

def plot_top_words(results_folder, n_col=4, n_row=5, example="NewtonMRTMP.json", n_top_words=5, n_features=1000):

    import numpy as np
    results, labels = get_results(results_folder)

    H = np.array(results[example]["Representation"])
    r, d = H.shape 

    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    # Use the TFIDF vectorizer to get the dataset
    data_path = os.path.join("datasets", "news20")
    news20 = fetch_20newsgroups(subset="all", data_home=data_path, remove=('headers', 'footers', 'quotes'))

    X, y = news20.data, news20.target

    vectorizer = TfidfVectorizer(max_features=n_features)
    X = vectorizer.fit_transform(X)
    feature_names = vectorizer.get_feature_names_out()

    # Plot on this example https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py
    fig, axes = plt.subplots(n_col, n_row, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    
    for topic_idx, topic in enumerate(H):
        top_features_ind = topic.argsort()[-n_top_words:] # get indices of top features.
        top_features = feature_names[top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(FILE_KEY[example], fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    return fig

def plot_image_representation(folder, n_col, n_row, im_size=(64,64), example="NewtonMRTMP.json"):
    """
    Plots the low rank representation images.
    """
    import numpy as np

    results, labels = get_results(folder)
    H = np.array(results[example]["Representation"])

    fig = plot_gallery(FILE_KEY[example], H, n_col, n_row, image_shape=im_size)
    return fig

def plot_gallery(title, images, n_col=5, n_row=2, image_shape=(28,28), cmap=plt.cm.gray):
    """
    Based on code from https://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html#sphx-glr-auto-examples-decomposition-plot-faces-decomposition-py
    """
    fig, axs = plt.subplots(
        nrows=n_row,
        ncols=n_col,
        figsize=(2.0 * n_col, 2.3 * n_row),
        facecolor="white",
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, hspace=0, wspace=0)
    fig.set_edgecolor("black")
    if title is not None:
        fig.suptitle(title, size=16)
    for ax, vec in zip(axs.flat, images):
        vmax = vec.max()
        im = ax.imshow(
            vec.reshape(image_shape),
            cmap=cmap,
            interpolation="nearest",
            vmin=0,
            vmax=vmax,
        )
        ax.axis("off")

    #fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.99, aspect=40, pad=0.01)
    #plt.show()

    return fig

def iteration_oracle_count(results_folder, ops_weighting=None):
    """
    Outputs the iteration and oracle counts for each method in the results. 
    """
    results, label = get_results(results_folder)
    
    for r, l in zip(results.values(), label):
        iter_count = len(r["ops"])
        oracle_calls = []
        for t in r["ops"]:
            oracle_calls.append(1+sum([i*j for i, j in zip(ops_weighting,t) ]))

        oracle_count = oracle_calls[-1]
        time_taken = r["t"][-1]
        flag = r["termination flag"]
        print("Method {} terminated with flag {} in {} iters ({} oracle calls) and {:.2f} sec".format(l, flag, iter_count, oracle_count, time_taken))

def result_accuracy(results_folder):
    """
    Gives the accuracy and sparsity achieved by each method. 
    """
    results, label = get_results(results_folder)

    for r, l in zip(results.values(), label):
        print("{} achieved accuracy of {:.2f} with sparsity {:.2f}%".format(l, r["Train accuracy"], r["Sparsity"]))

def result_loss_sparity(results_folder):
    """
    Give the loss and sparsity achieved by each method. 
    """
    results, label = get_results(results_folder)

    for r, l in zip(results.values(), label):
        print("{} achieved loss of {:.6f} with sparsity {:.2f}%".format(l, r["loss"], r["sparsity"]))

def save_plot(plot, name):
    FOLDER = os.path.join(os.getcwd(), "plots", name+".pdf")
    plot.savefig(FOLDER, format="pdf")