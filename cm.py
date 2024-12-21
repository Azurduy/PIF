from cm_utils import *


# calculate correlation matrix
def get_cm(df_pseudo_obs, headers, correlation_indicator, pdf_plot=False):
    cm_headers = headers[:-2]
    cm = pd.DataFrame(columns=cm_headers)
    for i, var1 in enumerate(cm_headers):
        cm.at[i, var1] = 1.0
        if i < len(cm_headers)-1:
            for j, var2 in enumerate(cm_headers[i+1:]):
                pseudo_obs1, pseudo_obs2 = df_pseudo_obs[var1], df_pseudo_obs[var2]

                if pdf_plot:
                    # scatterplot of pseudo-observations
                    # necessary for multi-scatterplot PDF
                    fig, ax = plt.subplots()
                    ax.scatter(pseudo_obs1, pseudo_obs2, marker=".")
                    ax.set_xlabel(var1, labelpad=5)
                    ax.set_ylabel(var2, labelpad=5)
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.0])
                    # plt.show()
                
                if correlation_indicator == "spearman":
                    corr = spearmanr(pseudo_obs1, pseudo_obs2)[0]
                if correlation_indicator == "kendall":
                    corr = kendalltau(pseudo_obs1, pseudo_obs2)[0]
                if correlation_indicator == "spearman_numerical":
                    corr = numerical_correlation(pseudo_obs1, pseudo_obs2, [var1, var2], type="spearman")
                if correlation_indicator == "absolute_distance_numerical":
                    corr = numerical_correlation(pseudo_obs1, pseudo_obs2, [var1, var2], type="abs_distance")
                if correlation_indicator == "chatterjee":
                    corr = xicorr(pseudo_obs1, pseudo_obs2)[0]
                if correlation_indicator == "MIC":
                    mine = MINE(alpha=0.6, c=15, est="mic_approx")
                    mine.compute_score(pseudo_obs1, pseudo_obs2)
                    corr = mine.mic()
                if correlation_indicator == "TIC":
                    mine = MINE(alpha=0.6, c=15, est="mic_approx")
                    mine.compute_score(pseudo_obs1, pseudo_obs2)
                    corr = mine.tic()
                # too many nans here
                # corr = mutual_info_regression(np.expand_dims(pseudo_obs1.to_numpy(), axis=1), pseudo_obs2.to_numpy())
                cm.at[i+j+1, var1] = corr
                cm.at[i, var2] = corr
    return cm


if __name__ == "__main__":

    #########
    # INPUT #
    #########
    path = "C:/Users/flowb/OneDrive/TWVL/tesis_grothe/ai_cbv/new/inout/nn_outputs.csv"
    # scatterplots with pseudo-observations of all possible variable combinations
    plot_copula_to_pdf = False
    # choose one
    correlation_indicator = "kendall"
    correlation_indicator = "spearman"
    correlation_indicator = "chatterjee"
    correlation_indicator = "MIC"
    # correlation_indicator = "TIC" # many NaNs
    # correlation_indicator = "spearman_numerical"
    correlation_indicator = "absolute_distance_numerical"

    df_pseudo_obs, headers = get_pseudo_observations(path)

    # test of correlation values 
    # select test-data of 2 variables
    pseudo_obs1, pseudo_obs2 = df_pseudo_obs[headers[0]].to_numpy(), df_pseudo_obs[headers[1]].to_numpy()
    test_correlation_indicators(pseudo_obs1, pseudo_obs2, headers[:2])

    cm = get_cm(df_pseudo_obs, headers, correlation_indicator, pdf_plot=plot_copula_to_pdf)
    if plot_copula_to_pdf: plot_to_pdf("./inout/copulae.pdf")
                    
    # replace weired values with np.nan
    cm.where(cm<=1.0, inplace=True)
    cm.where(cm>=-1.0, inplace=True)
    print("correlation_indicator: ", correlation_indicator)
    print("correlation matrix: \n", cm)
    print("")
    cm.to_csv("./inout/iris_CM_new.csv", header=False, index=None)

    # compare with original results
    path = "C:/Users/flowb/OneDrive/TWVL/tesis_grothe/ai_cbv/new/inout/iris_CM.csv"
    df_cm_original = pd.read_csv(path, header=None)
    print("original correlation matrix: \n", df_cm_original)
    print("")
    cm_vals = cm.to_numpy().ravel()
    cm_original_vals = df_cm_original.to_numpy().ravel()
    # print("spearmanr(cm_vals, cm_original_vals): ", spearmanr(cm_vals, cm_original_vals))