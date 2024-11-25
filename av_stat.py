from split_dataframe import *
from knn import *
from class_analysis import *
from stats import *
from compute_binary import *

def average_stats(classes, feat, k_values, cl, iterations):
    aggregated_stats0 = []
    aggregated_stats1 = []
    aggregated_stats2 = []
    
    for k in k_values:
        for class_index in range(len(cl)):
            stats_list = []
            
            # Iterating for the iterations required
            for _ in range(iterations):
                # Dividig the dataset
                tr_cl, tr_feat, test_cl, test_feat = split_matrix_random(classes, feat)

                # Creating the binary matrix for the current class
                binary_tr = compute_binary(tr_cl, cl)
                binary_test = compute_binary(test_cl, cl)

                # Calculate the statistics for the current value of k
                stats_list.append(stats(tr_feat, binary_tr[class_index], test_feat, binary_test[class_index], k=[k]))
            
            if(class_index == 0): 
                aggregated_stats0.append(compute_stat_stat(stats_list))
            elif(class_index == 1):
                aggregated_stats1.append(compute_stat_stat(stats_list))
            else:
                aggregated_stats2.append(compute_stat_stat(stats_list))
        
        plot_statistics(aggregated_stats0[0], 0, k)
        plot_statistics(aggregated_stats1[0], 1, k)
        plot_statistics(aggregated_stats2[0], 2, k)
            
    
        


def plot_statistics(matrix, cl, k):
    if not isinstance(matrix, list):
        raise ValueError("Input to plot_statistics must be a matrix (list of lists).")
    matr = [[round(el, 2) for el in row] for row in matrix]
    row_labels = ["Average", "Median", "Mode", "Standard dev", "0.25 percentile", "0.75 percentile"]
    col_labels = ["Sensitivity", "Specifity", "Precision", "F1_score", "Accuracy"]

    fig, ax = plt.subplots(figsize=(len(col_labels) * 1.5, len(row_labels) * 0.8))
    ax.axis("tight")
    ax.axis("off")

    tab = ax.table(
        cellText=matr,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )

    for (i, j), cell in tab.get_celld().items():
        if i == 0 or j == -1:  
            cell.set_text_props(color="black")
            cell.set_facecolor("#fffacd") 

    tab.auto_set_font_size(False)
    tab.set_fontsize(12)
    tab.auto_set_column_width(col_labels)
    ax.set_title(f"Statistics for class {cl} and k={k}", fontsize=14, weight="bold", pad=20)

    plt.show()
