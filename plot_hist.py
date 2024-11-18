import matplotlib.pyplot as plt

def plot_hist(matr, cl, k):

    # Dati
    labels = ['True Positive', 'False Positive', 'False Negative', 'True Negative']  # Etichette
    values = [matr[0], matr[1], matr[2], matr[3]]  # Valori da plottare

    # Creazione dell'istogramma
    plt.bar(labels, values)

    # Personalizzazione del grafico
    plt.title(f"Confusion matrix for class {cl} using a k equal to {k}")  # Titolo
    plt.xlabel('Values') 
    plt.ylabel('Cardinality') 

    # Mostrare il grafico
    plt.show()
