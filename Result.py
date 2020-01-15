import matplotlib.pyplot as plt
from Tools.DataTools import *
import DataSetLink as DLSet


def show_img(result):
    plt.figure()
    for each in result.keys():
        plt.plot(np.arange(result[each].shape[0]), result[each], label=each)
    plt.title("Training Loss and Accuracy")
    plt.xlabel("# round")
    plt.ylabel("ratio")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    result = {}
    for each in ['LinUCB', 'hLinUCB', 'coLin', 'factorUCB']:
        result[each] = load_data(DLSet.result_link % each)['ratio'][10000:].values
    show_img(result)
