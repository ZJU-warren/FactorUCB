import matplotlib.pyplot as plt
from Tools.DataTools import *
import DataSetLink as DLSet


def show_img(data_link):
    df = load_data(data_link)[:5000]
    print(df.columns)
    x = df['n_round'].values
    y = df['ratio'].values

    plt.figure()
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    show_img(DLSet.result_link % 'LinUCB')
    show_img(DLSet.result_link % 'hLinUCB')
