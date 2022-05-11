from pprint import pprint
import random
import neurolab as nl
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def formula(x, y):
    return x > y


def get_random_number():
    return random.randint(1, 300)


dataset_x = [[get_random_number(), get_random_number()] for _ in range(300)]
dataset_y = [[formula(a, b)] for a, b in dataset_x]

plot_x = []
plot_y = []


def reports(expected, pred):
    return classification_report(expected, pred, output_dict=True, zero_division=0)


for count in range(100):

    net = nl.net.newp([[-1, 2], [0, 2]], 1)

    X_train, X_test, y_train, y_test = train_test_split(
        dataset_x, dataset_y, test_size=0.25
    )

    error = net.train(X_train, y_train, epochs=100, show=10, lr=0.1)

    predictions = net.sim(X_test)

    f = nl.error.MSE()
    test_error = f(y_test, predictions)

    plot_x.append(count)
    plot_y.append(test_error)

    clf_report = reports(y_test, predictions)

    print(count)
    pprint(clf_report)


fig, ax = plt.subplots()
ax.plot(plot_x, plot_y)

ax.set(xlabel="iteration", ylabel="MSE", title="MSE plot")
ax.grid()

fig.savefig("plot1.png")
