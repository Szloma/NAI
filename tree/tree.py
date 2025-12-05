import argparse

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree


'''

Dawid Litwiński, Łukasz Kapkowski

classifiers: decision tree, svm to classify data for datasets wine.csv, and ads.csv

example of running with parameters:
python tree.py gini 5 rbf 1.0 32 150000
python tree.py gini 15 rbf 0.3 32 150000
python tree.py entropy 5 linear 1.0 32 150000
python tree.py entropy 15 linear 0.3 32 150000

'''

def classifiers(X_train, y_train, X_train_scaled, criterion, max_depth, kernel, regularisation_c):
    decision_tree = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        random_state=11
    )
    decision_tree.fit(X_train, y_train)

    svm_classifier = svm.SVC(
        kernel=kernel,  # rbf radial basis function (default)
        C=regularisation_c,  # regularisation strength
        gamma='scale',  # automatic scaling of the RBF width
        random_state=11
    )

    svm_classifier.fit(X_train_scaled, y_train)

    return decision_tree, svm_classifier


def main():
    df = pd.read_csv('wine.csv', sep=';', quotechar='"')
    X = df.drop(columns='quality')
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=1,
        stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    criterion = 'gini'
    max_depth = 5
    kernel = 'rbf'
    regularisation_c = 1.04


    parser = argparse.ArgumentParser(description="series of strings.")
    parser.add_argument("strings", nargs='*', type=str)

    args = parser.parse_args()

    if args.strings:
        criterion = args.strings[0]
        max_depth = int(args.strings[1])
        kernel = args.strings[2]
        regularisation_c = float(args.strings[3])
        feature1 = int(args.strings[4])
        feature2 = int(args.strings[5])


    # X_train_scaled = X_train
    # X_test_scaled = X_test
    if len(args.strings) < 1:
        decision_tree, svm_classifier = classifiers(X_train, y_train, X_train_scaled, criterion, max_depth, kernel, regularisation_c)

        evaluate(decision_tree, X_train, X_test, y_train, y_test, 'Decision Tree')
        evaluate(svm_classifier, X_train_scaled, X_test_scaled, y_train, y_test, 'SVM')

        feature_names = X.columns.tolist()
        class_names = [str(c) for c in sorted(y.unique())]

        plot_decision_tree(decision_tree,
                           feature_names=feature_names,
                           class_names=class_names)


    df = pd.read_csv('ads.csv', sep=',', quotechar='"')
    X = df.drop(columns='Purchased')
    y = df['Purchased']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=11,
        stratify=y
    )
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    decision_tree, svm_classifier = classifiers(X_train, y_train, X_train_scaled, criterion, max_depth, kernel, regularisation_c)

    evaluate(decision_tree, X_train, X_test, y_train, y_test, 'Decision Tree 2nd')
    evaluate(svm_classifier, X_train_scaled, X_test_scaled, y_train, y_test, 'SVM 2nd')
    if args.strings:
        X_new = pd.DataFrame([[feature1, feature2]], columns=["Age", "EstimatedSalary"])
        print("decission tree for given input= ",decision_tree.predict(X_new))
        print("svm for given input= ",svm_classifier.predict([[feature1,feature2]]))



def evaluate(model, X_tr, X_te, y_tr, y_te, name):

    y_pred_train = model.predict(X_tr)
    y_pred_test  = model.predict(X_te)

    acc_train = accuracy_score(y_tr, y_pred_train)
    acc_test  = accuracy_score(y_te, y_pred_test)

    print(f'=== {name} ===')
    print('Training accuracy :', accuracy_score(y_tr, y_pred_train))
    print('Test accuracy     :', accuracy_score(y_te, y_pred_test))


    fig, ax = plt.subplots(figsize=(4, 3))

    labels = ["Train", "Test"]  # x‑axis labels
    scores = [acc_train, acc_test]  # heights of the bars
    colors = ["blue", "green"]
    bars = ax.bar(labels, scores, color=colors)

    ax.set_ylim(0, 1)   #y‑axis from 0 to 1

    #axis titles
    ax.set_ylabel("Accuracy")  # y‑label
    ax.set_title(f"{name} – Accuracy")  # chart title

    #accuracy on top of each bar
    for bar in bars:
        height = bar.get_height()  # the numeric value of the bar
        ax.text(bar.get_x() + bar.get_width() / 2,  # x‑position (center of bar)
                height,  # y‑position (top of bar)
                f"{height:.2f}",  # text to display
                ha='center', va='bottom')  # centre‑align horizontally, bottom‑align vertically

    plt.tight_layout()
    plt.show()


def plot_decision_tree(model, feature_names, class_names):
    plt.figure(figsize=(140, 25))
    tree.plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,  # colour class purity
        impurity=True,
        rounded=True,
        fontsize=12
    )
    plt.title("Decision Tree")
    plt.show()

if __name__ == '__main__':
    main()