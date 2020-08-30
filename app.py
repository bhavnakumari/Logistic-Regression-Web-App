import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc,confusion_matrix, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

from sklearn.datasets import make_classification

def main():
    st.write("Authors:")
    st.write("[Bhavna Kumari](https://www.linkedin.com/in/bhavna-kumari-4b671a182/)")
    st.write("[Anjali](https://www.linkedin.com/in/anjali-kumari-687956156/)")
    st.title("LOGISTIC REGRESSION MODEL")
    st.sidebar.title("Logistic Regression Web App")
    st.markdown("Official Scikit Learn Documentation of Logistic Regression [Click Here] (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)")    

    @st.cache(persist=True)
    def load_data(random_var):
        X1, Y1 = make_classification(n_samples=500, n_features=2, n_redundant=0,
                                     n_informative=1, n_clusters_per_class=1, random_state=random_var)
        col_x = []
        col_y = []
        for i in range(len(X1)):
            col_x.append(X1[i][0])
            col_y.append(X1[i][1])
        data = pd.DataFrame({' X ': col_x, ' Y ': col_y, 'Classification': Y1})
        return data

    # @st.cache(persist=True)
    def split(df):
        y = df.Classification.values
        x = df.drop(columns=['Classification']).values
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list, model, trained_model, x_train, y_train):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            st.write("It is often used to describe the performance of a classifier and it helps to evaluate the number of True +ve (TP), False +ve (FP), False -ve (FN) and True -ve (TN)")
            st.write("ACCURACY=(TP+TN)/Total")
            def plotcm(cm, ax, title):
                sns.heatmap(cm, ax=ax, annot=True,fmt='d',annot_kws={'size':20},yticklabels=3)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix for {} '.format(title))

            fig, subplt = plt.subplots(1, 2, figsize=(10, 4))
            cm1 = confusion_matrix(y_train, trained_model.predict(x_train))
            ax1 = subplt[0]
            plotcm(cm1, ax1, 'Train data')
            cm2 = confusion_matrix(y_test, trained_model.predict(x_test))
            ax2 = subplt[1]
            plotcm(cm2, ax2, 'Test data')
            st.pyplot()


        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            st.write("Receiver Operating Characteristics. It presents a sensitivity pair corresponding to a particular decision threshold. ")
            st.write("SENSITIVITY / RECALL = TP/(TP + FN)")
            st.subheader("Values: ")
            st.write("No Discrimination = 0.5")
            st.write("Acceptable = 0.7-0.8")
            st.write("Excellent = 0.8-0.9")
            st.write("Outstanding = 0.9")
            
            


            train_fpr, train_tpr, thresholds = roc_curve(y_train, trained_model.predict_proba(x_train)[:, 1])
            test_fpr, test_tpr, thresholds = roc_curve(y_test, trained_model.predict_proba(x_test)[:, 1])
            plt.plot(train_fpr, train_tpr, label="Train Area Under Curve =" + str(auc(train_fpr, train_tpr)))
            plt.plot(test_fpr, test_tpr, label="Test Area Under Curve =" + str(auc(test_fpr, test_tpr)))
            plt.legend()
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC for Train and Test data with best_fit")
            plt.grid()
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            st.write("It presents the Relationship between Precision (Positive Predicted Values) & Recall (Sensitivity) for every possible cut. It is highly recommended to compare tests.")
            st.write("PRECISION = TP/predicted yes")
            st.subheader("Values: ")
            st.write("Good Precision = 1.0")
            st.write("Threshold = >0.5")




            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()

        if 'Decision Boundary' in metrics_list:
            st.subheader("Decision Boundary")
            st.write("Decision boundary helps to differentiate probabilities into positive class and negative class.")
            xaxis = np.arange(start=x_train[:, 0].min(
            ) - 1, stop=x_train[:, 0].max() + 1, step=0.01)
            yaxis = np.arange(start=x_train[:, 1].min(
            ) - 1, stop=x_train[:, 1].max() + 1, step=0.01)
            xx, yy = np.meshgrid(xaxis, yaxis)

            in_array = np.array([xx.ravel(), yy.ravel()]).T
            labels = trained_model.predict(in_array)

            plt.contourf(xx, yy, labels.reshape(
                xx.shape), alpha=0.5, cmap="RdBu")
            plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap="RdBu", vmin=-.2, vmax=1.2,
                        edgecolor="white", linewidth=1)
            plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap="RdBu", marker ='+', vmin=-.2, vmax=1.2,
                        edgecolor="white", linewidth=1)

            st.pyplot()

    if st.sidebar.checkbox("Show Raw Data", True):
        lol = st.sidebar.selectbox("Choose Dataset", (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20))
        df = load_data(lol)
        data_table = st.empty()
        data_table.dataframe(df)

        if st.sidebar.checkbox("Logistic Regression", False):
            st.sidebar.subheader("Model Hyperparameters")
            C = st.sidebar.number_input(
                "C (Regularization parameter)", 1.0, 10000.0, step=50.0, key='C_LR')
            st.write("C is a float, (default=1.0) Inverse of regularization strength must be positive")


            max_iter = st.sidebar.slider(
                "Maximum number of iterations", 0, 10000, key='max_iter')

            st.write("Maximum Iterations is used to fit the data. It is an int ,default=100.")    

            st.sidebar.subheader("Choose Solver ")
            solverr = st.sidebar.selectbox(
                "Solver", ("liblinear", "newton-cg", "lbfgs", "sag", "saga"))

            metrics = st.sidebar.multiselect("What metrics to plot?", (
                'Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve', 'Decision Boundary'))

            class_names = ['0', '1']
            x_train, x_test, y_train, y_test = split(df)

            # @st.cache(persist=True)
            def common(s, p, C, max_iter):
                st.subheader("Logistic Regression Results")
                model = LogisticRegression(
                    C=C, penalty=p, max_iter=max_iter, solver=s)
                trained_model = model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy: ", accuracy.round(2))

                st.write("Recall: ", recall_score(
                    y_test, y_pred, labels=class_names).round(2))
                st.write("Precision: ", precision_score(
                    y_test, y_pred, labels=class_names).round(2))                    
                plot_metrics(metrics, model, trained_model,
                             x_train, y_train)

            if (solverr == 'newton-cg'):
                st.subheader(
                    "For multiclass problems, ‘newton-cg’handle multinomial loss .‘newton-cg’ handle only l2 penalty or no")
                if st.sidebar.button("Classify", key='c_1'):
                    common('newton-cg', 'l2', C, max_iter)

            if (solverr == 'lbfgs'):
                st.subheader(
                    "For multiclass problems, ‘lbfgs’ handle multinomial loss .‘lbfgs’ handle only l2 penalty or no")
                if st.sidebar.button("Classify", key='c_2'):
                    common('lbfgs', 'l2', C, max_iter)

            if (solverr == 'sag'):
                st.subheader(
                    "For multiclass problems, ‘sag’ handle multinomial loss .‘sag’ handle only l2 penalty or no")
                if st.sidebar.button("Classify", key='c_3'):
                    common('sag', 'l2', C, max_iter)

            if solverr == 'liblinear':
                st.subheader(
                    " For small datasets, ‘liblinear’ is a good choice and the default solver. It is limited to one-versus-rest schemes. It handle l1 penalty")
                if st.sidebar.button("Classify", key='c_4'):
                    common('liblinear', 'l1', C, max_iter)

            if solverr == 'saga':
                st.subheader(
                    "For multiclass problems, ‘saga’ handle multinomial loss .‘saga’ handle l1, l2, elastic penalty . ‘saga’ are faster for large datasets. ")
                penaltyy = st.sidebar.radio(
                    "Choose Penalty", ('l1', 'l2'), key='pe')

                if penaltyy == 'l1':
                    st.markdown("For l1 Penalty")
                    if st.sidebar.button("Classify", key='c_5'):
                        common('saga', 'l1', C, max_iter)

                if penaltyy == 'l2':
                    st.markdown("For l2 Penalty")
                    if st.sidebar.button("Classify", key='c_6'):
                        common('saga', 'l2', C, max_iter)



if __name__ == '__main__':
    main()