def normalize_test(x_test, x_min, x_max):
    try:
        x_norm = (x_test - x_min) / (x_max - x_min)
        return x_norm,
    except Exception as e:
        print("Error normalizing test set: ", e)
        exit()

# Sauvegarde de ce qu'on avait fait vendredi dans train.py:
# Pas besoin de faire de prediction dans ce script,
# ca sera fait dans logreg_predict.py.

# polynomial test
# normalize test

# prediction = np.empty((x_train_norm.shape[0], 0))
# for house in houses:
#     mlr.theta = model[house]
#     y_hat = mlr.predict_(x_train_norm)
#     prediction = np.concatenate((prediction, y_hat), axis=1)

# # Argmax sur les predictions pour trouver la maison la plus probable
# # pour chaque ligne du dataset d'entrainement
# y_hat = np.argmax(prediction, axis=1)
# # On remplace les indices par les noms des maisons
# y_hat = np.array([houses[i] for i in y_hat])
# # On compare les predictions avec les vraies valeurs
# y_train = y_train.to_numpy().reshape(-1)

# # Confusion matrix
# mlr.confusion_matrix_(
#     y_train,
#     y_hat,
#     labels=houses,
#     df_option=True,
#     display=True
# )

# print("\nAccuracy on training set:")
# accuracy = mlr.accuracy_score_(y_hat, y_train)
# print(accuracy)