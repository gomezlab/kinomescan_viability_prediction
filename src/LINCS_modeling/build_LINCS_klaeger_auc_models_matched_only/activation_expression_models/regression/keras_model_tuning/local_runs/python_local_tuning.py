data = r.data
feats = r.cors

feat_counts = [100,200,300,400,500,1000,1500,2000,3000,4000,5000]

for f in feat_counts:

    feats_list = feats['feature'][:f].tolist()
    feats_list.append('auc')
    data = data[feats_list]

    exclude = ['auc']
    y = data['auc']
    X = data.drop(exclude, axis=1)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    # In[ ]:
    input_shape = [X_train.shape[1]]

    def build_model(n_hidden=1, n_neurons=100, dropout=0.4, learning_rate=3e-3):
        model = keras.models.Sequential()
        model.add(keras.layers.Flatten(input_shape=input_shape))
        for layer in range(n_hidden):
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Dense(n_neurons))
            model.add(keras.layers.Dropout(dropout))
            model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(1, activation='linear'))

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(loss="mean_squared_error", metrics=['RootMeanSquaredError'], optimizer=optimizer)
        return model


    # In[ ]:


    keras_clf = keras.wrappers.scikit_learn.KerasRegressor(build_model)


    # In[ ]:


    param_distribs = {
        "n_hidden": [1, 2, 4, 6],
        "n_neurons": [200, 600, 1000],
        "dropout": [0.2, 0.4, 0.6],
        "learning_rate": [3e-3],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    rnd_search_cv = GridSearchCV(keras_clf, param_distribs, cv=10, verbose=2, scoring='r2', refit=True)
    rnd_search_cv.fit(X_train, y_train, epochs=100, batch_size=512,
                    validation_data=(X_valid, y_valid),
                    callbacks=[keras.callbacks.EarlyStopping(patience=15, min_delta=1e-4, restore_best_weights=True)])
    
    best_model = rnd_search_cv.best_estimator_
    r2score = (cross_val_score(best_model, X, y, cv=10, scoring='r2')).mean()
    mse = (cross_val_score(best_model, X, y, cv=10, scoring='neg_mean_squared_error')).mean()
    
    
    preds = best_model.predict(X_valid)
    r2score = r2_score(y_valid, preds)
    mse = mean_squared_error(y_valid, preds, squared=False)
    scores = pd.DataFrame({'r2': [r2score], 'mse': [mse]})
    scores.to_csv('results/scores_keras_' + str(f) + '.csv', index=False)

    results = pd.DataFrame(rnd_search_cv.cv_results_)
  	results.sort_values(by='rank_test_score').to_csv('results/results/PRISM_LINCS_klaeger_models_auc/activation_expression/regression/keras/results/hps_keras_' + str(f) + '.csv')
