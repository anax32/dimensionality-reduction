def train (params):
  from keras.models import Model

  inputs, outputs = params["constructor_fn"] ()

  m = Model (inputs=inputs, outputs=outputs)

  print (m.summary ())

  m.compile(loss=params["loss"], optimizer = params["optimizer"])

  history = m.fit(params["x_train"], params["x_train"],
                  batch_size=params["batch_size"],
                  epochs=params["epochs"],
                  verbose=1,
                  shuffle=True,
                  validation_data=(params["x_test"], params["x_test"]))

  m.save (params["model_filename"])

  return m
