import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import anvil.server
import anvil.media
import anvil.mpl_util
anvil.server.connect("server_6YD55HP45RX7V6UHWRRAST3G-WOK7HWGMAN5O7RYP")

CNNmodel = tf.keras.models.load_model('Models/CNNmodel_mnist')
ViTmodel = tf.keras.models.load_model('Models/ViT_mnist')

@anvil.server.callable
def check_input(file):
  with anvil.media.TempFile(file) as filename:
    pixels = pd.read_csv(filename, header=None,index_col=False)
  if (((pixels >= 0)&(pixels <= 255))|((pixels >= 0)&(pixels <= 1))).all().all():
    if pixels.shape == (28,28):
      return 1
    else:
      return 2
  else:
    return 3

@anvil.server.callable
def print_image(file):
  with anvil.media.TempFile(file) as filename:
    pixels = pd.read_csv(filename, header=None,index_col=False)

  if (pixels > 1).any().any():
    pixels = pixels/255

  plt.axis('off')
  plt.pcolor(pixels,cmap='gray')

  return anvil.mpl_util.plot_image()

@anvil.server.callable
def predict_CNN(file):
  with anvil.media.TempFile(file) as filename:
    pixels = pd.read_csv(filename, header=None,index_col=False)

  if (pixels > 1).any().any():
    pixels = pixels/255
  pixels = np.reshape(pixels.values,(28,28,1))
  pixels = np.expand_dims(pixels, axis=0)

  pred_probs2 = CNNmodel.predict(pixels)
  ind = np.argpartition(pred_probs2[0], -5)[-5:]
  top5 = np.flip(ind[np.argsort(pred_probs2[0][ind])])
  values = pred_probs2[0][top5]
  pred2 = np.argmax(pred_probs2, axis=1)

  fig, ax1 = plt.subplots(figsize=(5, 2.5), layout='constrained')
  barchar = ax1.barh(top5.astype(str), values, align='center',color='#B0C6CE')
  ax1.invert_yaxis()
  barchar[0].set_color('#15616D')
  plt.show()
  return anvil.mpl_util.plot_image(), pred2[0]
    
@anvil.server.callable
def predict_ViT(file):
  with anvil.media.TempFile(file) as filename:
    pixels = pd.read_csv(filename, header=None,index_col=False)

  if (pixels > 1).any().any():
    pixels = pixels/255
  
  pos_feed = np.array([list(range(4*4))]*1)
  pixels = np.asarray(pixels)
  pixels_ravel = np.zeros((16,49))
  ind = 0
  for row in range(4):
      for col in range(4):
          pixels_ravel[ind] = pixels[(row*7):((row+1)*7),(col*7):((col+1)*7)].ravel()
          ind += 1

  pred_probs2 = ViTmodel.predict([np.array([pixels_ravel]), pos_feed])
  ind = np.argpartition(pred_probs2[0], -5)[-5:]
  top5 = np.flip(ind[np.argsort(pred_probs2[0][ind])])
  values = pred_probs2[0][top5]
  pred2 = np.argmax(pred_probs2, axis=1)

  fig, ax1 = plt.subplots(figsize=(5, 2.5), layout='constrained')
  barchar = ax1.barh(top5.astype(str), values, align='center',color='#B0C6CE')
  ax1.invert_yaxis()
  barchar[0].set_color('#15616D')
  plt.show()
    
  return anvil.mpl_util.plot_image(),pred2[0]



anvil.server.wait_forever()