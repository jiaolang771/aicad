
from keras.models import Model
from keras.layers import Input, LeakyReLU, concatenate, add, Add, Conv3D, Activation, Dropout 
from keras.layers import BatchNormalization, MaxPooling3D, Conv3DTranspose, AveragePooling3D, ZeroPadding3D
from keras.optimizers import RMSprop, Adam, SGD
from keras import backend as K


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)



def get_3D_DWMA_unet(img_rows, img_cols, img_depth):
    
    inputs = Input((img_rows, img_cols, img_depth, 1))
# ==========================================================##    
    # conv block 1
    conv1 = Conv3D(32, (3, 3, 3), padding='same')(inputs)
    conv1 = BatchNormalization(axis=4)(conv1)
    conv1 = LeakyReLU()(conv1)
    # residual block 1
    res1 = Conv3D(32, (3, 3, 3), padding='same')(conv1)   
    res1 = BatchNormalization(axis=4)(res1)
    res1 = LeakyReLU()(res1)
    res1 = Conv3D(32, (3, 3, 3), padding='same')(res1)   
    res1 = BatchNormalization(axis=4)(res1)
    res1 = LeakyReLU()(res1)
    res1 = Dropout(0.2)(res1)   
    res1 = Add()([conv1, res1])
    down1 = MaxPooling3D(pool_size=(2, 2, 2))(res1)
# ==========================================================##
    # conv block 2
    conv2 = Conv3D(64, (3, 3, 3), padding='same')(down1)
    conv2 = BatchNormalization(axis=4)(conv2)
    conv2 = LeakyReLU()(conv2)
    # residual block 1
    res2 = Conv3D(64, (3, 3, 3), padding='same')(conv2)   
    res2 = BatchNormalization(axis=4)(res2)
    res2 = LeakyReLU()(res2)
    res2 = Conv3D(64, (3, 3, 3), padding='same')(res2)   
    res2 = BatchNormalization(axis=4)(res2)
    res2 = LeakyReLU()(res2)
    res2 = Dropout(0.2)(res2)   
    res2 = Add()([conv2, res2])
    down2 = MaxPooling3D(pool_size=(2, 2, 2))(res2)    
# ==========================================================##
    # conv block 3
    conv3 = Conv3D(128, (3, 3, 3), padding='same')(down2)
    conv3 = BatchNormalization(axis=4)(conv3)
    conv3 = LeakyReLU()(conv3)
    # residual block 1
    res3 = Conv3D(128, (3, 3, 3), padding='same')(conv3)   
    res3 = BatchNormalization(axis=4)(res3)
    res3 = LeakyReLU()(res3)
    res3 = Conv3D(128, (3, 3, 3), padding='same')(res3)   
    res3 = BatchNormalization(axis=4)(res3)
    res3 = LeakyReLU()(res3)
    res3 = Dropout(0.2)(res3)   
    res3 = Add()([conv3, res3])
    down3 = MaxPooling3D(pool_size=(2, 2, 2))(res3)    
# ==========================================================##
    # conv block center
    conv_c = Conv3D(256, (3, 3, 3), padding='same')(down3)
    conv_c = BatchNormalization(axis=4)(conv_c)
    conv_c = LeakyReLU()(conv_c)
    # residual block center
    res_c = Conv3D(256, (3, 3, 3), padding='same')(conv_c)   
    res_c = BatchNormalization(axis=4)(res_c)
    res_c = LeakyReLU()(res_c)
    res_c = Conv3D(256, (3, 3, 3), padding='same')(res_c)   
    res_c = BatchNormalization(axis=4)(res_c)
    res_c = LeakyReLU()(res_c)
    res_c = Dropout(0.2)(res_c)   
    center = Add()([conv_c, res_c])
# ==========================================================##
    up4 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(center), res3], axis=4)
    # conv block 4
    conv4 = Conv3D(128, (3, 3, 3), padding='same')(up4)
    conv4 = BatchNormalization(axis=4)(conv4)
    conv4 = LeakyReLU()(conv4)
    # residual block 4
    res4 = Conv3D(128, (3, 3, 3), padding='same')(conv4) 
    res4 = BatchNormalization(axis=4)(res4)
    res4 = LeakyReLU()(res4)
    res4 = Conv3D(128, (3, 3, 3), padding='same')(res4)   
    res4 = BatchNormalization(axis=4)(res4)
    res4 = LeakyReLU()(res4)
    res4 = Dropout(0.2)(res4)     
# ==========================================================##    
    up5 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(res4), res2], axis=4)
    # conv block 5
    conv5 = Conv3D(64, (3, 3, 3), padding='same')(up5)
    conv5 = BatchNormalization(axis=4)(conv5)
    conv5 = LeakyReLU()(conv5)
    # residual block 6
    res5 = Conv3D(64, (3, 3, 3), padding='same')(conv5) 
    res5 = BatchNormalization(axis=4)(res5)
    res5 = LeakyReLU()(res5)
    res5 = Conv3D(64, (3, 3, 3), padding='same')(res5)   
    res5 = BatchNormalization(axis=4)(res5)
    res5 = LeakyReLU()(res5)
    res5 = Dropout(0.2)(res5) 
# ==========================================================##
    up6 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(res5), res1], axis=4)
    # conv block 6
    conv6 = Conv3D(32, (3, 3, 3), padding='same')(up6)
    conv6 = BatchNormalization(axis=4)(conv6)
    conv6 = LeakyReLU()(conv6)
    # residual block 6
    res6 = Conv3D(32, (3, 3, 3), padding='same')(conv6) 
    res6 = BatchNormalization(axis=4)(res6)
    res6 = LeakyReLU()(res6)
    res6 = Conv3D(32, (3, 3, 3), padding='same')(res6)   
    res6 = BatchNormalization(axis=4)(res6)
    res6 = LeakyReLU()(res6)
    res6 = Dropout(0.2)(res6) 
# ==========================================================##
    conv7 = Conv3D(1, (1, 1, 1), activation='sigmoid')(res6)

    model = Model(inputs=[inputs], outputs=[conv7])

#    model.summary()
#    for layer in model.layers:
#        print(layer.output_shape)
    #plot_model(model, to_file='model.png')

    model.compile(optimizer=Adam(lr=1e-2, beta_1=0.9, beta_2=0.999, 
                                 epsilon=1e-8, decay=1e-6),  
                  loss= dice_coef_loss, 
                  metrics= [dice_coef])

    return model